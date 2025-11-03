import os, csv
import talib
import pandas
import duckdb
import yfinance as yf
from flask import Flask, request, render_template
from markupsafe import escape
from patterns import candlestick_patterns
from custom_patterns import (custom_chart_patterns, detect_cup_and_handle, 
                              detect_ascending_triangle, detect_double_bottom,
                              detect_bull_flag, detect_bear_flag)
from qullamaggie_scanner import (detect_qullamaggie_breakout, 
                                  qullamaggie_pattern)
from momentum_burst_scanner import (detect_momentum_burst_1d,
                                     detect_momentum_burst_3d,
                                     detect_momentum_burst_5d,
                                     momentum_burst_patterns)
from supertrend_scanner import (detect_supertrend_bullish,
                                 detect_supertrend_fresh,
                                 detect_supertrend_recent,
                                 supertrend_patterns)
from explosive_volume_scanner import (detect_explosive_volume_3x,
                                       detect_explosive_volume_5x,
                                       detect_explosive_volume_10x,
                                       detect_volume_surge_with_price,
                                       explosive_volume_patterns)
from pattern_scoring import calculate_pattern_strength, get_signal_quality
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta, date

app = Flask(__name__)

# Database configuration - use environment variable for cloud deployment
DUCKDB_PATH = os.environ.get('DUCKDB_PATH', '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb')

# Set your Alpha Vantage API key here or use environment variable
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '75IGYUZ3C7AC2PBM')


def format_market_cap(market_cap):
    """Format market cap for display (e.g., 3.99T, 415.6B, 500.2M)."""
    if market_cap is None:
        return None
    
    try:
        # Convert to float if it's a string
        if isinstance(market_cap, str):
            market_cap = float(market_cap)
        
        if market_cap >= 1_000_000_000_000:
            return f"{market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"{market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"{market_cap / 1_000_000:.2f}M"
        else:
            return f"{market_cap:,.0f}"
    except:
        return None


def get_news_sentiment(symbol):
    """Get news sentiment for a symbol from Alpha Vantage."""
    try:
        import requests
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&limit=10'
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'feed' in data and len(data['feed']) > 0:
            # Calculate average sentiment from recent articles
            sentiment_scores = []
            sentiment_labels = []
            
            for article in data['feed'][:10]:  # Look at top 10 articles
                if 'ticker_sentiment' in article:
                    for ticker_sent in article['ticker_sentiment']:
                        if ticker_sent['ticker'] == symbol:
                            score = float(ticker_sent.get('ticker_sentiment_score', 0))
                            label = ticker_sent.get('ticker_sentiment_label', 'Neutral')
                            sentiment_scores.append(score)
                            sentiment_labels.append(label)
            
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                
                # Determine overall sentiment
                if avg_score >= 0.35:
                    overall = 'Bullish'
                elif avg_score >= 0.15:
                    overall = 'Somewhat-Bullish'
                elif avg_score <= -0.35:
                    overall = 'Bearish'
                elif avg_score <= -0.15:
                    overall = 'Somewhat-Bearish'
                else:
                    overall = 'Neutral'
                
                return {
                    'score': round(avg_score, 3),
                    'label': overall,
                    'article_count': len(sentiment_scores),
                    'total_articles': len(data['feed'])
                }
    except Exception as e:
        print(f"Error getting sentiment for {symbol}: {e}")
    
    return None


def get_earnings_date(symbol):
    """Get next earnings date for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            
            # Handle both single date and list of dates
            if not isinstance(earnings_dates, list):
                earnings_dates = [earnings_dates]
            
            # Get the first future date
            next_date = None
            for ed in earnings_dates:
                if ed:
                    next_date = ed
                    break
            
            if next_date:
                # Convert to date for comparison
                if hasattr(next_date, 'date'):
                    next_date = next_date.date()
                
                today = date.today()
                days_until = (next_date - today).days
                
                return {
                    'date': next_date.strftime('%Y-%m-%d'),
                    'days_until': days_until
                }
    except Exception as e:
        print(f"Error getting earnings for {symbol}: {e}")
    
    return None


@app.route('/snapshot')
def snapshot():
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    with open('datasets/symbols.csv') as f:
        for line in f:
            if "," not in line:
                continue
            symbol = line.split(",")[0]
            try:
                # Get daily data from Alpha Vantage (compact = last 100 days)
                # Use 'full' for complete history, 'compact' for recent data only
                data, meta_data = ts.get_daily(symbol=symbol, outputsize='compact')
                # Rename columns to match previous format (lowercase)
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                # Keep only last 252 trading days (~1 year)
                data = data.head(252)
                data.to_csv('datasets/daily/{}.csv'.format(symbol))
                print(f'Downloaded {symbol}')
            except Exception as e:
                print(f'Failed on {symbol}: {e}')

    return {
        "code": "success"
    }

@app.route('/')
def index():
    pattern = request.args.get('pattern', False)
    min_market_cap = request.args.get('min_market_cap', '')
    sector_filter = request.args.get('sector', '')
    min_strength = request.args.get('min_strength', '')
    stocks = {}

    # Connect to DuckDB and get list of symbols
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Build query with filters
    symbols_query = '''
        SELECT DISTINCT d.symbol, 
               COALESCE(f.company_name, d.symbol) as company,
               f.market_cap,
               f.sector
        FROM daily_cache d
        LEFT JOIN fundamental_cache f ON d.symbol = f.symbol
        WHERE 1=1
    '''
    
    params = []
    
    # Add market cap filter
    if min_market_cap:
        # Parse market cap values like "1B", "100M", "500M", "5B", "10B"
        cap_value = min_market_cap.upper()
        if 'B' in cap_value:
            min_cap = float(cap_value.replace('B', '')) * 1_000_000_000
        elif 'M' in cap_value:
            min_cap = float(cap_value.replace('M', '')) * 1_000_000
        else:
            min_cap = float(cap_value)
        
        symbols_query += ' AND f.market_cap IS NOT NULL'
        # Market cap in DuckDB is stored as string like "1.5B", "500M"
        
    # Add sector filter
    if sector_filter and sector_filter != 'All':
        symbols_query += ' AND f.sector = ?'
        params.append(sector_filter)
    
    symbols_query += ' ORDER BY d.symbol'
    
    symbol_rows = conn.execute(symbols_query, params).fetchall()
    
    # Filter by market cap if needed (since market_cap is stored as string)
    if min_market_cap:
        filtered_rows = []
        for row in symbol_rows:
            symbol, company, market_cap, sector = row
            if market_cap:
                try:
                    cap_str = market_cap.upper()
                    if 'T' in cap_str:
                        cap_num = float(cap_str.replace('T', '')) * 1_000_000_000_000
                    elif 'B' in cap_str:
                        cap_num = float(cap_str.replace('B', '')) * 1_000_000_000
                    elif 'M' in cap_str:
                        cap_num = float(cap_str.replace('M', '')) * 1_000_000
                    else:
                        cap_num = float(cap_str)
                    
                    if cap_num >= min_cap:
                        filtered_rows.append((symbol, company, market_cap, sector))
                except:
                    pass
        symbol_rows = [(s, c, mc, sec) for s, c, mc, sec in filtered_rows]
    else:
        symbol_rows = [(row[0], row[1], row[2], row[3]) for row in symbol_rows]
    
    for symbol, company, market_cap, sector in symbol_rows:
        stocks[symbol] = {
            'company': company, 
            'market_cap': format_market_cap(market_cap),
            'sector': sector
        }

    if pattern:
        # Check if it's a custom chart pattern
        custom_patterns = {
            'CUP_AND_HANDLE': detect_cup_and_handle,
            'ASCENDING_TRIANGLE': detect_ascending_triangle,
            'DOUBLE_BOTTOM': detect_double_bottom,
            'BULL_FLAG': detect_bull_flag,
            'BEAR_FLAG': detect_bear_flag,
            'QULLAMAGGIE_BREAKOUT': detect_qullamaggie_breakout,
            'MOMENTUM_BURST_1D': detect_momentum_burst_1d,
            'MOMENTUM_BURST_3D': detect_momentum_burst_3d,
            'MOMENTUM_BURST_5D': detect_momentum_burst_5d,
            'SUPERTREND_BULLISH': detect_supertrend_bullish,
            'SUPERTREND_FRESH': detect_supertrend_fresh,
            'SUPERTREND_RECENT': detect_supertrend_recent,
            'EXPLOSIVE_VOLUME_3X': detect_explosive_volume_3x,
            'EXPLOSIVE_VOLUME_5X': detect_explosive_volume_5x,
            'EXPLOSIVE_VOLUME_10X': detect_explosive_volume_10x,
            'VOLUME_SURGE_WITH_PRICE': detect_volume_surge_with_price
        }
        
        for symbol in list(stocks.keys()):
            try:
                # Get data from DuckDB (last 252 days)
                query = '''
                    SELECT date, open, high, low, close, volume,
                           rsi_14, sma_20, sma_50, sma_200, atr_14, rvol
                    FROM daily_cache
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 252
                '''
                df = conn.execute(query, [symbol]).df()
                
                if df.empty:
                    continue
                
                # Sort oldest to newest for pattern detection
                df = df.sort_values('date')
                
                # Skip stocks with outdated data (older than 30 days)
                latest_date = df['date'].iloc[-1]
                days_old = (pandas.Timestamp.now() - latest_date).days
                if days_old > 30:
                    continue
                
                # Rename columns to match TA-Lib expectations
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)

                result = None
                if pattern in custom_patterns:
                    # Custom chart pattern
                    result = custom_patterns[pattern](df)
                else:
                    # TA-Lib candlestick pattern
                    pattern_function = getattr(talib, pattern)
                    results = pattern_function(
                        df['Open'], df['High'], df['Low'], df['Close']
                    )
                    last = results.tail(1).values[0]

                    if last > 0:
                        result = 'bullish'
                    elif last < 0:
                        result = 'bearish'
                
                # Calculate pattern strength
                if result and result != 'None':
                    strength = calculate_pattern_strength(df, result, pattern)
                    quality = get_signal_quality(strength)
                    
                    # Calculate volume statistics
                    latest_volume = df['Volume'].iloc[-1]
                    avg_volume_20 = df['Volume'].tail(20).mean()
                    volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                    
                    # Apply minimum strength filter
                    min_strength_value = float(min_strength) if min_strength else 0
                    if strength >= min_strength_value:
                        stocks[symbol][pattern] = result
                        stocks[symbol][f'{pattern}_strength'] = strength
                        stocks[symbol][f'{pattern}_quality'] = quality
                        stocks[symbol][f'{pattern}_volume'] = int(latest_volume)
                        stocks[symbol][f'{pattern}_avg_volume'] = int(avg_volume_20)
                        stocks[symbol][f'{pattern}_volume_ratio'] = round(volume_ratio, 2)
                        
                        # Get earnings date
                        earnings_info = get_earnings_date(symbol)
                        if earnings_info:
                            stocks[symbol][f'{pattern}_earnings_date'] = earnings_info['date']
                            stocks[symbol][f'{pattern}_earnings_days'] = earnings_info['days_until']
                        else:
                            stocks[symbol][f'{pattern}_earnings_date'] = None
                            stocks[symbol][f'{pattern}_earnings_days'] = None
                        
                        # Get news sentiment
                        sentiment_info = get_news_sentiment(symbol)
                        if sentiment_info:
                            stocks[symbol][f'{pattern}_sentiment_score'] = sentiment_info['score']
                            stocks[symbol][f'{pattern}_sentiment_label'] = sentiment_info['label']
                            stocks[symbol][f'{pattern}_sentiment_articles'] = sentiment_info['article_count']
                        else:
                            stocks[symbol][f'{pattern}_sentiment_score'] = None
                            stocks[symbol][f'{pattern}_sentiment_label'] = None
                            stocks[symbol][f'{pattern}_sentiment_articles'] = None
                    else:
                        stocks[symbol][pattern] = None
                else:
                    stocks[symbol][pattern] = None
                    
            except Exception as e:
                print(f'failed on {symbol}: {e}')
    
    # Get available sectors for dropdown
    sectors_query = '''
        SELECT DISTINCT sector 
        FROM fundamental_cache 
        WHERE sector IS NOT NULL 
        ORDER BY sector
    '''
    available_sectors = [row[0] for row in conn.execute(sectors_query).fetchall()]
    
    conn.close()

    # Combine all patterns
    all_patterns = {
        **candlestick_patterns, 
        **custom_chart_patterns,
        **qullamaggie_pattern,
        **momentum_burst_patterns,
        **supertrend_patterns,
        **explosive_volume_patterns
    }
    
    return render_template(
        'index.html',
        candlestick_patterns=all_patterns,
        stocks=stocks,
        pattern=pattern,
        available_sectors=available_sectors,
        selected_sector=sector_filter,
        selected_market_cap=min_market_cap,
        selected_min_strength=min_strength
    )
