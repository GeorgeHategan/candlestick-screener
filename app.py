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

# Skip slow database initialization on startup - connect lazily when needed
print(f"INFO: Database path configured: {DUCKDB_PATH}")


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

@app.route('/stats')
def stats():
    """Display database statistics landing page."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    stats_data = {}
    
    try:
        # Total number of scanner results
        total_results = conn.execute("""
            SELECT COUNT(*) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['total_results'] = total_results
        
        # Number of unique assets scanned
        unique_assets = conn.execute("""
            SELECT COUNT(DISTINCT symbol) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['unique_assets'] = unique_assets
        
        # Number of scanners
        num_scanners = conn.execute("""
            SELECT COUNT(DISTINCT scanner_name) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['num_scanners'] = num_scanners
        
        # Last updated date
        last_updated = conn.execute("""
            SELECT MAX(scan_date) FROM scanner_data.scanner_results
        """).fetchone()[0]
        stats_data['last_updated'] = str(last_updated)[:10] if last_updated else 'N/A'
        
        # Results per scanner
        scanner_breakdown = conn.execute("""
            SELECT scanner_name, COUNT(*) as count
            FROM scanner_data.scanner_results
            GROUP BY scanner_name
            ORDER BY count DESC
        """).fetchall()
        stats_data['scanner_breakdown'] = [(row[0], row[1]) for row in scanner_breakdown]
        
        # Results per date
        date_breakdown = conn.execute("""
            SELECT DATE(scan_date) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY DATE(scan_date)
            ORDER BY date DESC
            LIMIT 10
        """).fetchall()
        stats_data['date_breakdown'] = [(str(row[0]), row[1]) for row in date_breakdown]
        
        # Top picked assets (by multiple scanners)
        top_picks = conn.execute("""
            SELECT symbol, COUNT(DISTINCT scanner_name) as scanner_count
            FROM scanner_data.scanner_results
            GROUP BY symbol
            HAVING COUNT(DISTINCT scanner_name) > 1
            ORDER BY scanner_count DESC
            LIMIT 20
        """).fetchall()
        stats_data['top_picks'] = [(row[0], row[1]) for row in top_picks]
        
        # Signal strength distribution
        strength_dist = conn.execute("""
            SELECT 
                CASE 
                    WHEN signal_strength >= 90 THEN '90-100'
                    WHEN signal_strength >= 80 THEN '80-89'
                    WHEN signal_strength >= 70 THEN '70-79'
                    WHEN signal_strength >= 60 THEN '60-69'
                    ELSE '<60'
                END as strength_range,
                COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE signal_strength IS NOT NULL
            GROUP BY strength_range
            ORDER BY strength_range DESC
        """).fetchall()
        stats_data['strength_distribution'] = [(row[0], row[1]) for row in strength_dist]
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        stats_data['error'] = str(e)
    
    conn.close()
    
    return render_template('stats.html', stats=stats_data)


@app.route('/')
def index():
    pattern = request.args.get('pattern', False)
    min_market_cap = request.args.get('min_market_cap', '')
    sector_filter = request.args.get('sector', '')
    min_strength = request.args.get('min_strength', '')
    selected_scan_date = request.args.get('scan_date', '')
    stocks = {}

    # Connect to DuckDB and get list of symbols
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Build query with filters
    symbols_query = '''
        SELECT DISTINCT d.symbol, 
               COALESCE(f.company_name, d.symbol) as company,
               f.market_cap,
               f.sector
        FROM scanner_data.daily_cache d
        LEFT JOIN scanner_data.fundamental_cache f ON d.symbol = f.symbol
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
        # Use pattern name directly as scanner name
        print(f"Loading scanner results for: {pattern}")
        
        # Read pre-calculated scanner results from database
        # Build query with optional date filter
        if selected_scan_date:
            scanner_query = '''
                SELECT symbol,
                       signal_type,
                       COALESCE(signal_strength, 75) as signal_strength,
                       COALESCE(setup_stage, 'N/A') as quality_placeholder,
                       entry_price,
                       picked_by_scanners,
                       setup_stage,
                       scan_date
                FROM scanner_data.scanner_results
                WHERE scanner_name = ? AND DATE(scan_date) = ?
            '''
            query_params = [pattern, selected_scan_date]
        else:
            scanner_query = '''
                SELECT symbol,
                       signal_type,
                       COALESCE(signal_strength, 75) as signal_strength,
                       COALESCE(setup_stage, 'N/A') as quality_placeholder,
                       entry_price,
                       picked_by_scanners,
                       setup_stage,
                       scan_date
                FROM scanner_data.scanner_results
                WHERE scanner_name = ?
            '''
            query_params = [pattern]

        scanner_dict = {}
        # Simple query - just get all results
        try:
            scanner_results = conn.execute(scanner_query, query_params).fetchall()
            scanner_dict = {
                row[0]: {
                    'signal': row[1],
                    'strength': row[2],
                    'quality': row[3],
                    'entry_price': row[4],
                    'picked_by_scanners': row[5],
                    'setup_stage': row[6],
                    'scan_date': str(row[7])[:10] if row[7] else ''
                } for row in scanner_results
            }
            print(f'Found {len(scanner_dict)} results for {pattern}')
        except Exception as e:
            print(f'Scanner query failed: {e}')
            scanner_dict = {}
        
        for symbol in list(stocks.keys()):
            # Check if symbol has scanner results
            if symbol in scanner_dict:
                scanner_result = scanner_dict[symbol]
                result = scanner_result['signal']
                strength = scanner_result['strength']
                quality = scanner_result['quality']
                entry_price = scanner_result.get('entry_price')
                picked_by_scanners = scanner_result.get('picked_by_scanners')
                setup_stage = scanner_result.get('setup_stage')
                scan_date = scanner_result.get('scan_date', '')
                
                # Apply minimum strength filter
                min_strength_value = float(min_strength) if min_strength else 0
                if strength >= min_strength_value:
                    try:
                        # Get volume data for display
                        vol_query = '''
                            SELECT volume, avg_volume_20
                            FROM scanner_data.daily_cache
                            WHERE symbol = ?
                            ORDER BY date DESC
                            LIMIT 1
                        '''
                        vol_data = conn.execute(vol_query, [symbol]).fetchone()
                        
                        if vol_data:
                            latest_volume = int(vol_data[0])
                            avg_volume_20 = int(vol_data[1]) if vol_data[1] else latest_volume
                            volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else 0
                            
                            stocks[symbol][pattern] = result
                            stocks[symbol][f'{pattern}_strength'] = strength
                            stocks[symbol][f'{pattern}_quality'] = quality
                            stocks[symbol][f'{pattern}_scan_date'] = scan_date
                            stocks[symbol][f'{pattern}_volume'] = latest_volume
                            stocks[symbol][f'{pattern}_avg_volume'] = avg_volume_20
                            stocks[symbol][f'{pattern}_volume_ratio'] = round(volume_ratio, 2)
                            if entry_price is not None:
                                stocks[symbol][f'{pattern}_entry_price'] = entry_price
                            if picked_by_scanners is not None:
                                stocks[symbol][f'{pattern}_picked_count'] = picked_by_scanners
                            if setup_stage:
                                stocks[symbol][f'{pattern}_setup_stage'] = setup_stage
                            
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
                    except Exception as e:
                        print(f'failed on {symbol}: {e}')
                        stocks[symbol][pattern] = None
                else:
                    stocks[symbol][pattern] = None
            else:
                stocks[symbol][pattern] = None
    
    # Get available sectors for dropdown
    sectors_query = '''
        SELECT DISTINCT sector 
        FROM scanner_data.fundamental_cache 
        WHERE sector IS NOT NULL 
        ORDER BY sector
    '''
    available_sectors = [row[0] for row in conn.execute(sectors_query).fetchall()]
    
    # Get available pre-calculated scanners from database
    available_scanners = []
    try:
        scanners_query = '''
            SELECT DISTINCT scanner_name 
            FROM scanner_data.scanner_results 
            ORDER BY scanner_name
        '''
        available_scanners = [row[0] for row in conn.execute(scanners_query).fetchall()]
    except Exception as e:
        print(f'Could not get scanner list: {e}')
        # Fallback: empty list
        available_scanners = []
    
    # Get scanner names from database for the dropdown
    try:
        scanner_names = conn.execute("""
            SELECT DISTINCT scanner_name
            FROM scanner_data.scanner_results
            ORDER BY scanner_name
        """).fetchall()
        # Create patterns dict with scanner_name as both key and display value
        all_patterns = {row[0]: row[0].replace('_', ' ').title() for row in scanner_names}
        available_scanners = list(all_patterns.keys())
    except Exception as e:
        print(f"Could not load scanners from DB: {e}")
        all_patterns = {}
        available_scanners = []
    
    # Get available scan dates with setup counts
    available_scan_dates = []
    try:
        dates = conn.execute("""
            SELECT DATE(scan_date) as date, COUNT(*) as count
            FROM scanner_data.scanner_results
            WHERE scan_date IS NOT NULL
            GROUP BY DATE(scan_date)
            ORDER BY date DESC
        """).fetchall()
        available_scan_dates = [(str(row[0]), row[1]) for row in dates]
    except Exception as e:
        print(f"Could not load scan dates: {e}")
    
    conn.close()
    
    return render_template(
        'index.html',
        candlestick_patterns=all_patterns,
        stocks=stocks,
        pattern=pattern,
        available_sectors=available_sectors,
        available_scanners=available_scanners,
        available_scan_dates=available_scan_dates,
        selected_scan_date=selected_scan_date,
        selected_sector=sector_filter,
        selected_market_cap=min_market_cap,
        selected_min_strength=min_strength
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)
