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

# Database configuration
# For local development, use MotherDuck to access production data
# For production (Render), use environment variable
motherduck_token = os.environ.get('motherduck_token') or os.environ.get('MOTHERDUCK_TOKEN', '')
if motherduck_token:
    DUCKDB_PATH = os.environ.get('DUCKDB_PATH', f'md:scanner_data?motherduck_token={motherduck_token}')
    print("INFO: Connecting to MotherDuck production database")
else:
    # Fallback to local DB if no MotherDuck token
    DUCKDB_PATH = os.environ.get('DUCKDB_PATH', '/Users/george/scannerPOC/breakoutScannersPOCs/scanner_data.duckdb')
    print("WARNING: No motherduck_token found, using local database (may not have scanner_results)")

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


@app.route('/scanner-docs')
def scanner_docs():
    """Display documentation landing page with all scanners."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get scanner info
    scanner_data = conn.execute("""
        SELECT scanner_name, COUNT(*) as count
        FROM scanner_data.scanner_results
        GROUP BY scanner_name
        ORDER BY scanner_name
    """).fetchall()
    
    conn.close()
    
    # Scanner descriptions
    scanner_descriptions = {
        'accumulation_distribution': 'Detects institutional smart money buying patterns using volume indicators',
        'breakout': 'Identifies stocks breaking out above key resistance levels',
        'bull_flag': 'Finds bullish continuation patterns with consolidation after uptrend',
        'momentum_burst': 'Spots explosive momentum moves with high volume',
        'tight_consolidation': 'Detects tight consolidation patterns before potential breakouts'
    }
    
    scanners = []
    for name, count in scanner_data:
        scanners.append({
            'name': name,
            'display_name': name.replace('_', ' ').title(),
            'short_desc': scanner_descriptions.get(name, 'Technical pattern scanner'),
            'count': count
        })
    
    return render_template('scanner_docs.html', scanners=scanners)


@app.route('/scanner-docs/<scanner_name>')
def scanner_detail(scanner_name):
    """Display detailed documentation for a specific scanner."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get scanner stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(signal_strength) as avg_strength,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM scanner_data.scanner_results
        WHERE scanner_name = ?
    """, [scanner_name]).fetchone()
    
    conn.close()
    
    scanner_info = {
        'name': scanner_name,
        'display_name': scanner_name.replace('_', ' ').title(),
        'total_setups': stats[0] if stats else 0,
        'avg_strength': f"{stats[1]:.1f}" if stats and stats[1] else "N/A",
        'unique_symbols': stats[2] if stats else 0
    }
    
    # Load scanner-specific content
    content = get_scanner_documentation(scanner_name)
    
    return render_template('scanner_detail.html', scanner_info=scanner_info, content=content)


def get_scanner_documentation(scanner_name):
    """Return HTML documentation for specific scanner."""
    
    docs = {
        'accumulation_distribution': '''
<h2>What It Does</h2>
<p>The Accumulation/Distribution scanner detects institutional smart money buying patterns using multiple volume-based indicators.</p>

<div class="indicator-box">
    <h3>Core Indicators:</h3>
    <ul>
        <li><strong>A/D Line</strong> - Tracks money flow (buying vs selling pressure)</li>
        <li><strong>OBV (On-Balance Volume)</strong> - Volume-weighted price momentum</li>
        <li><strong>CMF (Chaikin Money Flow)</strong> - 20-period money flow oscillator</li>
        <li><strong>Volume Profile</strong> - Up-day vs down-day volume ratios</li>
        <li><strong>Divergence Detection</strong> - Price/indicator mismatches (bullish signals)</li>
    </ul>
</div>

<h2>What "Accumulation" Means</h2>
<ul>
    <li>Smart money (institutions, hedge funds) quietly buying shares</li>
    <li>Price might be flat/consolidating, but volume shows hidden buying</li>
    <li>Typically happens before major breakouts</li>
    <li><strong>Example:</strong> TER at $83.08 - institutions accumulated there, now at $187 (+125%)</li>
</ul>

<h2>Why Multiple Setups Are Found</h2>

<h3>1. Quality Threshold (70/100)</h3>
<p>Stocks meeting a 70+ quality score are included. Most cluster around:</p>
<ul>
    <li>73 score: ~90 stocks (most common)</li>
    <li>78 score: ~48 stocks</li>
    <li>80 score: ~45 stocks</li>
    <li>100 score: ~6 stocks (perfect signals)</li>
</ul>

<h3>2. Volume Filter (Testing Mode)</h3>
<div class="alert">
    <strong>Current:</strong> Accepts stocks with $5M+ daily volume<br>
    <strong>Production:</strong> Would require $50M+ (10x stricter → ~90% fewer results)
</div>

<h3>3. Sector Filtering Not Active</h3>
<p>Currently all sectors included. When enabled, would filter:</p>
<ul>
    <li><strong>Avoid:</strong> UTILITIES (8.9% success), REAL ESTATE (10.7%)</li>
    <li><strong>Prefer:</strong> TECHNOLOGY (30.6%), ENERGY (25.8%)</li>
</ul>

<h2>Quality Score Breakdown</h2>
<table>
    <thead>
        <tr>
            <th>Quality</th>
            <th>Typical %</th>
            <th>Interpretation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>100</td>
            <td>~2%</td>
            <td>Perfect - All indicators aligned</td>
        </tr>
        <tr>
            <td>93-98</td>
            <td>~8%</td>
            <td>Excellent - Strong accumulation</td>
        </tr>
        <tr>
            <td>85-90</td>
            <td>~8%</td>
            <td>Very Good - Clear buying</td>
        </tr>
        <tr>
            <td>80-83</td>
            <td>~18%</td>
            <td>Good - Solid setup</td>
        </tr>
        <tr>
            <td>73-78</td>
            <td>~47%</td>
            <td>Fair - Marginal quality</td>
        </tr>
        <tr>
            <td>70-72</td>
            <td>~9%</td>
            <td>Minimum - Barely qualifies</td>
        </tr>
    </tbody>
</table>

<div class="alert">
    <strong>Key Issue:</strong> About 47% of results are "Fair" quality (73-78 range) - these are borderline setups. Consider raising minimum threshold to 80+ for higher quality signals.
</div>
''',
        'breakout': '''
<h2>What It Does</h2>
<p>The Breakout scanner identifies stocks breaking above key resistance levels with strong volume confirmation.</p>

<div class="indicator-box">
    <h3>Detection Criteria:</h3>
    <ul>
        <li><strong>Resistance Break</strong> - Price closes above 52-week high or major pivot</li>
        <li><strong>Volume Surge</strong> - 2x+ average volume on breakout day</li>
        <li><strong>Momentum</strong> - Strong upward price action</li>
        <li><strong>Consolidation</strong> - Prior base building phase detected</li>
    </ul>
</div>

<h2>Why Breakouts Matter</h2>
<ul>
    <li>New all-time highs have no overhead resistance</li>
    <li>Attracts momentum traders and institutions</li>
    <li>Often leads to extended moves (continuation pattern)</li>
    <li>High volume confirms genuine interest vs false breakout</li>
</ul>

<h2>Quality Factors</h2>
<h3>Strong Breakouts (80-100)</h3>
<ul>
    <li>Volume 3x+ average</li>
    <li>Clean technical setup</li>
    <li>Multiple timeframe confirmation</li>
    <li>Tight consolidation before break</li>
</ul>

<h3>Moderate Breakouts (60-79)</h3>
<ul>
    <li>Volume 1.5-3x average</li>
    <li>Some resistance overhead</li>
    <li>Wider consolidation pattern</li>
</ul>

<div class="success">
    <strong>Best Practice:</strong> Wait for 2-3 day confirmation above breakout level before entering. False breakouts often fail within 48 hours.
</div>
''',
        'bull_flag': '''
<h2>What It Does</h2>
<p>The Bull Flag scanner finds bullish continuation patterns - a strong uptrend followed by a tight consolidation "flag" that typically leads to another leg higher.</p>

<div class="indicator-box">
    <h3>Pattern Components:</h3>
    <ul>
        <li><strong>Flagpole</strong> - Sharp upward move (20%+ gain in 1-3 weeks)</li>
        <li><strong>Flag</strong> - Tight consolidation (5-10% pullback, 1-3 weeks)</li>
        <li><strong>Volume</strong> - Heavy on flagpole, light during flag</li>
        <li><strong>Breakout</strong> - Move above flag high with volume surge</li>
    </ul>
</div>

<h2>Why Bull Flags Work</h2>
<ul>
    <li>Profit-taking creates healthy consolidation</li>
    <li>Strong hands accumulate during pullback</li>
    <li>Short sellers trapped when breakout resumes</li>
    <li>Pattern measured move: flagpole height added to breakout</li>
</ul>

<h2>Quality Assessment</h2>

<h3>High Quality Flags (85-100)</h3>
<ul>
    <li>Steep flagpole (30%+ in <2 weeks)</li>
    <li>Tight flag (3-5% range)</li>
    <li>Volume contracts 50%+ during flag</li>
    <li>Clean chart with no overhead resistance</li>
</ul>

<h3>Standard Flags (70-84)</h3>
<ul>
    <li>Moderate flagpole (15-30%)</li>
    <li>Wider flag (5-10% range)</li>
    <li>Some overhead resistance levels</li>
</ul>

<div class="alert">
    <strong>Risk Factor:</strong> Bull flags that take >4 weeks to form lose their potency. Best setups resolve within 2-3 weeks.
</div>

<h2>Entry Strategies</h2>
<ol>
    <li><strong>Aggressive:</strong> Buy near flag low with tight stop</li>
    <li><strong>Conservative:</strong> Wait for breakout above flag high</li>
    <li><strong>Confirmation:</strong> Enter on first pullback after breakout</li>
</ol>
''',
        'momentum_burst': '''
<h2>What It Does</h2>
<p>The Momentum Burst scanner detects explosive price moves with massive volume - the kind of "rocket ship" moves that can deliver 20-50% gains in days.</p>

<div class="indicator-box">
    <h3>Detection Triggers:</h3>
    <ul>
        <li><strong>Price Surge</strong> - 5%+ move in single day</li>
        <li><strong>Volume Explosion</strong> - 5x+ average volume</li>
        <li><strong>Momentum Shift</strong> - RSI break above 70</li>
        <li><strong>Buying Pressure</strong> - Strong closing range (top 25% of day)</li>
    </ul>
</div>

<h2>What Causes Momentum Bursts</h2>
<ul>
    <li><strong>Earnings Surprises</strong> - Blowout results trigger buying frenzy</li>
    <li><strong>News Catalysts</strong> - FDA approval, major contract wins, analyst upgrades</li>
    <li><strong>Short Squeeze</strong> - Heavily shorted stock forces shorts to cover</li>
    <li><strong>Breakout Momentum</strong> - Technical breakout attracts algorithm buying</li>
</ul>

<h2>Timeframe Variants</h2>

<h3>1-Day Burst (Most Common)</h3>
<ul>
    <li>Single day explosive move</li>
    <li>Often news-driven</li>
    <li>Higher risk of immediate reversal</li>
    <li><strong>Strategy:</strong> Quick scalp or wait for pullback</li>
</ul>

<h3>3-Day Burst (Stronger)</h3>
<ul>
    <li>Sustained momentum over 3 days</li>
    <li>More reliable follow-through</li>
    <li><strong>Strategy:</strong> Swing trade for continuation</li>
</ul>

<h3>5-Day Burst (Strongest)</h3>
<ul>
    <li>Week-long momentum move</li>
    <li>Usually major fundamental change</li>
    <li>Best for position trades</li>
</ul>

<h2>Quality Score Interpretation</h2>
<table>
    <thead>
        <tr>
            <th>Strength</th>
            <th>Volume Multiple</th>
            <th>Risk Level</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>90-100</td>
            <td>10x+ volume</td>
            <td>Extreme - Take profits quickly</td>
        </tr>
        <tr>
            <td>80-89</td>
            <td>7-10x volume</td>
            <td>High - Tight stops required</td>
        </tr>
        <tr>
            <td>70-79</td>
            <td>5-7x volume</td>
            <td>Moderate - Normal position sizing</td>
        </tr>
    </tbody>
</table>

<div class="alert">
    <strong>Warning:</strong> Momentum bursts are the most dangerous signals to trade. 60-70% fade within 3-5 days. Never chase at end of day - wait for pullback or consolidation.
</div>

<h2>Best Practices</h2>
<ol>
    <li><strong>Let It Breathe:</strong> Wait 1-2 days after initial burst</li>
    <li><strong>Find Support:</strong> Enter on pullback to VWAP or breakout level</li>
    <li><strong>Use Tight Stops:</strong> 5-7% max loss on these setups</li>
    <li><strong>Take Profits:</strong> Scale out at 10%, 20%, 30% gains</li>
</ol>
''',
        'tight_consolidation': '''
<h2>What It Does</h2>
<p>The Tight Consolidation scanner finds stocks coiling in extremely narrow price ranges - the "calm before the storm" pattern that often precedes explosive breakouts.</p>

<div class="indicator-box">
    <h3>Pattern Requirements:</h3>
    <ul>
        <li><strong>Narrow Range</strong> - Daily ranges <3% for 5+ days</li>
        <li><strong>Declining Volume</strong> - Volume drying up (shows no sellers)</li>
        <li><strong>Near Highs</strong> - Consolidating within 5% of 52-week high</li>
        <li><strong>Clean Base</strong> - No major overhead resistance</li>
    </ul>
</div>

<h2>Why Tight Consolidation Works</h2>
<ul>
    <li><strong>Spring Loading:</strong> Energy builds like compressed spring</li>
    <li><strong>No Sellers:</strong> Low volume proves supply exhausted</li>
    <li><strong>Resolution Required:</strong> Narrow ranges must eventually resolve (usually upward near highs)</li>
    <li><strong>Institutional Setup:</strong> Big money often accumulates during these periods</li>
</ul>

<h2>The Volatility Contraction Pattern</h2>
<p>This is based on Mark Minervini's "VCP" methodology:</p>
<ol>
    <li><strong>Phase 1:</strong> Initial consolidation (wider)</li>
    <li><strong>Phase 2:</strong> Tighter consolidation (narrower)</li>
    <li><strong>Phase 3:</strong> Very tight (explosive breakout imminent)</li>
</ol>

<h2>Quality Metrics</h2>

<h3>Perfect Setup (90-100)</h3>
<ul>
    <li>10+ days of <2% daily ranges</li>
    <li>Volume down 60%+ from average</li>
    <li>Within 2% of all-time high</li>
    <li>Strong sector/market backdrop</li>
</ul>

<h3>Good Setup (75-89)</h3>
<ul>
    <li>7-9 days of <3% ranges</li>
    <li>Volume down 40-60%</li>
    <li>Within 5% of 52-week high</li>
</ul>

<h3>Marginal Setup (60-74)</h3>
<ul>
    <li>5-6 days of consolidation</li>
    <li>Some overhead resistance visible</li>
    <li>Moderate volume contraction</li>
</ul>

<div class="success">
    <strong>Advantage:</strong> These are the SAFEST high-probability setups. Low-risk entry at consolidation low with stop just below. If it breaks out, often runs 20-50%+.
</div>

<h2>Entry Strategies</h2>

<h3>Strategy 1: Anticipation (Aggressive)</h3>
<ul>
    <li>Buy within the consolidation zone</li>
    <li>Stop below consolidation low</li>
    <li>Risk: 3-5%</li>
    <li>Reward: Can catch entire breakout move</li>
</ul>

<h3>Strategy 2: Breakout (Conservative)</h3>
<ul>
    <li>Wait for break above consolidation high</li>
    <li>Enter on volume surge</li>
    <li>Stop at consolidation low</li>
    <li>Risk: 5-7%</li>
</ul>

<h3>Strategy 3: Pullback (Best Risk/Reward)</h3>
<ul>
    <li>Let it break out first</li>
    <li>Wait for 2-3 day pullback</li>
    <li>Enter when it finds support</li>
    <li>Stop below pullback low</li>
    <li>Risk: 3-4%</li>
</ul>

<div class="alert">
    <strong>Key Insight:</strong> The tighter and longer the consolidation (up to ~3 weeks), the more explosive the eventual move. Consolidations >4 weeks start to lose their spring.
</div>

<h2>Why So Few Setups?</h2>
<p>Tight Consolidation is one of the rarest and highest-quality patterns:</p>
<ul>
    <li>Most stocks are either trending or in wide ranges</li>
    <li>True "tight" consolidation requires specific market conditions</li>
    <li>Pattern only forms on strongest stocks (weak stocks break down)</li>
    <li>Few stocks meet the strict >5 day, <3% range criteria</li>
</ul>

<p><strong>Historical Success Rate:</strong> When this pattern appears near all-time highs with proper volume characteristics, it has a ~65-70% success rate for producing 20%+ moves within 4-8 weeks.</p>
'''
    }
    
    return docs.get(scanner_name, '<p>Documentation coming soon...</p>')


@app.route('/')
def index():
    min_market_cap = request.args.get('min_market_cap', '')
    sector_filter = request.args.get('sector', '')
    min_strength = request.args.get('min_strength', '')
    selected_scan_date = request.args.get('scan_date', '')
    confirmed_only = request.args.get('confirmed_only', '')
    stocks = {}

    # Connect to DuckDB and get list of symbols
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    
    # Get default pattern (scanner with lowest count) if none selected
    pattern = request.args.get('pattern', None)
    if not pattern:
        try:
            default_scanner = conn.execute("""
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                GROUP BY scanner_name
                ORDER BY count ASC, scanner_name
                LIMIT 1
            """).fetchone()
            pattern = default_scanner[0] if default_scanner else False
            print(f"INFO: Set default scanner to: {pattern} ({default_scanner[1] if default_scanner else 0} setups)")
        except Exception as e:
            print(f"ERROR: Could not get default scanner: {e}")
            pattern = False
    else:
        pattern = pattern if pattern != '' else False
    
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
            # Use date range instead of DATE() function to allow index usage
            date_obj = datetime.strptime(selected_scan_date, '%Y-%m-%d')
            next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            
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
                AND scan_date >= ? 
                AND scan_date < ?
            '''
            query_params = [pattern, selected_scan_date, next_day]
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
        
        # Optimization: Fetch ALL volume data in one query instead of N queries
        symbols_list = list(stocks.keys())
        volume_data_dict = {}
        
        if symbols_list:
            try:
                # Build a query with IN clause to get all volumes at once
                placeholders = ','.join(['?' for _ in symbols_list])
                bulk_vol_query = f'''
                    SELECT dc.symbol, dc.volume, dc.avg_volume_20
                    FROM scanner_data.daily_cache dc
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM scanner_data.daily_cache
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    ) latest ON dc.symbol = latest.symbol AND dc.date = latest.max_date
                '''
                vol_results = conn.execute(bulk_vol_query, symbols_list).fetchall()
                volume_data_dict = {
                    row[0]: {
                        'volume': int(row[1]),
                        'avg_volume_20': int(row[2]) if row[2] else int(row[1])
                    } for row in vol_results
                }
                print(f'Loaded volume data for {len(volume_data_dict)} symbols in single query')
            except Exception as e:
                print(f'Bulk volume query failed: {e}')
                volume_data_dict = {}
        
        # Fetch all scanner confirmations for each symbol
        confirmations_dict = {}
        if symbols_list:
            try:
                placeholders = ','.join(['?' for _ in symbols_list])
                confirmations_query = f'''
                    SELECT symbol, scanner_name, scan_date, signal_strength
                    FROM scanner_data.scanner_results
                    WHERE symbol IN ({placeholders})
                    AND scanner_name != ?
                    ORDER BY symbol, scan_date DESC, scanner_name
                '''
                params = symbols_list + [pattern]
                conf_results = conn.execute(confirmations_query, params).fetchall()
                
                for row in conf_results:
                    sym = row[0]
                    scanner = row[1]
                    scan_dt = str(row[2])[:10] if row[2] else ''
                    strength = row[3]
                    
                    if sym not in confirmations_dict:
                        confirmations_dict[sym] = []
                    confirmations_dict[sym].append({
                        'scanner': scanner,
                        'date': scan_dt,
                        'strength': strength
                    })
                
                print(f'Loaded scanner confirmations for {len(confirmations_dict)} symbols')
            except Exception as e:
                print(f'Confirmations query failed: {e}')
                confirmations_dict = {}
        
        for symbol in symbols_list:
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
                        # Get pre-fetched volume data
                        if symbol in volume_data_dict:
                            vol_data = volume_data_dict[symbol]
                            latest_volume = vol_data['volume']
                            avg_volume_20 = vol_data['avg_volume_20']
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
                            
                            # Add scanner confirmations
                            if symbol in confirmations_dict:
                                stocks[symbol][f'{pattern}_confirmations'] = confirmations_dict[symbol]
                            else:
                                stocks[symbol][f'{pattern}_confirmations'] = []
                            
                            # Skip external API calls - too slow for Render
                            stocks[symbol][f'{pattern}_earnings_date'] = None
                            stocks[symbol][f'{pattern}_earnings_days'] = None
                            
                            # Skip sentiment API calls - too slow
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
    
    # Apply confirmed_only filter
    if confirmed_only == 'yes' and pattern:
        filtered_stocks = {}
        for symbol, data in stocks.items():
            if data.get(f'{pattern}_confirmations') and len(data[f'{pattern}_confirmations']) > 0:
                filtered_stocks[symbol] = data
        stocks = filtered_stocks
        print(f'Filtered to {len(stocks)} stocks confirmed by other scanners')
    
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
    
    # Get scanner names from database for the dropdown with counts
    try:
        # Get scanner counts based on selected date
        if selected_scan_date:
            # Use the selected date
            date_to_use = selected_scan_date
            print(f"INFO: Using selected date: {date_to_use}")
        else:
            # Get the latest scan date
            latest_date_result = conn.execute("""
                SELECT MAX(DATE(scan_date)) 
                FROM scanner_data.scanner_results
            """).fetchone()
            date_to_use = str(latest_date_result[0]) if latest_date_result and latest_date_result[0] else None
            print(f"INFO: Using latest scan date: {date_to_use}")
        
        if date_to_use:
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                WHERE DATE(scan_date) = ?
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = conn.execute(scanner_counts_query, [date_to_use]).fetchall()
            print(f"INFO: Found {len(scanner_counts)} scanners for date {date_to_use}")
        else:
            # Fallback if no date available
            print("WARNING: No date available, getting all scanner counts")
            scanner_counts_query = """
                SELECT scanner_name, COUNT(*) as count
                FROM scanner_data.scanner_results
                GROUP BY scanner_name
                ORDER BY scanner_name
            """
            scanner_counts = conn.execute(scanner_counts_query).fetchall()
        
        # Create patterns dict with scanner_name and count
        all_patterns = {}
        for row in scanner_counts:
            scanner_name = row[0]
            count = row[1]
            display_name = f"{scanner_name.replace('_', ' ').title()} ({count})"
            all_patterns[scanner_name] = display_name
        
        available_scanners = list(all_patterns.keys())
        print(f"INFO: Loaded {len(all_patterns)} scanner patterns")
    except Exception as e:
        print(f"ERROR: Could not load scanners from DB: {e}")
        import traceback
        traceback.print_exc()
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
        selected_min_strength=min_strength,
        confirmed_only=confirmed_only
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    app.run(debug=debug, host='0.0.0.0', port=port)
