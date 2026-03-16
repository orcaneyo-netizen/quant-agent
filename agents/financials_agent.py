import sys
import os

# Add parent dir to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.yfinance_tool import get_ticker_info, get_earnings_history

def analyze_fundamentals(ticker: str) -> dict:
    """
    Fetches fundamental stats and returns structured output.
    """
    print(f"[{ticker}] Running FinancialsAgent...")
    info = get_ticker_info(ticker)
    earn_hist = get_earnings_history(ticker)
    
    # Structure fundamentals for the agent node
    # Book to Market: 1 / priceToBook
    price_to_book = info.get('priceToBook')
    book_to_market = 1.0 / price_to_book if price_to_book and price_to_book > 0 else 0.0

    fundamentals = {
        'trailing_pe': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'eps': info.get('eps'),
        'revenue_growth': info.get('revenueGrowth'),
        'debt_to_equity': info.get('debtToEquity'),
        'profit_margins': info.get('profitMargins'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'book_to_market_ratio': round(book_to_market, 4),
        'market_cap': info.get('marketCap')
    }
    
    # We don't have immediate access to industry/sector averages unless 
    # we aggregate all tickers. Let's add a note or placeholder for sector_comparison.
    # In orchestrator, we could compute the basket average if multiple tickers are inside the same sector.
    earnings_surprise = []
    if earn_hist:
        try:
            # Format nicely for reporting
            for h in earn_hist:
                date_str = str(h.get('Earnings Date', 'NA'))
                estimate = h.get('EPS Estimate', 'NA')
                reported = h.get('Reported EPS', 'NA')
                surprise = h.get('Surprise %', 'NA')
                earnings_surprise.append({
                    'date': date_str,
                    'estimate': estimate,
                    'reported': reported,
                    'surprise': surprise
                })
        except Exception as e:
            print(f"[{ticker}] Error parsing earnings history: {e}")
            
    print(f"[{ticker}] Financials obtained: PE={fundamentals.get('trailing_pe')}")
    
    return {
        'fundamentals': fundamentals,
        'earnings_surprise_history': earnings_surprise[:4] # Top 4
    }

if __name__ == "__main__":
    import json
    res = analyze_fundamentals("NVDA")
    print(json.dumps(res, indent=2))
