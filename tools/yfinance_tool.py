import yfinance as yf
import pandas as pd

def get_ticker_info(ticker: str) -> dict:
    """Gets fundamental stats from yfinance info dict."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        return {
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'eps': info.get('trailingEps'),
            'revenueGrowth': info.get('revenueGrowth'),
            'debtToEquity': info.get('debtToEquity'),
            'profitMargins': info.get('profitMargins'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'longName': info.get('longName'),
            'marketCap': info.get('marketCap')
        }
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return {}

def get_historical_data(ticker: str, period="1y") -> pd.DataFrame:
    """Gets OHLCV data."""
    try:
        t = yf.Ticker(ticker.upper())
        df = t.history(period=period)
        return df
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

def get_earnings_history(ticker: str) -> list:
    """Gets earnings surprise history."""
    try:
        t = yf.Ticker(ticker.upper())
        # yfinance has t.earnings_dates or history
        df = t.earnings_dates
        if df is not None and not df.empty:
            # Format nicely
            df = df.reset_index()
            # Select relevant cols
            cols = ['Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise %']
            df.columns = [c if c in df.columns else c for c in df.columns] # Handle renamed or missing
            # Standardize for prompt
            res = df.to_dict('records')[:5] # Top 5 recent
            return res
        return []
    except Exception as e:
        print(f"Error fetching earnings history for {ticker}: {e}")
        return []

if __name__ == "__main__":
    print(get_ticker_info("NVDA"))
    print(get_historical_data("NVDA")[-3:])
