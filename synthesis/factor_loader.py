import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_fama_french_factors(tickers: list, momentum_dict: dict = None) -> dict:
    """
    Fetches Fama-French 3 Factors and calculates beta, smb, hml loadings
    for each ticker using daily returns via OLS regression back 12 months.
    """
    print(f"Loading Fama-French factors for {tickers}...")
    factor_results = {}
    
    # 1. Fetch FF factors daily
    try:
        ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='2024-01-01')
        factors = ff_data[0] # Daily factors DataFrame
        # Standardize dates for join
        # FF dates are string indices like '20240102'. We convert to datetime index.
        factors.index = pd.to_datetime(factors.index.astype(str))
        
        # FF factors are in percent e.g. 1.2 (= 1.2%). Convert to decimal
        factors = factors / 100.0
        
    except Exception as e:
        print(f"Error fetching Fama-French factors: {e}")
        # dummy fallback factors
        return {t: {'market_beta': 1.0, 'smb_loading': 0.0, 'hml_loading': 0.0, 'momentum': 0.0, 'alpha': 0.0} for t in tickers}

    for ticker in tickers:
        try:
            # 2. Fetch daily prices for 12 months
            t = yf.Ticker(ticker)
            df = t.history(period="1y")
            if df.empty or len(df) < 50:
                 factor_results[ticker] = {'market_beta': 1.0, 'smb_loading': 0.0, 'hml_loading': 0.0, 'momentum': 0.0, 'alpha': 0.0}
                 continue
                 
            # 3. Calculate daily return
            df['Return'] = df['Close'].pct_change().dropna()
            
            # 4. Join with FF factors
            joined = df[['Return']].join(factors, how='inner').dropna()
            
            if len(joined) < 50:
                print(f"[{ticker}] Not enough date overlaps for regression.")
                factor_results[ticker] = {'market_beta': 1.1, 'smb_loading': 0.1, 'hml_loading': 0.1, 'momentum': 0.0, 'alpha': 0.0}
                continue
                
            # (Ri - Rf) = alpha + b*Mkt + s*SMB + h*HML
            # FF cols are: 'Mkt-RF', 'SMB', 'HML', 'RF'
            joined['ExcessReturn'] = joined['Return'] - joined['RF']
            
            X = joined[['Mkt-RF', 'SMB', 'HML']]
            y = joined['ExcessReturn']
            
            # Fit OLS
            model = LinearRegression()
            model.fit(X, y)
            
            beta = model.coef_[0]
            smb = model.coef_[1]
            hml = model.coef_[2]
            alpha = model.intercept_ # daily alpha. Can annualize it if needed.
            
            mom = momentum_dict.get(ticker, 0.0) if momentum_dict else 0.0
            
            factor_results[ticker] = {
                'market_beta': round(float(beta), 4),
                'smb_loading': round(float(smb), 4),
                'hml_loading': round(float(hml), 4),
                'momentum': round(float(mom), 4),
                'alpha': round(float(alpha * 252), 4) # Annualized alpha
            }
            
            print(f"[{ticker}] FF Factor Loadings: b={beta:.2f}, s={smb:.2f}, h={hml:.2f}")

        except Exception as e:
            print(f"Error computing FF loaders for {ticker}: {e}")
            factor_results[ticker] = {'market_beta': 1.0, 'smb_loading': 0.0, 'hml_loading': 0.0, 'momentum': 0.0, 'alpha': 0.0}

    return factor_results

if __name__ == "__main__":
    res = load_fama_french_factors(["NVDA", "AAPL"], momentum_dict={'NVDA': 0.5, 'AAPL': 0.1})
    print(res)
