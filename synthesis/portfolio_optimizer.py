import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
import random

def optimize_portfolio(ticker_signals: dict) -> dict:
    """
    Optimizes a portfolio using Markowitz MPT (Mean-Variance) that maximizes 
    the Sharpe ratio, blending historical returns with agent-synthesized signals.
    """
    tickers = list(ticker_signals.keys())
    if not tickers:
        print("No tickers provided for optimization.")
        return {}
        
    print(f"Optimizing portfolio for: {tickers}")
    
    # 1. Fetch 1yr daily returns for all tickers
    try:
        data = yf.download(tickers, period="1y", group_by='ticker')
        # Extract Adj Close or Close
        # Handle single ticker DataFrame structure vs multi ticker
        prices = pd.DataFrame()
        for t in tickers:
             if 'Close' in data[t]:
                  prices[t] = data[t]['Close']
             elif 'Adj Close' in data:
                  # For single ticker, yfinance returns different structure sometimes
                  prices[t] = data['Close']
                  
        prices = prices.dropna()
        
    except Exception as e:
        print(f"Error downloading history for optimization: {e}")
        return {}

    if prices.empty:
        print("No historical price data loaded.")
        return {}

    # 2. Compute expected returns (mean_historical_return)
    mu_hist = expected_returns.mean_historical_return(prices)
    
    # 3. Blend with sentiment scores (Kahneman Bias Adjusted)
    mu_blended = mu_hist.copy()
    for t in tickers:
        signals = ticker_signals.get(t, {})
        # Note: input expects adjusted_score from news_agent
        sentiment = signals.get('sentiment', 0.0)
        
        # Formula: historical * (1 + 0.3 * sentiment_score)
        mu_blended[t] = mu_hist[t] * (1 + 0.3 * sentiment)
        
    print(f"Blended Expected Returns (Kahneman-adjusted): \n{mu_blended}")

    # 4. Compute covariance matrix using Ledoit-Wolf shrinkage
    cov = CovarianceShrinkage(prices).ledoit_wolf()

    # 5. Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu_blended, cov)
    
    try:
        ef.max_sharpe()
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        exp_ret, vol, sharpe = performance
    except Exception as e:
        print(f"Optimization failed: {e}. Defaulting to equal weights.")
        weights = {t: 1.0/len(tickers) for t in tickers}
        exp_ret, vol, sharpe = 0.0, 0.0, 0.0

    # 6. Generate 5000 random portfolios for frontier visualization
    frontier_data = []
    try:
        num_portfolios = 5000
        for _ in range(num_portfolios):
            # Generate random weights that sum to 1
            rand_weights = np.random.dirichlet(np.ones(len(tickers)))
            
            # Portfolio expected return: np.dot(weights, mu_blended)
            p_ret = np.dot(rand_weights, mu_blended)
            
            # Portfolio volatility: sqrt(np.dot(weights.T, np.dot(cov, weights)))
            p_var = np.dot(rand_weights.T, np.dot(cov, rand_weights))
            p_vol = np.sqrt(p_var)
            
            p_sharpe = p_ret / p_vol if p_vol > 0 else 0.0
            
            # Subsample for display efficiency (e.g. 500 points for chart)
            if _ % 10 == 0:
                 frontier_data.append({
                     'return': round(float(p_ret), 4),
                     'volatility': round(float(p_vol), 4),
                     'sharpe': round(float(p_sharpe), 4)
                 })
                 
    except Exception as e:
        print(f"Error generating random portfolios: {e}")

    result = {
        'weights': weights,
        'sharpe_ratio': round(float(sharpe), 4),
        'expected_return': round(float(exp_ret), 4),
        'volatility': round(float(vol), 4),
        'frontier_data': frontier_data
    }
    
    print(f"Optimal Weights: {weights}")
    return result

if __name__ == "__main__":
    # Test
    signals = {
        'NVDA': {'sentiment': 0.8},
        'AAPL': {'sentiment': 0.2},
        'MSFT': {'sentiment': 0.5},
        'TSLA': {'sentiment': -0.4}
    }
    optimize_portfolio(signals)
