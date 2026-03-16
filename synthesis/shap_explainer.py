import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import shap

def explain_decisions(ticker_features: dict, weights: dict) -> dict:
    """
    Trains a local model using GradientBoosting to explain why allocations were made,
    using SHAP TreeExplainer to find top feature impacts on portfolio weight.
    """
    print("Generating SHAP Explanations...")
    
    # 1. Prepare datarows
    rows = []
    tickers = list(ticker_features.keys())
    
    for t in tickers:
        f = ticker_features.get(t, {})
        weight = weights.get(t, 0.0)
        
        row = {
            'ticker': t,
            'market_beta': f.get('ff_factors', {}).get('market_beta', 1.0),
            'smb_loading': f.get('ff_factors', {}).get('smb_loading', 0.0),
            'hml_loading': f.get('ff_factors', {}).get('hml_loading', 0.0),
            'momentum': f.get('ff_factors', {}).get('momentum', 0.0),
            'alpha': f.get('ff_factors', {}).get('alpha', 0.0),
            'adjusted_sentiment_score': f.get('adjusted_sentiment_score', 0.0),
            'pe_ratio': f.get('fundamentals', {}).get('trailing_pe', 0.0) if f.get('fundamentals') else 0.0,
            'revenue_growth': f.get('fundamentals', {}).get('revenue_growth', 0.0) if f.get('fundamentals') else 0.0,
            # Target
            'weight_allocation': weight
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df = df.fillna(0) # clean NaN
    
    if df.empty or len(df) < 2:
        print("Not enough samples for SHAP analysis.")
        return {}
        
    X = df.drop(columns=['ticker', 'weight_allocation'])
    y = df['weight_allocation']

    # GradientBoosting requires multiple samples. Pad dataframe if needed.
    # Duplicating rows with Gaussian noise to create local explainer dataset
    if len(X) < 10:
        multiplier = 20
        X_aug = pd.concat([X] * multiplier, ignore_index=True)
        y_aug = pd.concat([y] * multiplier, ignore_index=True)
        # Add slight noise
        noise = np.random.normal(0, 0.01, X_aug.shape)
        X_aug = X_aug + noise
    else:
        X_aug = X.copy()
        y_aug = y.copy()

    try:
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
        model.fit(X_aug, y_aug)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X) # on original rows
        
        explanations = {}
        features = list(X.columns)
        
        for i, t in enumerate(tickers):
            # Row index matches ticker index
            scores = shap_values[i]
            # Pair with column names
            importance = []
            for j, f_name in enumerate(features):
                importance.append({
                    'feature': f_name,
                    'shap_impact': round(float(scores[j]), 4),
                    'value': round(float(X.iloc[i][f_name]), 4)
                })
            # Sort by absolute shape impact
            importance = sorted(importance, key=lambda x: abs(x['shap_impact']), reverse=True)
            explanations[t] = importance[:3] # Top 3
            
        print("SHAP explanations generated.")
        return explanations
        
    except Exception as e:
         print(f"Error in SHAP explainer: {e}")
         return {t: [] for t in tickers}

if __name__ == "__main__":
    ticker_features = {
        'NVDA': {
            'sentiment_score': 0.8,
            'fundamentals': {'trailing_pe': 40, 'revenue_growth': 0.5, 'debt_to_equity': 0.1},
            'signals': {'rsi': 65, 'macd': 2.0}
        },
        'AAPL': {
            'sentiment_score': 0.4,
            'fundamentals': {'trailing_pe': 30, 'revenue_growth': 0.1, 'debt_to_equity': 1.5},
            'signals': {'rsi': 45, 'macd': -0.5}
        }
    }
    weights = {'NVDA': 0.7, 'AAPL': 0.3}
    res = explain_decisions(ticker_features, weights)
    import json
    print(json.dumps(res, indent=2))
