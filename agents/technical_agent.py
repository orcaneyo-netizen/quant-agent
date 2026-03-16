import sys
import os
import pandas as pd
import numpy as np

# Add parent dir to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.yfinance_tool import get_historical_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    print("[TechnicalAgent] TA-Lib not found. Using Pandas fallbacks index computation.")
    HAS_TALIB = False

def compute_rsi_pandas(series: pd.Series, period=14) -> pd.Series:
    """Fallback RSI using pandas."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1+rs))

def compute_macd_pandas(series: pd.Series, slow=26, fast=12, signal=9) -> tuple:
    """Fallback MACD using pandas."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=signal, adjust=False).mean()
    return macd, exp3, macd - exp3 # macd, signal, hist

def compute_bollinger_pandas(series: pd.Series, period=20, std=2) -> tuple:
    """Fallback BBands using pandas."""
    sma = series.rolling(window=period).mean()
    rstd = series.rolling(window=period).std()
    upper = sma + (std * rstd)
    lower = sma - (std * rstd)
    return upper, sma, lower

def analyze_technicals(ticker: str) -> dict:
    """
    Computes technical indicators RSI, MACD, BBands, SMA.
    Detects buy/sell signal flags.
    """
    print(f"[{ticker}] Running TechnicalAgent...")
    # Fetch 1y for 12-1 mth momentum calculation
    df = get_historical_data(ticker, period="1y")
    
    if df.empty or len(df) < 50: 
        print(f"[{ticker}] Not enough history for technical analysis.")
        return {'signals': {}, 'pattern_flags': [], 'momentum': 0.0}
        
    close = df['Close']
    
    # Calculate Momentum: 12-1 month return (Carhart 4-factor definition)
    # We look back 252 trading days (approx 1 year). 1 month is approx 21 trading days.
    # momentum = close[-21] / close[-252] - 1
    momentum = 0.0
    try:
         if len(close) >= 252:
              momentum = (close.iloc[-21] / close.iloc[-252]) - 1
         elif len(close) >= 40: # estimate over lower frame
              momentum = (close.iloc[-10] / close.iloc[0]) - 1
    except Exception as e:
         print(f"Error calculating momentum for {ticker}: {e}")

    rsi_val = 50.0
    macd_val = 0.0
    macd_signal_val = 0.0
    upper_band = close.iloc[-1]
    lower_band = close.iloc[-1]
    sma50_val = 50.0
    sma200_val = 50.0
    
    # Compute using TA-Lib if available, or fall back to Pandas
    if HAS_TALIB:
        try:
            # TA-Lib wants numpy float64 arrays
            close_arr = close.values.astype(float)
            rsi_val = talib.RSI(close_arr, timeperiod=14)[-1]
            macd, macd_signal, _ = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_val = macd[-1]
            macd_signal_val = macd_signal[-1]
            up, mid, lw = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            upper_band = up[-1]
            lower_band = lw[-1]
            sma50 = talib.SMA(close_arr, timeperiod=50)
            sma200 = talib.SMA(close_arr, timeperiod=200) if len(close_arr) >= 200 else np.full(len(close_arr), np.nan)
            sma50_val = sma50[-1]
            sma200_val = sma200[-1] if not np.isnan(sma200[-1]) else sma50_val # fallback if < 200 days
        except Exception as e:
            print(f"[{ticker}] TA-Lib computation failed: {e}. Falling back to Pandas.")
            # fallback will trigger next
            HAS_TALIB_ERR = True
            
    if not HAS_TALIB:
        # Fallback to Pandas
        rsi_series = compute_rsi_pandas(close)
        macd, macd_signal, _ = compute_macd_pandas(close)
        up, mid, lw = compute_bollinger_pandas(close)
        sma50 = close.rolling(window=50).mean()
        # SMA200 requires 200 days, if not available use SMA50 or close
        sma200 = close.rolling(window=200).mean() if len(close) >= 200 else close
        
        rsi_val = rsi_series.iloc[-1]
        macd_val = macd.iloc[-1]
        macd_signal_val = macd_signal.iloc[-1]
        upper_band = up.iloc[-1]
        lower_band = lw.iloc[-1]
        sma50_val = sma50.iloc[-1]
        sma200_val = sma200.iloc[-1] if len(close) >= 200 else sma50_val

    # Detect signals
    pattern_flags = []
    
    # RSI Condition
    if rsi_val > 70:
        pattern_flags.append('OVERBOUGHT')
    elif rsi_val < 30:
        pattern_flags.append('OVERSOLD')
        
    # Crosses
    # To detect golden/death cross, compare current and previous values
    # We need index inside series
    if len(df) >= 200:
        # Check transition if golden cross (SMA50 crosses above SMA200)
        curr_cross = sma50_val > sma200_val
        if HAS_TALIB:
             prev_sma50 = talib.SMA(close_arr, timeperiod=50)[-2]
             prev_sma200 = talib.SMA(close_arr, timeperiod=200)[-2]
        else:
             prev_sma50 = sma50.iloc[-2]
             prev_sma200 = sma200.iloc[-2]
             
        prev_cross = prev_sma50 > prev_sma200
        if curr_cross and not prev_cross:
             pattern_flags.append('GOLDEN_CROSS')
        elif not curr_cross and prev_cross:
             pattern_flags.append('DEATH_CROSS')

    # MACD Condition
    if macd_val > macd_signal_val:
        macd_cond = 'BULLISH'
    else:
        macd_cond = 'BEARISH'
        
    price = close.iloc[-1]
    if price > upper_band:
        pattern_flags.append('BB_UPPER_BREAKOUT')
    elif price < lower_band:
        pattern_flags.append('BB_LOWER_BREAKOUT')

    signals = {
        'rsi': round(float(rsi_val), 2),
        'macd': round(float(macd_val), 2),
        'macd_signal': round(float(macd_signal_val), 2),
        'macd_trend': macd_cond,
        'sma50': round(float(sma50_val), 2),
        'sma200': round(float(sma200_val), 2),
        'price': round(float(price), 2)
    }

    print(f"[{ticker}] Technicals: RSI={signals.get('rsi')}, Signals={pattern_flags}")
    
    return {
        'signals': signals,
        'pattern_flags': pattern_flags,
        'momentum': round(float(momentum), 4)
    }

if __name__ == "__main__":
    import json
    res = analyze_technicals("NVDA")
    print(json.dumps(res, indent=2))
