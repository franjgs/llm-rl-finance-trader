# src/features.py
"""
Common feature engineering functions used by multiple models.
All based on proven indicators from literature.
"""
import pandas as pd
import numpy as np

def add_momentum_features(df):
    """Add time-series momentum features (20-60 days, scaled by vol)."""
    close = df["close"]
    for w in [20, 40, 60]:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"log_ret_{w}"] = np.log(close / close.shift(w))
    df["mom_signal_raw"] = np.sign(df["ret_40"])
    vol_20 = close.pct_change().rolling(20).std()
    df["mom_signal"] = df["mom_signal_raw"] / vol_20.replace(0, np.nan).fillna(0.01)
    return df

def add_vol_features(df):
    """Add realized volatility features."""
    ret = df["close"].pct_change()
    df["vol_20"] = ret.rolling(20).std()
    df["vol_60"] = ret.rolling(60).std()
    return df

def add_rsi(df, period=14):
    """Add RSI indicator."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan).fillna(1)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_obv(df):
    """Add On-Balance Volume."""
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).cumsum()
    return df

def generate_features(df):
    """Generate full feature set for ML models."""
    df = add_momentum_features(df)
    df = add_vol_features(df)
    df = add_rsi(df)
    df = add_obv(df)
    for lag in [1, 5]:
        df[f"ret_lag_{lag}"] = df["close"].pct_change(lag)
    return df