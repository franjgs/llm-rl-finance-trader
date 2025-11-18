# src/features.py
"""
Robust feature engineering module.
All indicators are forward-filled and cleaned to avoid NaN propagation.
Ready for LSTM, XGBoost, and any ML model.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np

# Use the same logger as the rest of the project
logger = logging.getLogger(__name__)


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-series momentum features (volatility-scaled)."""
    close = df["close"]
    for w in [20, 40, 60]:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"log_ret_{w}"] = np.log(close / close.shift(w))

    df["mom_signal_raw"] = np.sign(df["ret_40"])
    vol_20 = close.pct_change().rolling(20).std()
    df["mom_signal"] = df["mom_signal_raw"] / vol_20.replace(0, np.nan)

    # Forward fill + final fill to avoid NaN
    df["mom_signal"] = df["mom_signal"].ffill().fillna(0.0)
    return df


def add_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add realized volatility features (cleaned)."""
    ret = df["close"].pct_change()
    df["vol_20"] = ret.rolling(20).std().ffill().fillna(0.01)
    df["vol_60"] = ret.rolling(60).std().ffill().fillna(0.01)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI indicator with safe division and forward fill."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Safe RSI: forward fill + back fill + clamp
    df["rsi"] = df["rsi"].ffill().bfill().clip(0, 100)
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (no NaN possible)."""
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).cumsum().fillna(0)
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: generate full clean feature set.
    NO NaN EVER. Ready for LSTM/XGBoost.
    Logs via project logger (no print statements).
    """
    df = df.copy()

    df = add_momentum_features(df)
    df = add_vol_features(df)
    df = add_rsi(df)
    df = add_obv(df)

    # Lagged returns
    for lag in [1, 5, 10]:
        df[f"ret_lag_{lag}"] = df["close"].pct_change(lag)

    # Final robust cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0.0)

    # Professional logging instead of print
    nan_count = df.isna().sum().sum()
    logger.info(
        f"Feature generation completed → {len(df.columns)} columns | {nan_count} NaN remaining"
    )

    return df