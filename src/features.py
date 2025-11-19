"""
Advanced and robust feature engineering module.
No NaN values are left in the final output. 
Suitable for LSTM, XGBoost, and any ML model.

This version includes:
- Momentum features (multi-horizon)
- Volatility features (realized + Parkinson + Garman–Klass)
- RSI
- OBV
- Lagged returns
- ATR and True Range
- Price structure signals (rolling maxima/minima, z-score)
- Volatility-normalized features
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
#  BASIC HELPERS
# ----------------------------------------------------------------------

def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, -inf, forward-fill, then fill remaining NaN with zero."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0.0)
    return df


# ----------------------------------------------------------------------
#  MOMENTUM FEATURES
# ----------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add classical momentum and volatility-scaled momentum features."""
    close = df["close"]

    for w in [20, 40, 60]:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"log_ret_{w}"] = np.log(close / close.shift(w))

    df["mom_signal_raw"] = np.sign(df["ret_40"])
    vol_40 = close.pct_change().rolling(40).std()
    df["mom_signal"] = df["mom_signal_raw"] / vol_40.replace(0, np.nan)

    return df


# ----------------------------------------------------------------------
#  VOLATILITY FEATURES
# ----------------------------------------------------------------------

def add_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add realized and high-low-based volatility measures."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ret = close.pct_change()

    # Realized volatility
    df["vol_20"] = ret.rolling(20).std()
    df["vol_60"] = ret.rolling(60).std()

    # Parkinson volatility
    log_hl = np.log(high / low.replace(0, np.nan))
    parkinson_inner = (1.0 / (4 * np.log(2))) * (log_hl ** 2).rolling(20).mean()
    df["vol_parkinson"] = np.sqrt(parkinson_inner.clip(lower=0.0))

    # Garman–Klass volatility
    hl_term = (np.log(high / low.replace(0, np.nan))) ** 2
    oc_term = (np.log(close / close.shift(1))) ** 2
    gk_inner = 0.5 * hl_term.rolling(20).mean() - (2 * np.log(2) - 1) * oc_term.rolling(20).mean()
    df["vol_gk"] = np.sqrt(gk_inner.clip(lower=0.0))

    return df


# ----------------------------------------------------------------------
#  OSCILLATORS
# ----------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI technical indicator."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume."""
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).cumsum()
    return df


# ----------------------------------------------------------------------
#  TRUE RANGE / ATR
# ----------------------------------------------------------------------

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add True Range and ATR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["true_range"].rolling(period).mean()
    return df


# ----------------------------------------------------------------------
#  PRICE STRUCTURE FEATURES
# ----------------------------------------------------------------------

def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling maxima/minima and z-score features."""
    close = df["close"]

    for w in [20, 60, 120]:
        df[f"max_{w}"] = close.rolling(w).max()
        df[f"min_{w}"] = close.rolling(w).min()

        mean_w = close.rolling(w).mean()
        std_w = close.rolling(w).std()
        df[f"zscore_{w}"] = (close - mean_w) / std_w.replace(0, np.nan)

    return df


# ----------------------------------------------------------------------
#  LAGGED FEATURES
# ----------------------------------------------------------------------

def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged returns for short-term price dynamics."""
    for lag in [1, 2, 5, 10]:
        df[f"ret_lag_{lag}"] = df["close"].pct_change(lag)
    return df


# ----------------------------------------------------------------------
#  VOLATILITY NORMALIZED FEATURES
# ----------------------------------------------------------------------

def add_vol_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize momentum by local volatility."""
    if "vol_20" in df.columns:
        df["mom_over_vol"] = df["ret_40"] / df["vol_20"].replace(0, np.nan)
    return df


# ----------------------------------------------------------------------
#  MASTER PIPELINE
# ----------------------------------------------------------------------

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: generates a complete feature set.
    Ensures the final DataFrame contains zero NaN values.

    Steps:
    - Momentum (classic + log + vol-scaled)
    - Volatility (realized, Parkinson, Garman–Klass)
    - RSI
    - OBV
    - ATR and True Range
    - Price structure (rolling extremes + z-score)
    - Lagged returns
    - Volatility-normalized signals
    """
    df = df.copy()

    df = add_momentum_features(df)
    df = add_vol_features(df)
    df = add_rsi(df)
    df = add_obv(df)
    df = add_atr(df)
    df = add_price_structure(df)
    df = add_lagged_returns(df)
    df = add_vol_normalized(df)

    df = _safe_fill(df)

    logger.info(
        f"Feature generation completed → {len(df.columns)} columns | "
        f"No NaN remaining."
    )

    return df