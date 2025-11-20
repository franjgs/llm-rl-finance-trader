# src/features/__init__.py
"""
Advanced and robust feature engineering module — now with strict leakage control.

New in this version:
- Configurable feature mode: "strict" (no future data), "research" (allow future), "legacy" (old behavior)
- Explicit allow/block lists for fine-grained control
- All future-looking features are clearly marked and isolated
- Zero NaN guarantee preserved
"""

from __future__ import annotations
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# BASIC HELPERS
# ----------------------------------------------------------------------
def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf → NaN → forward-fill → fill remaining with 0.0."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0.0)
    return df


# ----------------------------------------------------------------------
# MOMENTUM FEATURES (no future leakage)
# ----------------------------------------------------------------------
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    for w in [20, 40, 60]:
        df[f"ret_{w}"] = close.pct_change(w)
        df[f"log_ret_{w}"] = np.log(close / close.shift(w))
    df["mom_signal_raw"] = np.sign(df["ret_40"])
    vol_40 = close.pct_change().rolling(40).std()
    df["mom_signal"] = df["mom_signal_raw"] / vol_40.replace(0, np.nan)
    return df


# ----------------------------------------------------------------------
# VOLATILITY FEATURES (no future leakage)
# ----------------------------------------------------------------------
def add_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ret = close.pct_change()

    df["vol_20"] = ret.rolling(20).std()
    df["vol_60"] = ret.rolling(60).std()

    # Parkinson
    log_hl = np.log(high / low.replace(0, np.nan))
    parkinson_inner = (1.0 / (4 * np.log(2))) * (log_hl ** 2).rolling(20).mean()
    df["vol_parkinson"] = np.sqrt(parkinson_inner.clip(lower=0.0))

    # Garman–Klass
    hl_term = (np.log(high / low.replace(0, np.nan))) ** 2
    oc_term = (np.log(close / close.shift(1))) ** 2
    gk_inner = 0.5 * hl_term.rolling(20).mean() - (2 * np.log(2) - 1) * oc_term.rolling(20).mean()
    df["vol_gk"] = np.sqrt(gk_inner.clip(lower=0.0))

    return df


# ----------------------------------------------------------------------
# OSCILLATORS, ATR, PRICE STRUCTURE, LAGGED RETURNS (no future leakage)
# ----------------------------------------------------------------------
def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).cumsum()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["true_range"].rolling(period).mean()
    return df


def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    for w in [20, 60, 120]:
        df[f"max_{w}"] = close.rolling(w).max()
        df[f"min_{w}"] = close.rolling(w).min()
        mean_w = close.rolling(w).mean()
        std_w = close.rolling(w).std()
        df[f"zscore_{w}"] = (close - mean_w) / std_w.replace(0, np.nan)
    return df


def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 5, 10]:
        df[f"ret_lag_{lag}"] = df["close"].pct_change(lag)
    return df


def add_vol_normalized(df: pd.DataFrame) -> pd.DataFrame:
    if "vol_20" in df.columns and "ret_40" in df.columns:
        df["mom_over_vol"] = df["ret_40"] / df["vol_20"].replace(0, np.nan)
    return df


# ----------------------------------------------------------------------
# FUTURE-LOOKING FEATURES (ONLY ENABLED IN RESEARCH MODE)
# ----------------------------------------------------------------------
def add_future_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    These features use future price information → FORBIDDEN in live/strict mode.
    Only added when explicitly allowed via config.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Forward returns (8-hour horizon example)
    df["ret_8h_fwd"] = close.pct_change(8).shift(-8)
    df["vol_8h_fwd"] = close.pct_change().rolling(8).std().shift(-8)
    df["range_8h_fwd"] = (high.rolling(8).max() - low.rolling(8).min()).shift(-8)
    df["high_8h_fwd"] = high.rolling(8).max().shift(-8)
    df["low_8h_fwd"] = low.rolling(8).min().shift(-8)

    logger.info("Added future-looking features (ret_8h_fwd, vol_8h_fwd, etc.) — RESEARCH MODE ONLY")
    return df


# ----------------------------------------------------------------------
# MASTER PIPELINE — NOW WITH LEAKAGE CONTROL
# ----------------------------------------------------------------------
def generate_features(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Main feature generation function with configurable leakage protection.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    config : dict, optional
        Feature configuration. Expected keys under "features":
          - mode: "strict" | "research" | "legacy"
          - allow_future_features: bool or list of feature names
          - block_features: list of feature names to remove

    Returns
    -------
    pd.DataFrame
        Feature-rich dataframe with zero NaN values
    """
    if config is None:
        config = {}
    
    mode = config.get("feature_mode", "strict").lower()
    
    # Define what is considered a dangerous future feature
    FUTURE_FEATURES = {
        "ret_8h_fwd", "vol_8h_fwd", "range_8h_fwd",
        "high_8h_fwd", "low_8h_fwd", "ret_8h", "vol_8h"
    }

    df = df.copy()

    # === Always-safe features (100% real-time) ===
    df = add_momentum_features(df)
    df = add_vol_features(df)
    df = add_rsi(df)
    df = add_obv(df)
    df = add_atr(df)
    df = add_price_structure(df)
    df = add_lagged_returns(df)
    df = add_vol_normalized(df)

    # === Future features: only in research mode ===
    if mode == "research":
        df = add_future_features(df)
    else:
        # strict or production → ensure NO future features exist
        for col in FUTURE_FEATURES:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Extra manual blocks (safety net)
    if config.get("features", {}).get("block_features", []) is not None:
        for col in config.get("features", {}).get("block_features", []):
            df.drop(columns=[col], inplace=True, errors="ignore")

    df = _safe_fill(df)

    future_count = sum(1 for col in df.columns if col in FUTURE_FEATURES)
    logger.info(
        f"Feature generation completed → {len(df.columns)} columns | "
        f"Mode: {mode.upper()} | Future features: {future_count}"
    )
    return df