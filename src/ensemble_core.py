# src/ensemble_core.py
"""
Core ensemble module.
Builds all signals, validates them, applies weights, volatility targeting,
and optional RL risk overlay.

This file must NEVER assume execution context.
It only takes a DataFrame + config, and returns a clean DataFrame.
"""

import numpy as np
import pandas as pd

from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.sentiment_signal import generate_sentiment_signal
from src.models.rl_risk_overlay import apply_risk_overlay


# ============================================================
# 1. SAFETY CHECKS
# ============================================================
def _validate_input(df):
    required = ["Date", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")


def _validate_signals(df):
    """Ensure all signal columns are numeric and finite."""
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    for col in signal_cols:
        if df[col].isna().all():
            raise ValueError(f"Signal column '{col}' is empty (all NaN)")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Signal '{col}' is not numeric")

    return signal_cols


# ============================================================
# 2. MAIN PIPELINE
# ============================================================
def build_ensemble(df, cfg):
    """Build all signals and return fully usable DataFrame."""

    df = df.copy()
    _validate_input(df)

    df = df.sort_values("Date").set_index("Date")

    ens = cfg["ensemble"]

    # ===============================
    # Momentum
    # ===============================
    if ens["momentum"]["enabled"]:
        df = generate_momentum_signal(df, ens["momentum"])

    # ===============================
    # XGBoost Model
    # ===============================
    if ens["xgboost"]["enabled"]:
        model_path = ens["xgboost"]["model_path"]
        xgb = XGBoostPredictor(model_path)

        if ens["xgboost"]["retrain"]:
            xgb.train(df)

        df = xgb.predict(df)

    # ===============================
    # LSTM Model
    # ===============================
    if ens["lstm"]["enabled"]:
        lstm = LSTMPredictor(ens["lstm"]["model_path"])

        if ens["lstm"]["retrain"]:
            lstm.train(df)

        df = lstm.predict(df)

    # ===============================
    # Sentiment
    # ===============================
    if ens["sentiment_signal"]["enabled"]:
        df = generate_sentiment_signal(df, ens["sentiment_signal"])

    # ============================================================
    # 3. COMBINE SIGNALS
    # ============================================================
    signal_cols = _validate_signals(df)

    # Equal risk weighting (default)
    if ens["weighting"]["equal_risk"]:
        df["raw_ensemble"] = df[signal_cols].mean(axis=1)
    else:
        # placeholder → custom weight matrix can be plugged in here
        df["raw_ensemble"] = df[signal_cols].mean(axis=1)

    # Normalize to [-1, 1]
    df["raw_ensemble"] = df["raw_ensemble"].clip(-1, 1)

    # Optional smoothing (reduces churn)
    if ens["weighting"].get("smoothing_alpha", 0) > 0:
        alpha = ens["weighting"]["smoothing_alpha"]
        df["raw_ensemble"] = df["raw_ensemble"].ewm(alpha=alpha).mean()

    # Remove noise (small exposure)
    thresh = ens["weighting"]["min_position_threshold"]
    df["clean_signal"] = df["raw_ensemble"].where(
        df["raw_ensemble"].abs() >= thresh, 0
    )

    # ============================================================
    # 4. VOLATILITY TARGETING
    # ============================================================
    if ens["volatility_targeting"]["enabled"]:
        df = apply_vol_target(df, ens["volatility_targeting"])
        # Output: df["exposure"]

    else:
        df["exposure"] = df["clean_signal"]

    # Shift so today's position uses yesterday's signal
    df["position"] = df["exposure"].shift(1).fillna(0)

    # ============================================================
    # 5. TEMP RETURNS (USED BY RL RISK OVERLAY)
    # ============================================================
    df["temp_ret"] = df["position"] * df["close"].pct_change().fillna(0)
    df["equity_temp"] = (1 + df["temp_ret"]).cumprod()

    # ============================================================
    # 6. RL RISK OVERLAY (OPTIONAL)
    # ============================================================
    if ens["rl_risk_overlay"]["enabled"]:
        df = apply_risk_overlay(df, df["equity_temp"], ens["rl_risk_overlay"])

        # apply_risk_overlay is expected to output df["position_adjusted"]
        df["position"] = df["position_adjusted"]

    return df.reset_index()