# src/models/momentum.py
"""
Time-series momentum signal generator.
Design goals / notes:
- Produces a single column: `signal_momentum` in range [-1..1] (or scaled).
- Uses log-returns by default (more stable for composition).
- Scales by realized volatility (vol-targeting) so different assets are comparable.
- Optional smoothing and thresholding to avoid micro-churn.
- Safe with NaNs and small-volatility regimes.
Reference: Moskowitz, Ooi & Pedersen (2012) — time-series momentum (implementation adapted).
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


def generate_momentum_signal(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Add momentum signal columns to df (expects df contains 'close' and index or column 'Date').

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a 'close' column.
    params : dict
        Configuration dictionary (see docstring in the module header).

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - signal_momentum : final scaled and smoothed momentum signal
        - clean_signal    : guaranteed Series (identical to signal_momentum)
        - (optional) mom_raw, vol
    """
    df = df.copy()

    lb = int(params.get("lookback_bars", 20))
    vol_lb = int(params.get("vol_lookback_bars", 20))
    use_log = bool(params.get("use_log_returns", True))
    target_vol = params.get("target_vol", None)
    max_exposure = float(params.get("max_exposure", 1.0))
    smoothing_alpha = float(params.get("smoothing_alpha", 0.0))
    min_abs_mom = float(params.get("min_abs_mom", 0.0))
    fillna_method = params.get("fillna_method", "ffill")
    return_raw = bool(params.get("return_raw", False))

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column")

    # Momentum calculation
    if use_log:
        logp = np.log(df["close"]).replace([np.inf, -np.inf], np.nan)
        mom_raw = logp.diff(lb)
    else:
        mom_raw = df["close"].pct_change(lb)

    # Volatility estimation
    ret = np.log(df["close"]).diff() if use_log else df["close"].pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.rolling(vol_lb, min_periods=1).std()
    vol = vol.replace(0, np.nan)

    direction = np.sign(mom_raw.fillna(0.0))

    if target_vol is not None:
        exposure_raw = direction * (float(target_vol) / vol)
    else:
        mom_z = mom_raw.abs().div(vol).replace([np.inf, -np.inf], np.nan)
        exposure_raw = direction * mom_z.fillna(0.0)

    # Fill NaNs (deprecated method removed)
    if fillna_method == "ffill":
        exposure_raw = exposure_raw.ffill().fillna(0.0)
    else:
        exposure_raw = exposure_raw.fillna(0.0)

    # Threshold and smoothing
    exposure_thresh = exposure_raw.where(exposure_raw.abs() >= min_abs_mom, 0.0)
    exposure_smoothed = (
        exposure_thresh.ewm(alpha=smoothing_alpha).mean()
        if 0.0 < smoothing_alpha < 1.0
        else exposure_thresh
    )

    # Final signal
    signal = exposure_smoothed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal = signal.clip(-abs(max_exposure), abs(max_exposure))

    # Critical fix: ensure clean_signal is always a proper Series
    df["signal_momentum"] = signal

    # ← THIS IS THE ONLY LINE THAT CHANGED
    df["clean_signal"] = signal.copy()          # signal is already a Series → perfect
    # ← END OF CHANGE

    # Optional debug columns
    if return_raw:
        df["mom_raw"] = mom_raw
        df["vol"] = vol

    return df