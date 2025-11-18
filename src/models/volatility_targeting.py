# src/models/volatility_targeting.py
"""
Volatility targeting and exposure scaling module.
This module takes a DataFrame containing a pre-computed `clean_signal` (typically in range [-1, 1])
and scales it so that the resulting position has a target annualized volatility.
Key features:
- Automatically detects intraday vs daily data and annualizes volatility correctly
- Robust against misaligned indices, DataFrame columns, or broadcasting issues
- Bulletproof exposure calculation using .squeeze() + reindex
- Full protection against division by zero, infs, and extreme scaling
- Output columns: vol_annual, vol_scale_factor, exposure
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

def _infer_bars_per_day(df: pd.DataFrame) -> float:
    """
    Estimate the average number of bars (rows) per trading day from the datetime index.
  
    Returns
    -------
    float
        Average bars per day (e.g., 1 for daily, 6.5 for 1h data).
    """
    if df.empty:
        return 1.0
    idx = pd.to_datetime(df.index)
    counts_per_day = pd.Series(1, index=idx).groupby(idx.date).count()
    avg = counts_per_day.mean()
    return float(avg) if not pd.isna(avg) and avg > 0 else 1.0

def apply_vol_target(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply volatility targeting to scale `clean_signal` to a target annual volatility.
   
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'close' and 'clean_signal'.
        Must have a datetime index or a 'Date' column.
    params : dict
        Configuration options:
        - target_vol : float, desired annual volatility (e.g., 0.20 for 20%)
        - vol_lookback_bars : int, rolling window for volatility estimation (default 20)
        - max_leverage : float, maximum absolute exposure (default 3.0)
        - min_vol : float, floor for volatility to avoid division by zero (default 1e-6)
        - max_scale : float, maximum scaling factor magnitude (default 5.0)
        - return_raw : bool, keep intermediate columns for debugging (default False)
   
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - vol_annual : estimated annualized volatility
        - vol_scale_factor : scaling factor applied (vol_annual / target_vol)
        - exposure : final position after scaling and leverage cap
        - (optional) clean_signal_in
    """
    df = df.copy()
    # Default parameters
    target_vol = float(params.get("target_vol", 0.20))
    vol_lb = int(params.get("vol_lookback_bars", 20))
    max_leverage = float(params.get("max_leverage", 3.0))
    min_vol = float(params.get("min_vol", 1e-6))
    max_scale = float(params.get("max_scale", 5.0))
    return_raw = bool(params.get("return_raw", False))

    # ------------------------------------------------------------------ #
    # 1. REQUIRED COLUMNS + FORCE NUMERIC
    # ------------------------------------------------------------------ #
    required_cols = ["close", "clean_signal"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"apply_vol_target: missing required columns {missing}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["clean_signal"] = pd.to_numeric(df["clean_signal"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["close"])

    # ------------------------------------------------------------------ #
    # 2. Ensure datetime index
    # ------------------------------------------------------------------ #
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index(pd.to_datetime(df["Date"]))
            df = df.drop(columns=["Date"], errors="ignore")
        else:
            raise ValueError("DataFrame must have a datetime index or a 'Date' column")

    # ------------------------------------------------------------------ #
    # 3. Annualization factor
    # ------------------------------------------------------------------ #
    bars_per_day = _infer_bars_per_day(df)
    bars_per_year = bars_per_day * 252.0

    # ------------------------------------------------------------------ #
    # 4. Volatility calculation (log returns)
    # ------------------------------------------------------------------ #
    returns = np.log(df["close"]).diff().fillna(0.0)
    vol_per_bar = returns.rolling(vol_lb, min_periods=1).std()
    vol_annual = vol_per_bar * np.sqrt(bars_per_year)
    vol_annual = vol_annual.ffill().fillna(min_vol).clip(lower=min_vol)

    # ------------------------------------------------------------------ #
    # 5. Scaling factor — ¡¡CORREGIDO AQUÍ!! 
    # ------------------------------------------------------------------ #
    scale = vol_annual / target_vol                                # ← ¡¡AHORA SÍ!!
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    scale = scale.clip(-max_scale, max_scale)

    # ------------------------------------------------------------------ #
    # 6. BULLETPROOF exposure
    # ------------------------------------------------------------------ #
    scale_1d = scale.squeeze().reindex(df.index).fillna(1.0)
    clean_1d = df["clean_signal"].reindex(df.index).fillna(0.0)
    exposure = (scale_1d * clean_1d).clip(-max_leverage, max_leverage)

    # ------------------------------------------------------------------ #
    # 7. Assign results
    # ------------------------------------------------------------------ #
    df["vol_annual"] = vol_annual
    df["vol_scale_factor"] = scale_1d
    df["exposure"] = exposure
    if return_raw:
        df["clean_signal_in"] = clean_1d

    return df