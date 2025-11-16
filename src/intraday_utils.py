# src/intraday_utils.py
"""
Utilities to convert day/hour-based lookbacks to bar counts.
Fully compatible with both legacy configs (keys ending in _days/_hours)
and the current config_ensemble.yaml (keys already in bars).
"""

from typing import Dict, Any


def bars_per_trading_day(interval: str) -> float:
    """Return average number of bars per trading day for a given interval."""
    mapping = {
        "1m": 390, "5m": 78, "15m": 26, "30m": 13,
        "60m": 6.5, "1h": 6.5, "2h": 3.25, "4h": 1.625,
        "1d": 1.0
    }
    return mapping.get(interval.lower().replace(" ", ""), 1.0)


def days_to_bars(days: float, interval: str) -> int:
    """Convert calendar days (or fraction of day) to number of bars."""
    bpd = bars_per_trading_day(interval)
    return max(1, int(days * bpd))


def adjust_config_for_interval(cfg: Dict[str, Any], interval: str) -> Dict[str, Any]:
    """
    Adjust configuration when running intraday data.
    
    - If the config already contains bar-based keys → nothing is changed.
    - If legacy day/hour keys exist → they are converted to bars.
    - Daily data ("1d") is returned unchanged.
    """
    if interval == "1d":
        return cfg

    adj = cfg.copy()
    ens = adj["ensemble"]

    # === Momentum ===
    mom = ens["momentum"]
    if "lookback_days" in mom:
        mom["lookback_bars"] = days_to_bars(mom["lookback_days"], interval)
    if "vol_lookback_days" in mom:
        mom["vol_lookback_bars"] = days_to_bars(mom["vol_lookback_days"], interval)

    # === Volatility targeting ===
    vt = ens["volatility_targeting"]
    if "vol_lookback_days" in vt:
        vt["vol_lookback_bars"] = days_to_bars(vt["vol_lookback_days"], interval)

    # === XGBoost prediction horizon ===
    xgb = ens["xgboost"]
    if "prediction_horizon_hours" in xgb:
        hours_as_days = xgb["prediction_horizon_hours"] / 24.0
        xgb["prediction_horizon_bars"] = days_to_bars(hours_as_days, interval)

    # === LSTM sequence length ===
    lstm = ens["lstm"]
    if "sequence_days" in lstm:
        lstm["sequence_length"] = days_to_bars(lstm["sequence_days"], interval)

    # === RL Risk Overlay Sharpe lookback ===
    rl = ens.get("rl_risk_overlay", {})
    if "sharpe_lookback_days" in rl:
        rl["sharpe_lookback_bars"] = days_to_bars(rl["sharpe_lookback_days"], interval)

    return adj