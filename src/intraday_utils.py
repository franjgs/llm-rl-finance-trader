# src/intraday_utils.py
"""
Utilities to automatically convert day-based lookbacks to bar count based on interval.
Used by all models to work seamlessly on 1h, 30m, 15m, daily, etc.
"""
def bars_per_trading_day(interval):
    """Return average number of bars per trading day for a given interval."""
    mapping = {
        "1m": 390, "5m": 78, "15m": 26, "30m": 13,
        "60m": 6.5, "1h": 6.5, "2h": 3.25, "4h": 1.625,
        "1d": 1.0
    }
    return mapping.get(interval.lower().replace(" ", ""), 1.0)

def days_to_bars(days, interval):
    """Convert calendar days to number of bars."""
    bpd = bars_per_trading_day(interval)
    return max(1, int(days * bpd))

def adjust_config_for_interval(cfg, interval):
    """Adjust all day-based lookbacks to bar count for intraday data."""
    if interval == "1d":
        return cfg

    adj = cfg.copy()
    ens = adj["ensemble"]

    ens["momentum"]["lookback_bars"] = days_to_bars(ens["momentum"]["lookback_days"], interval)
    ens["momentum"]["vol_lookback_bars"] = days_to_bars(ens["momentum"]["vol_lookback_days"], interval)

    ens["volatility_targeting"]["vol_lookback_bars"] = days_to_bars(ens["volatility_targeting"].get("vol_lookback_days", 20), interval)

    ens["xgboost"]["prediction_horizon_bars"] = days_to_bars(ens["xgboost"]["prediction_horizon_hours"] / 24, interval)

    ens["lstm"]["sequence_length"] = days_to_bars(ens["lstm"]["sequence_days"], interval)

    ens["rl_risk_overlay"]["sharpe_lookback_bars"] = days_to_bars(ens["rl_risk_overlay"]["sharpe_lookback_days"], interval)

    return adj