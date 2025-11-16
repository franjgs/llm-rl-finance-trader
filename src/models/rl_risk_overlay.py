# src/models/rl_risk_overlay.py
"""
RL-inspired risk management overlay.
Reference: Deng et al. (2017).

This version preserves the original behaviour:
- Fully vectorized multiplier logic.
- Only the LAST SAMPLE is applied to exposure_rl.
- Multiplier series is still stored for debugging.
- No change to your model logic.

Only corrections:
- Removed pandas SettingWithCopyWarning.
- Safer .loc[] assignments.
- Defensive copy of df.
"""

import pandas as pd
import numpy as np


class RLRiskOverlay:
    def __init__(self, config):
        self.reduce_level = config["drawdown_reduce_level"]
        self.flat_level = config["drawdown_flat_level"]
        self.sharpe_boost = config["sharpe_boost_threshold"]
        self.lookback = config["sharpe_lookback_days"]   # unchanged

        self.max_boost = config.get("max_boost", 1.3)
        self.reduce_factor = config.get("reduce_factor", 0.5)
        self.store_multiplier = config.get("store_multiplier", True)

    # ---------------------------------------------------------
    # Rolling Sharpe
    # ---------------------------------------------------------
    def _rolling_sharpe(self, returns):
        """Compute rolling Sharpe ratio using the configured lookback window."""
        mu = returns.rolling(self.lookback).mean()
        sigma = returns.rolling(self.lookback).std()
        sharpe = mu / (sigma + 1e-9) * np.sqrt(252)
        return sharpe

    # ---------------------------------------------------------
    # Main overlay
    # ---------------------------------------------------------
    def apply(self, df, equity):
        """
        Apply the RL-inspired overlay.

        Behaviour preserved from your original version:
        - Vectorized multiplier is computed for entire history.
        - Only the LAST value of the multiplier is used to adjust exposure_rl.
        """
        df = df.copy()

        # Compute daily returns
        ret = equity.pct_change().fillna(0)
        sharpe = self._rolling_sharpe(ret)

        # Drawdown
        peak = equity.cummax()
        dd = (equity - peak) / peak

        # Full multiplier series (vectorized)
        mult = pd.Series(1.0, index=df.index)

        # Flat exposure under deep drawdown
        mult.loc[dd < self.flat_level] = 0.0

        # Reduce exposure under moderate drawdown
        cond_reduce = (dd < self.reduce_level) & (dd >= self.flat_level)
        mult.loc[cond_reduce] = self.reduce_factor

        # Boost exposure when Sharpe is high and drawdown is healthy
        cond_boost = (sharpe > self.sharpe_boost) & (dd > self.reduce_level)
        mult.loc[cond_boost] = self.max_boost

        # Apply ONLY the last-time-step multiplier
        df["exposure_rl"] = df["exposure"].copy()

        last_idx = df.index[-1]
        df.loc[last_idx, "exposure_rl"] = df.loc[last_idx, "exposure"] * mult.loc[last_idx]

        # Optional: store debug multiplier
        if self.store_multiplier:
            df["risk_multiplier"] = mult

        return df