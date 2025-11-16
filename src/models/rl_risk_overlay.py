# src/models/rl_risk_overlay.py
"""
RL-inspired risk management overlay.
Reference: Deng et al. (2017).
Vectorized implementation.
"""

import pandas as pd
import numpy as np


class RLRiskOverlay:
    def __init__(self, config):
        self.reduce_level = config["drawdown_reduce_level"]
        self.flat_level = config["drawdown_flat_level"]
        self.sharpe_boost = config["sharpe_boost_threshold"]
        self.lookback = config["sharpe_lookback_days"]

        self.max_boost = config.get("max_boost", 1.3)
        self.reduce_factor = config.get("reduce_factor", 0.5)
        self.store_multiplier = config.get("store_multiplier", True)

    # ---------------------------------------------------------
    # Compute rolling Sharpe
    # ---------------------------------------------------------

    def _rolling_sharpe(self, returns):
        mu = returns.rolling(self.lookback).mean()
        sigma = returns.rolling(self.lookback).std()
        sharpe = mu / (sigma + 1e-9) * np.sqrt(252)
        return sharpe

    # ---------------------------------------------------------
    # Main overlay
    # ---------------------------------------------------------

    def apply(self, df, equity):
        """
        df: dataframe with 'exposure' column
        equity: account equity curve (same index)
        """

        # Compute returns and sharpe
        ret = equity.pct_change().fillna(0)
        sharpe = self._rolling_sharpe(ret)

        # Compute drawdown
        peak = equity.cummax()
        dd = (equity - peak) / peak

        # Initialize multiplier array
        mult = pd.Series(1.0, index=df.index)

        # --- FLAT during deep drawdown ---
        mult[dd < self.flat_level] = 0.0

        # --- Reduce exposure during moderate drawdown ---
        cond_reduce = (dd < self.reduce_level) & (dd >= self.flat_level)
        mult[cond_reduce] = self.reduce_factor

        # --- Boost exposure when Sharpe is great and drawdown healthy ---
        cond_boost = (sharpe > self.sharpe_boost) & (dd > self.reduce_level)
        mult[cond_boost] = self.max_boost

        # Apply
        df["exposure_rl"] = df["exposure"] * mult

        # optional debugging
        if self.store_multiplier:
            df["risk_multiplier"] = mult

        return df