"""
RL-inspired risk management overlay.

Behavior
--------
- Compute a vectorized multiplier series across the history (for debugging/inspection).
- Apply only the LAST multiplier to adjust exposure -> exposure_rl (real-time behavior).
- Store the multiplier series in df["risk_multiplier"] when configured.
- No in-place surprises (defensive copy).
- Emits info-level logs when the last multiplier != 1.0 for quick debugging.

Configuration keys expected (in your config):
- drawdown_reduce_level: float (negative, e.g. -0.03)
- drawdown_flat_level: float (negative, e.g. -0.06)
- sharpe_boost_threshold: float (positive, e.g. 0.6)
- sharpe_lookback_days: int (rolling window for Sharpe)
- max_boost: float (>=1.0)
- reduce_factor: float (0..1)
- store_multiplier: bool
- optional: max_exposure_abs: float (caps exposure_rl magnitude)
- periods_per_year: int (New: Annualization factor for Sharpe - 2016 for hourly, 252 for daily)
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class RLRiskOverlay:
    def __init__(self, config: dict):
        # Drawdown thresholds (negative numbers)
        self.reduce_level = float(config.get("drawdown_reduce_level", -0.03))
        self.flat_level = float(config.get("drawdown_flat_level", -0.06))

        # Sharpe boost parameters
        self.sharpe_boost = float(config.get("sharpe_boost_threshold", 0.6))
        # lookback name is 'sharpe_lookback_days' in your config; treat as number of bars
        self.lookback = int(config.get("sharpe_lookback_days", 30))
        
        # New: Annualization factor for Sharpe (2016 for 1h, 252 for 1d)
        # We default to 2016 since the current config uses "1h"
        self.periods_per_year = int(config.get("periods_per_year", 2016)) 

        # Multipliers
        self.max_boost = float(config.get("max_boost", 1.3))
        self.reduce_factor = float(config.get("reduce_factor", 0.5))

        # Optional debug storage and caps
        self.store_multiplier = bool(config.get("store_multiplier", True))
        self.max_exposure_abs = float(config.get("max_exposure_abs", np.inf))

        # Sanity checks
        if self.flat_level >= self.reduce_level:
            logger.warning(
                "rl_risk_overlay config: flat_level >= reduce_level; "
                "this will prevent reduction logic. Check config values."
            )
        if self.max_boost < 1.0:
            logger.warning("rl_risk_overlay: max_boost < 1.0 is unusual (should be >= 1.0).")

    def _rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """
        Compute rolling Sharpe (annualized). `returns` is a pd.Series of pct returns.
        If series is shorter than lookback, result will have NaNs at the start.
        """
        # use min_periods=1 to avoid all-NaN early - we'll rely on thresholding later
        mu = returns.rolling(self.lookback, min_periods=1).mean()
        sigma = returns.rolling(self.lookback, min_periods=1).std()
        
        # Adjusted annualization factor based on self.periods_per_year
        annualization_factor = np.sqrt(self.periods_per_year)
        
        sharpe = mu / (sigma + 1e-9) * annualization_factor
        return sharpe

    def apply(self, df: pd.DataFrame, equity: Optional[pd.Series]) -> pd.DataFrame:
        """
        Apply overlay to a DataFrame that contains at least an 'exposure' column (or clean_ensemble).
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame indexed by timestamps and containing 'exposure' or 'clean_ensemble' and optionally 'close'.
        equity : pd.Series or None
            Equity curve aligned with df.index. If None, a synthetic equity will be built from df values.
        Returns
        -------
        pd.DataFrame
            Copy of df with added columns:
            - exposure_rl : exposure adjusted for RL overlay (only last row changed)
            - risk_multiplier : full multiplier series (if store_multiplier True)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        df = df.copy()  # defensive copy

        # Ensure we have an index we can align to
        if df.index.empty:
            raise ValueError("df must have a non-empty DatetimeIndex")

        # Build or align equity series
        if equity is None:
            # Build temporary equity from available signal -> try exposure then clean_ensemble
            if "exposure" in df.columns:
                tmp_pos = df["exposure"].fillna(0.0)
            else:
                tmp_pos = df.get("clean_ensemble", pd.Series(0.0, index=df.index)).fillna(0.0)

            if "close" in df.columns:
                # synthetic equity: start at 1.0
                returns = df["close"].pct_change().fillna(0.0) * tmp_pos
                equity = (1.0 + returns).cumprod()
                equity.index = df.index
            else:
                # fallback flat equity
                equity = pd.Series(1.0, index=df.index)
        else:
            # align equity to df index, forward-fill/backfill as needed
            equity = pd.Series(equity).reindex(df.index)
            # Fix: Replacing deprecated fillna(method=...) with ffill/bfill
            equity = equity.ffill().bfill().fillna(1.0)


        # compute returns for sharpe and drawdown
        ret = equity.pct_change().fillna(0.0)
        sharpe = self._rolling_sharpe(ret)

        # compute drawdown series
        peak = equity.cummax()
        dd = (equity - peak) / peak

        # vectorized multiplier (for entire history) - start at 1.0
        mult = pd.Series(1.0, index=df.index)

        # flat if deep drawdown
        try:
            mult.loc[dd < self.flat_level] = 0.0
        except Exception:
            logger.exception("Error applying flat drawdown condition; skipping flat assignment.")

        # reduce when moderate drawdown
        cond_reduce = (dd < self.reduce_level) & (dd >= self.flat_level)
        if cond_reduce.any():
            mult.loc[cond_reduce] = self.reduce_factor

        # boost when Sharpe is high AND drawdown healthy
        cond_boost = (sharpe > self.sharpe_boost) & (dd > self.reduce_level)
        if cond_boost.any():
            mult.loc[cond_boost] = self.max_boost

        # prepare exposure source
        if "exposure" in df.columns:
            base_exposure = df["exposure"].copy()
        else:
            base_exposure = df.get("clean_ensemble", pd.Series(0.0, index=df.index)).copy()

        # clamp base_exposure to a safe range before applying RL multiplier
        if np.isfinite(self.max_exposure_abs):
            base_exposure = base_exposure.clip(-abs(self.max_exposure_abs), abs(self.max_exposure_abs))

        # initialize exposure_rl equal to base_exposure
        df["exposure_rl"] = base_exposure

        # apply only the last multiplier (real-time behaviour)
        last_idx = df.index[-1]
        last_mult = float(mult.loc[last_idx]) if last_idx in mult.index else 1.0
        if last_mult != 1.0:
            # compute and set last exposure with .loc to avoid SettingWithCopyWarning
            new_val = float(base_exposure.loc[last_idx]) * last_mult
            # optional cap after multiplication (ensures magnitude safe)
            if np.isfinite(self.max_exposure_abs):
                new_val = max(min(new_val, abs(self.max_exposure_abs)), -abs(self.max_exposure_abs))
            df.loc[last_idx, "exposure_rl"] = new_val
            logger.info(f"rl_risk_overlay: applied last multiplier {last_mult:.3f} at {last_idx}; "
                        f"prev_exposure={float(base_exposure.loc[last_idx]):.4f} -> exposure_rl={new_val:.4f}")

        # store full multiplier series if requested (useful for debugging)
        if self.store_multiplier:
            df["risk_multiplier"] = mult

        return df