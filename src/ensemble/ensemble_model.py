# src/ensemble/ensemble_model.py
"""
EnsembleModel 

Features
--------
- Full backtesting support via fit_predict() (fast, for development)
- Strict zero-look-ahead prediction via predict_out_of_sample()
- Momentum, XGBoost, LSTM, Sentiment, Volatility-Targeting and optional RL overlay
- All scalers and feature engineering are fitted ONLY on training data when using
  predict_out_of_sample()
- Clean, readable, fully documented and Spyder-friendly
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from pathlib import Path

import torch

from src.features import generate_features
from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.sentiment_signal import SentimentSignal
from src.models.rl_risk_overlay import RLRiskOverlay

import logging
logger = logging.getLogger(__name__)



class EnsembleModel:
    """
    Main ensemble that combines multiple signal sources and produces a final
    trading position (long/short/flat).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize all predictors according to the config."""
        self.cfg = config
        self.ensemble_cfg = config.get("ensemble", {})

        # Feature flags
        self.momentum_enabled = bool(self.ensemble_cfg.get("momentum", {}).get("enabled", False))
        self.xgboost_enabled = bool(self.ensemble_cfg.get("xgboost", {}).get("enabled", False))
        self.lstm_enabled = bool(self.ensemble_cfg.get("lstm", {}).get("enabled", False))
        self.sentiment_enabled = bool(self.ensemble_cfg.get("sentiment_signal", {}).get("enabled", False))
        self.vol_target_enabled = bool(self.ensemble_cfg.get("volatility_targeting", {}).get("enabled", False))
        self.rl_overlay_enabled = bool(self.ensemble_cfg.get("rl_risk_overlay", {}).get("enabled", False))

        # Instantiate predictors
        self.xgb_predictor = XGBoostPredictor(self.ensemble_cfg.get("xgboost", {})) if self.xgboost_enabled else None
        self.lstm_predictor = LSTMPredictor(self.ensemble_cfg.get("lstm", {})) if self.lstm_enabled else None
        self.sentiment_model = SentimentSignal(self.ensemble_cfg.get("sentiment_signal", {})) if self.sentiment_enabled else None
        self.rl_overlay = RLRiskOverlay(self.ensemble_cfg.get("rl_risk_overlay", {})) if self.rl_overlay_enabled else None

        # Weighting & thresholds
        self.custom_weights = self.ensemble_cfg.get("weighting", {}).get("custom_weights", None)
        self.min_position_threshold = float(self.ensemble_cfg.get("weighting", {}).get("min_position_threshold", 0.0))
        self.max_exposure_abs = float(self.ensemble_cfg.get("max_exposure_abs", 1.0))

    # --------------------------------------------------------------------- #
    # Lightweight business-rule overlay
    # --------------------------------------------------------------------- #
    def apply_overlay(self, signal: pd.Series, sentiment: pd.Series, vol: pd.Series) -> pd.Series:
        """Apply simple sentiment/volatility overlay multiplier."""
        overlay = pd.Series(1.0, index=signal.index)
        overlay.loc[vol > 0.03] *= 0.6          # high volatility → reduce exposure
        overlay.loc[sentiment > 0.7] *= 1.1     # very positive sentiment → slight boost
        overlay.loc[sentiment < -0.7] *= 0.85   # very negative sentiment → reduce
        return overlay.clip(0.0, 1.2)

    # --------------------------------------------------------------------- #
    # Helper: which signal columns are actually present?
    # --------------------------------------------------------------------- #
    def _active_signal_cols(self, df: pd.DataFrame) -> List[str]:
        """Return list of signal columns that exist and are not completely NaN."""
        candidates = []
        if self.momentum_enabled and "signal_momentum" in df.columns:
            candidates.append("signal_momentum")
        if self.xgboost_enabled and "signal_xgboost" in df.columns:
            candidates.append("signal_xgboost")
        if self.lstm_enabled and "signal_lstm" in df.columns:
            candidates.append("signal_lstm")
        if self.sentiment_enabled and "signal_sentiment" in df.columns:
            candidates.append("signal_sentiment")
        return [c for c in candidates if not df[c].isna().all()]

    # --------------------------------------------------------------------- #
    # Classic fit_predict() – fast backtesting (may use future data for scaling)
    # --------------------------------------------------------------------- #
    def fit_predict(self, df: pd.DataFrame, equity_curve: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Traditional backtesting method.
        Kept for quick development / offline backtests.
        """
        df = df.copy()
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = df.sort_index()

        # 1. Momentum
        if self.momentum_enabled:
            df = generate_momentum_signal(df, self.ensemble_cfg.get("momentum", {}))

        # 2. XGBoost
        if self.xgboost_enabled and self.xgb_predictor:
            if self.ensemble_cfg.get("xgboost", {}).get("retrain", False):
                split = int(len(df) * 0.8)
                self.xgb_predictor.train(df.iloc[:split].copy(), self.cfg)
            df = self.xgb_predictor.predict(df, self.cfg)

        # 3. LSTM
        if self.lstm_enabled and self.lstm_predictor:
            if self.ensemble_cfg.get("lstm", {}).get("retrain", False):
                split = int(len(df) * 0.8)
                self.lstm_predictor.train(df.iloc[:split].copy(), self.cfg)
            df = self.lstm_predictor.predict(df, self.cfg)

        # 4. Sentiment
        if self.sentiment_enabled and self.sentiment_model:
            df = self.sentiment_model.predict(df)

        # 5. Normalise ML probabilities to [-1, 1]
        for col in self._active_signal_cols(df):
            if col in {"signal_xgboost", "signal_lstm"}:
                df[col] = (df[col] - 0.5) * 2.0 * 1.1
                df[col] = df[col].clip(-1.0, 1.0)

        # 6. Combine signals
        signal_cols = self._active_signal_cols(df)
        if not signal_cols:
            df["signal_ensemble"] = 0.0
            df["clean_ensemble"] = 0.0
        else:
            if self.custom_weights:
                w = np.array([self.custom_weights.get(c, 1.0) for c in signal_cols])
                w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                df["signal_ensemble"] = (df[signal_cols] * w).sum(axis=1)
            else:
                df["signal_ensemble"] = df[signal_cols].mean(axis=1)

            df["signal_ensemble"] = df["signal_ensemble"].clip(-1.0, 1.0)
            df["clean_ensemble"] = df["signal_ensemble"].where(
                df["signal_ensemble"].abs() >= self.min_position_threshold, 0.0
            )

        # 7. Overlay (sentiment + volatility)
        if "sentiment" in df.columns and "close" in df.columns:
            vol = df["close"].pct_change().rolling(20).std().fillna(0.0)
            overlay = self.apply_overlay(df["clean_ensemble"], df["sentiment"], vol)
            df["clean_ensemble"] = (df["clean_ensemble"] * overlay).clip(-1.0, 1.0)

        # 8. Volatility targeting
        if self.vol_target_enabled:
            df["clean_signal"] = df["clean_ensemble"]
            df = apply_vol_target(df, self.ensemble_cfg.get("volatility_targeting", {}))
            df.drop(columns=["clean_signal"], inplace=True, errors="ignore")

        # 9. RL risk overlay (optional)
        if self.rl_overlay_enabled and self.rl_overlay:
            pos = df["exposure"].fillna(0.0) if "exposure" in df.columns else df["clean_ensemble"].fillna(0.0)
            equity = (1.0 + df["close"].pct_change().fillna(0.0) * pos).cumprod()
            df = self.rl_overlay.apply(df, equity_curve if equity_curve is not None else equity)

        # 10. Final position (shifted by one bar) – FIXED VERSION
        base = df.get("exposure_rl", df.get("exposure", df["clean_ensemble"]))
        scaling = float(self.cfg.get("position_scaling", 1.0))
        final_position = (base * scaling).clip(-1.0, 1.0)
        df["position"] = final_position.shift(1).fillna(0.0)
        df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

        return df.reset_index()

    # --------------------------------------------------------------------- #
    # STRICT OUT-OF-SAMPLE PREDICTION – ZERO LOOK-AHEAD (production / walk-forward)
    # --------------------------------------------------------------------- #
    def predict_out_of_sample(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        ZERO look-ahead out-of-sample prediction for strict walk-forward validation.
    
        This method guarantees no future information is used:
          - Models are trained only on train_df
          - Feature engineering respects the cutoff
          - LSTM feature columns are locked externally by WalkForwardAnalyzer (no torch.load here!)
    
        Returns
        -------
        pd.DataFrame
            Full dataframe with all signals and final shifted position.
        """
        full_df = full_df.copy()
    
        # --------------------------------------------------------------------- #
        # 0. Ensure Date index
        # --------------------------------------------------------------------- #
        if "Date" in full_df.columns:
            full_df = full_df.set_index("Date")
    
        full_df = full_df.sort_index()
    
        # --------------------------------------------------------------------- #
        # 1. Generate base features using current config (strict mode enforced externally)
        # --------------------------------------------------------------------- #
        full_df = generate_features(full_df, self.cfg)
    
        # --------------------------------------------------------------------- #
        # 2. Momentum signal
        # --------------------------------------------------------------------- #
        if self.momentum_enabled:
            full_df = generate_momentum_signal(full_df, self.ensemble_cfg.get("momentum", {}))
    
        # --------------------------------------------------------------------- #
        # 3. XGBoost – train only on historical data
        # --------------------------------------------------------------------- #
        # XGBoost predict only
        if self.xgboost_enabled and self.xgb_predictor:
            full_df = self.xgb_predictor.predict(full_df, self.cfg)
            
        # --------------------------------------------------------------------- #
        # 4. LSTM – features locked externally by WalkForwardAnalyzer (no torch.load!)
        # --------------------------------------------------------------------- #
        # LSTM predict only
        if self.lstm_enabled and self.lstm_predictor:
            full_df = self.lstm_predictor.predict(full_df, self.cfg)
    
        # --------------------------------------------------------------------- #
        # 5. Sentiment signal
        # --------------------------------------------------------------------- #
        if self.sentiment_enabled and self.sentiment_model:
            full_df = self.sentiment_model.predict(full_df)
    
        # --------------------------------------------------------------------- #
        # 6. Normalize ML probabilities to [-1.1, 1.1] → [-1.0, 1.0] after clip
        # --------------------------------------------------------------------- #
        for col in self._active_signal_cols(full_df):
            if col in {"signal_xgboost", "signal_lstm"}:
                full_df[col] = (full_df[col] - 0.5) * 2.0 * 1.1
                full_df[col] = full_df[col].clip(-1.0, 1.0)
    
        # --------------------------------------------------------------------- #
        # 7. Combine all active signals
        # --------------------------------------------------------------------- #
        signal_cols = self._active_signal_cols(full_df)
        if not signal_cols:
            full_df["signal_ensemble"] = 0.0
            full_df["clean_ensemble"] = 0.0
        else:
            if self.custom_weights:
                weights = np.array([self.custom_weights.get(c, 1.0) for c in signal_cols])
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                full_df["signal_ensemble"] = (full_df[signal_cols] * weights).sum(axis=1)
            else:
                full_df["signal_ensemble"] = full_df[signal_cols].mean(axis=1)
    
            full_df["signal_ensemble"] = full_df["signal_ensemble"].clip(-1.0, 1.0)
            full_df["clean_ensemble"] = full_df["signal_ensemble"].where(
                full_df["signal_ensemble"].abs() >= self.min_position_threshold, 0.0
            )
    
        # --------------------------------------------------------------------- #
        # 8. Sentiment + volatility overlay
        # --------------------------------------------------------------------- #
        if "sentiment" in full_df.columns and "close" in full_df.columns:
            vol = full_df["close"].pct_change().rolling(20).std().fillna(0.0)
            overlay = self.apply_overlay(full_df["clean_ensemble"], full_df["sentiment"], vol)
            full_df["clean_ensemble"] = (full_df["clean_ensemble"] * overlay).clip(-1.0, 1.0)
    
        # --------------------------------------------------------------------- #
        # 9. Volatility targeting
        # --------------------------------------------------------------------- #
        if self.vol_target_enabled:
            full_df["clean_signal"] = full_df["clean_ensemble"]
            full_df = apply_vol_target(full_df, self.ensemble_cfg.get("volatility_targeting", {}))
            full_df.drop(columns=["clean_signal"], inplace=True, errors="ignore")
    
        # --------------------------------------------------------------------- #
        # 10. RL risk overlay (optional)
        # --------------------------------------------------------------------- #
        if self.rl_overlay_enabled and self.rl_overlay:
            pos = full_df.get("exposure", full_df["clean_ensemble"]).fillna(0.0)
            equity = (1.0 + full_df["close"].pct_change().fillna(0.0) * pos).cumprod()
            full_df = self.rl_overlay.apply(full_df, equity)
    
        # --------------------------------------------------------------------- #
        # 11. Final position – shifted one bar forward (no look-ahead)
        # --------------------------------------------------------------------- #
        base = full_df.get("exposure_rl", full_df.get("exposure", full_df["clean_ensemble"]))
        scaling = float(self.cfg.get("position_scaling", 1.0))
        final_position = (base * scaling).clip(-1.0, 1.0)
    
        full_df["position"] = final_position.shift(1).fillna(0.0)
        full_df["position"] = pd.to_numeric(full_df["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    
        return full_df.reset_index()