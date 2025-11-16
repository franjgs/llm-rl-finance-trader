# src/ensemble/ensemble_model.py
"""
Ensemble 2025 – Unified Model
Date: November 16, 2025

Integrates:
- Momentum (Moskowitz 2012)
- XGBoost (TimeSeriesSplit)
- LSTM (Fischer & Krauss 2018)
- Sentiment Signal
- Volatility Targeting
- RL Risk Overlay (Deng 2017)

Supports:
- Configurable weights (equal, custom, or learned)
- Optional stacking mode
- Equal risk parity by default
- Minimum position threshold
- Full compatibility with config_ensemble.yaml
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Import all models with correct current names
from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target
from src.models.xgboost_model import XGBoostPredictor  
from src.models.lstm_model import LSTMPredictor
from src.models.sentiment_signal import SentimentSignal
from src.models.rl_risk_overlay import RLRiskOverlay


class EnsembleModel:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ensemble_cfg = config["ensemble"]

        # Initialize models
        self.momentum_enabled = self.ensemble_cfg["momentum"]["enabled"]
        self.xgboost_enabled = self.ensemble_cfg["xgboost"]["enabled"]
        self.lstm_enabled = self.ensemble_cfg["lstm"]["enabled"]
        self.sentiment_enabled = self.ensemble_cfg["sentiment_signal"]["enabled"]
        self.vol_target_enabled = self.ensemble_cfg["volatility_targeting"]["enabled"]
        self.rl_overlay_enabled = self.ensemble_cfg["rl_risk_overlay"]["enabled"]

        # Predictors
        self.xgb_predictor = XGBoostPredictor(self.ensemble_cfg["xgboost"]) if self.xgboost_enabled else None
        self.lstm_predictor = LSTMPredictor(self.ensemble_cfg["lstm"]) if self.lstm_enabled else None
        self.sentiment_model = SentimentSignal(self.ensemble_cfg["sentiment_signal"]) if self.sentiment_enabled else None
        self.rl_overlay = RLRiskOverlay(self.ensemble_cfg["rl_risk_overlay"]) if self.rl_overlay_enabled else None

        # Weighting
        self.equal_risk = self.ensemble_cfg["weighting"].get("equal_risk", True)
        self.custom_weights = self.ensemble_cfg["weighting"].get("custom_weights", None)
        self.min_position_threshold = self.ensemble_cfg["weighting"].get("min_position_threshold", 0.05)

    def _get_active_signals(self, df: pd.DataFrame) -> List[str]:
        """Detect which signal columns are present and valid."""
        possible = []
        if "clean_signal" in df.columns and self.momentum_enabled:
            possible.append("clean_signal")
        if "signal_xgboost" in df.columns and self.xgboost_enabled:
            possible.append("signal_xgboost")
        if "signal_lstm" in df.columns and self.lstm_enabled:
            possible.append("signal_lstm")
        if "signal_sentiment" in df.columns and self.sentiment_enabled:
            possible.append("signal_sentiment")
        return [col for col in possible if not df[col].isna().all()]

    def fit_predict(self, df: pd.DataFrame, equity_curve: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Main method: builds full ensemble signal from raw data.
        """
        df = df.copy()
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = df.sort_index()

        # 1. Momentum
        if self.momentum_enabled:
            df = generate_momentum_signal(df, self.ensemble_cfg["momentum"])

        # 2. XGBoost
        if self.xgboost_enabled and self.xgb_predictor:
            if self.ensemble_cfg["xgboost"]["retrain"]:
                self.xgb_predictor.train(df)
            df = self.xgb_predictor.predict(df)

        # 3. LSTM
        if self.lstm_enabled and self.lstm_predictor:
            if self.ensemble_cfg["lstm"]["retrain"]:
                self.lstm_predictor.train(df)
            df = self.lstm_predictor.predict(df)

        # 4. Sentiment
        if self.sentiment_enabled and self.sentiment_model:
            df = self.sentiment_model.apply(df)

        # 5. Combine signals
        signal_cols = self._get_active_signals(df)
        if not signal_cols:
            raise ValueError("No valid signals generated!")

        # Apply weights
        if self.custom_weights:
            weights = np.array([self.custom_weights.get(col, 1.0) for col in signal_cols])
            weights = weights / weights.sum()
            df["raw_ensemble"] = sum(df[col] * w for col, w in zip(signal_cols, weights))
        else:
            df["raw_ensemble"] = df[signal_cols].mean(axis=1)

        # Clip and clean
        df["raw_ensemble"] = df["raw_ensemble"].clip(-1, 1)
        df["clean_signal"] = df["raw_ensemble"].where(
            df["raw_ensemble"].abs() >= self.min_position_threshold, 0.0
        )

        # 6. Volatility Targeting
        if self.vol_target_enabled:
            df["clean_signal"] = df["clean_signal"]  # ensure column exists
            df = apply_vol_target(df, self.ensemble_cfg["volatility_targeting"])

        # 7. RL Risk Overlay
        if self.rl_overlay_enabled and self.rl_overlay and equity_curve is not None:
            df = self.rl_overlay.apply(df, equity_curve)

        # Final position
        df["position"] = df.get("exposure_rl", df.get("exposure", df["clean_signal"])).shift(1).fillna(0)

        return df.reset_index()