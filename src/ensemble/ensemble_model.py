# src/ensemble/ensemble_model.py
"""
EnsembleModel — final integration.

Overlay is applied AFTER combining signals (option B) and BEFORE volatility-targeting.
This keeps base models pure and lets vol-targeting size the overlay-adjusted signal.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.sentiment_signal import SentimentSignal
from src.models.rl_risk_overlay import RLRiskOverlay


class EnsembleModel:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ensemble_cfg = config.get("ensemble", {})

        # flags
        self.momentum_enabled = bool(self.ensemble_cfg.get("momentum", {}).get("enabled", False))
        self.xgboost_enabled = bool(self.ensemble_cfg.get("xgboost", {}).get("enabled", False))
        self.lstm_enabled = bool(self.ensemble_cfg.get("lstm", {}).get("enabled", False))
        self.sentiment_enabled = bool(self.ensemble_cfg.get("sentiment_signal", {}).get("enabled", False))
        self.vol_target_enabled = bool(self.ensemble_cfg.get("volatility_targeting", {}).get("enabled", False))
        self.rl_overlay_enabled = bool(self.ensemble_cfg.get("rl_risk_overlay", {}).get("enabled", False))

        # instantiate
        self.xgb_predictor = XGBoostPredictor(self.ensemble_cfg.get("xgboost", {})) if self.xgboost_enabled else None
        self.lstm_predictor = LSTMPredictor(self.ensemble_cfg.get("lstm", {})) if self.lstm_enabled else None
        self.sentiment_model = SentimentSignal(self.ensemble_cfg.get("sentiment_signal", {})) if self.sentiment_enabled else None
        self.rl_overlay = RLRiskOverlay(self.ensemble_cfg.get("rl_risk_overlay", {})) if self.rl_overlay_enabled else None

        # weighting
        self.custom_weights = self.ensemble_cfg.get("weighting", {}).get("custom_weights", None)
        self.min_position_threshold = float(self.ensemble_cfg.get("weighting", {}).get("min_position_threshold", 0.0))
        self.max_exposure_abs = float(self.ensemble_cfg.get("max_exposure_abs", 1.0))

    # -------------------------
    def apply_overlay(self, signal: pd.Series, sentiment: pd.Series, vol: pd.Series) -> pd.Series:
        """
        Lightweight business-rule overlay:
         - reduce exposure when vol high
         - boost/reduce on sentiment extremes
         - returns a multiplicative factor series aligned to signal.index
        """
        overlay = pd.Series(1.0, index=signal.index)
        vol_th = 0.03
        overlay.loc[vol > vol_th] *= 0.6
        overlay.loc[sentiment > 0.7] *= 1.1
        overlay.loc[sentiment < -0.7] *= 0.85
        return overlay.clip(0.0, 1.2)

    # -------------------------
    def _active_signal_cols(self, df: pd.DataFrame) -> List[str]:
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

    def _safe_train_on_past(self, model_train_fn, df: pd.DataFrame):
        """Train on first 80% of the passed df (prevent accidental future leakage)."""
        if df.shape[0] < 50:
            model_train_fn(df.copy())
            return
        split = int(len(df) * 0.8)
        model_train_fn(df.iloc[:split].copy())

    # -------------------------
    def fit_predict(self, df: pd.DataFrame, equity_curve: Optional[pd.Series] = None) -> pd.DataFrame:
        df = df.copy()
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = df.sort_index()

        # 1) momentum
        if self.momentum_enabled:
            df = generate_momentum_signal(df, self.ensemble_cfg.get("momentum", {}))
            if "signal_momentum" not in df.columns and "clean_signal" in df.columns:
                df["signal_momentum"] = df["clean_signal"]

        # 2) xgboost
        if self.xgboost_enabled and self.xgb_predictor:
            if self.ensemble_cfg.get("xgboost", {}).get("retrain", False):
                self._safe_train_on_past(self.xgb_predictor.train, df)
            df = self.xgb_predictor.predict(df)

        # 3) lstm
        if self.lstm_enabled and self.lstm_predictor:
            if self.ensemble_cfg.get("lstm", {}).get("retrain", False):
                self._safe_train_on_past(self.lstm_predictor.train, df)
            df = self.lstm_predictor.predict(df)

        # 4) sentiment model
        if self.sentiment_enabled and self.sentiment_model:
            df = self.sentiment_model.predict(df)

        # Normalize ML probabilities to [-1,1]
        signal_cols = self._active_signal_cols(df)
        for col in signal_cols:
            if col in {"signal_xgboost", "signal_lstm"}:
                df[col] = (df[col] - 0.5) * 2.0 * 1.1
                df[col] = df[col].clip(-1.0, 1.0)

        # 5) combine signals
        signal_cols = self._active_signal_cols(df)
        if not signal_cols:
            df["signal_ensemble"] = 0.0
            df["clean_ensemble"] = 0.0
        else:
            if self.custom_weights:
                weights = np.array([self.custom_weights.get(col, 1.0) for col in signal_cols], dtype=float)
                weights = weights if weights.sum() != 0 else np.ones_like(weights)
                weights = weights / weights.sum()
                df_signals = df[signal_cols].fillna(0.0)
                df["signal_ensemble"] = df_signals.multiply(weights, axis=1).sum(axis=1)
            else:
                df["signal_ensemble"] = df[signal_cols].fillna(0.0).mean(axis=1)
            df["signal_ensemble"] = df["signal_ensemble"].clip(-1.0, 1.0)

            # clean_ensemble respects min_position_threshold
            is_sent_raw = (
                len(signal_cols) == 1
                and "signal_sentiment" in signal_cols
                and self.ensemble_cfg.get("sentiment_signal", {}).get("output_type", "class") == "raw"
            )
            if is_sent_raw:
                df["clean_ensemble"] = df["signal_ensemble"]
            else:
                df["clean_ensemble"] = df["signal_ensemble"].where(
                    df["signal_ensemble"].abs() >= self.min_position_threshold, 0.0
                )

        # -----------------------
        # APPLY OVERLAY (Option B): AFTER combine, BEFORE vol-target
        # -----------------------
        if "sentiment" in df.columns and "close" in df.columns:
            vol_series = df["close"].pct_change().rolling(20).std().fillna(0.0)
            overlay = self.apply_overlay(df["clean_ensemble"].fillna(0.0), df["sentiment"].fillna(0.0), vol_series)
            df["clean_ensemble"] = (df["clean_ensemble"].fillna(0.0) * overlay).clip(-1.0, 1.0)

        # 6) Volatility targeting (reads clean_ensemble, writes exposure)
        if self.vol_target_enabled:
            df["clean_signal"] = df["clean_ensemble"]
            df = apply_vol_target(df, self.ensemble_cfg.get("volatility_targeting", {}))
            if "clean_signal" in df.columns:
                df.drop(columns=["clean_signal"], inplace=True)

        # Safety cap on intermediate exposure
        if "exposure" in df.columns:
            df["exposure"] = df["exposure"].clip(-abs(self.max_exposure_abs), abs(self.max_exposure_abs))

        # 7) RL overlay (final real-time multiplier)
        if self.rl_overlay_enabled and self.rl_overlay:
            if equity_curve is None:
                tmp_pos = df["exposure"].fillna(0.0) if "exposure" in df.columns else df["clean_ensemble"].fillna(0.0)
                if "close" in df.columns:
                    equity = (1.0 + df["close"].pct_change().fillna(0) * tmp_pos).cumprod()
                else:
                    equity = pd.Series(1.0, index=df.index)
            else:
                equity = equity_curve
            df = self.rl_overlay.apply(df, equity)

        # 8) final position (shifted)
        if "exposure_rl" in df.columns:
            base = df["exposure_rl"]
        elif "exposure" in df.columns:
            base = df["exposure"]
        else:
            base = df["clean_ensemble"].fillna(0.0)

        scaling = float(self.cfg.get("position_scaling", 1.0))
        base = (base * scaling).clip(-1.0, 1.0)
        df["position"] = base.shift(1).fillna(0.0)
        df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

        return df.reset_index()