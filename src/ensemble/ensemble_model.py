"""
Ensemble 2025 – Unified Model (cleaned + safety fixes)
Notes:
- Output final ensemble signal as `signal_ensemble`.
- Momentum produces `signal_momentum`.
- Volatility targeting expects `clean_ensemble` input and writes `exposure`.
- RL overlay expects `exposure` and writes `exposure_rl`.
- Retrain protection: when retrain=True we train on historical portion only (no future leakage).
- Safety: intermediate exposure is capped by ensemble.max_exposure_abs to avoid runaway leverage.
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
    """
    Main ensemble orchestrator that combines multiple trading signals into a final position.
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ensemble_cfg = config.get("ensemble", {})
        # Enabled flags
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
        # Weighting / aggregation
        self.equal_risk = bool(self.ensemble_cfg.get("weighting", {}).get("equal_risk", True))
        self.custom_weights = self.ensemble_cfg.get("weighting", {}).get("custom_weights", None)
        self.min_position_threshold = float(self.ensemble_cfg.get("weighting", {}).get("min_position_threshold", 0.0))
        # Safety: max absolute exposure before final clipping (can be overridden in config)
        self.max_exposure_abs = float(self.ensemble_cfg.get("max_exposure_abs", 1.0))

    # -------------------------
    def apply_overlay(self, signal, sentiment, vol):
        """
        Apply a simple overlay layer:
        - Dampens exposure when volatility is high.
        - Adjusts exposure with sentiment extremes.
        - Smooth and bounded output.
        """
        overlay = np.ones(len(signal))

        # Volatility dampening
        vol_th = 0.03
        overlay = np.where(vol > vol_th, overlay * 0.6, overlay)

        # Sentiment reinforcement
        overlay = np.where(sentiment > 0.7, overlay * 1.1, overlay)
        overlay = np.where(sentiment < -0.7, overlay * 0.85, overlay)

        return np.clip(overlay, 0.0, 1.2)

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
        """Train using an initial historical slice (simple 80/20 heuristic)."""
        if df.shape[0] < 50:
            model_train_fn(df.copy())
            return
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].copy()
        model_train_fn(train_df)

    # -------------------------
    def fit_predict(self, df: pd.DataFrame, equity_curve: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Build ensemble signals and return DataFrame with signals, exposures and final position.
        """
        df = df.copy()
        if "Date" in df.columns:
            df = df.set_index("Date")
        df = df.sort_index()

        # 1) Momentum
        if self.momentum_enabled:
            df = generate_momentum_signal(df, self.ensemble_cfg.get("momentum", {}))
            if "signal_momentum" not in df.columns and "clean_signal" in df.columns:
                df["signal_momentum"] = df["clean_signal"]

        # 2) XGBoost
        if self.xgboost_enabled and self.xgb_predictor:
            if self.ensemble_cfg.get("xgboost", {}).get("retrain", False):
                self._safe_train_on_past(self.xgb_predictor.train, df)
            df = self.xgb_predictor.predict(df)

        # 3) LSTM
        if self.lstm_enabled and self.lstm_predictor:
            if self.ensemble_cfg.get("lstm", {}).get("retrain", False):
                self._safe_train_on_past(self.lstm_predictor.train, df)
            df = self.lstm_predictor.predict(df)

        # 4) Sentiment
        if self.sentiment_enabled and self.sentiment_model:
            df = self.sentiment_model.predict(df)

        # Normalize ML outputs to [-1, 1]
        signal_cols = self._active_signal_cols(df)
        for col in signal_cols:
            if col in ["signal_xgboost", "signal_lstm"]:
                df[col] = (df[col] - 0.5) * 2.0 * 1.1
                df[col] = df[col].clip(-1.0, 1.0)

        # 5) Combine signals
        signal_cols = self._active_signal_cols(df)
        if not signal_cols:
            df["signal_ensemble"] = 0.0
            df["clean_ensemble"] = 0.0
        else:
            if self.custom_weights:
                weights = np.array([self.custom_weights.get(col, 1.0) for col in signal_cols], dtype=float)
                if weights.sum() == 0:
                    weights = np.ones_like(weights)
                weights = weights / weights.sum()
                df_signals = df[signal_cols].fillna(0.0)
                df["signal_ensemble"] = df_signals.multiply(weights, axis=1).sum(axis=1)
            else:
                df["signal_ensemble"] = df[signal_cols].fillna(0.0).mean(axis=1)

            df["signal_ensemble"] = df["signal_ensemble"].clip(-1.0, 1.0)

            is_sent_raw = (
                len(signal_cols) == 1
                and "signal_sentiment" in signal_cols
                and self.ensemble_cfg.get("sentiment_signal", {}).get("output_type", "class") == "raw"
            )

            if is_sent_raw:
                df["clean_ensemble"] = df["signal_ensemble"]
            else:
                df["clean_ensemble"] = df["signal_ensemble"].where(
                    df["signal_ensemble"].abs() >= self.min_position_threshold,
                    0.0
                )

        # -----------------------
        # 5.1 Apply Overlay HERE
        # -----------------------
        if "sentiment" in df.columns and "close" in df.columns:
            vol_series = df["close"].pct_change().rolling(20).std()
        
            overlay_vec = self.apply_overlay(
                df["clean_ensemble"],
                df["sentiment"].fillna(0.0),
                vol_series.fillna(0.0),
            )
        
            df["clean_ensemble"] = (df["clean_ensemble"] * overlay_vec).clip(-1, 1)

        # 6) Volatility targeting
        if self.vol_target_enabled:
            df["clean_signal"] = df["clean_ensemble"]
            df = apply_vol_target(df, self.ensemble_cfg.get("volatility_targeting", {}))
            if "clean_signal" in df.columns:
                df.drop(columns=["clean_signal"], inplace=True)

        # Safety cap
        if "exposure" in df.columns:
            df["exposure"] = df["exposure"].clip(-abs(self.max_exposure_abs), abs(self.max_exposure_abs))

        # 7) RL overlay
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

        # 8) Final position
        if "exposure_rl" in df.columns:
            base = df["exposure_rl"]
        elif "exposure" in df.columns:
            base = df["exposure"]
        else:
            base = df["clean_ensemble"]

        scaling = float(self.cfg.get("position_scaling", 1.0))
        base = (base * scaling).clip(-1.0, 1.0)
        df["position"] = base.shift(1).fillna(0.0)
        df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

        return df.reset_index()