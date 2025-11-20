# src/models/xgboost_model.py
"""
XGBoost Predictor — production-ready, leak-free, fully logged.

Key features:
- Uses project-wide logger (no print statements)
- Full support for fixed_feature_columns → perfect walk-forward stability
- Unified attribute name: self.feature_columns (same as LSTM)
- Saves feature_columns with model for exact reproducibility
- Zero-fill missing features, drops extras → no dimension mismatch
- Backward compatible with old models (accepts legacy "feature_list")
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

# Project-specific imports
from src.features import generate_features
from src.logging_config import setup_logging  # ← Professional logger

# Suppress XGBoost warnings
warnings.filterwarnings(
    "ignore",
    message=".*use_label_encoder.*",
    category=UserWarning,
    module="xgboost"
)

# Joblib with fallback (dev/testing)
try:
    import joblib as real_joblib
    joblib = real_joblib
except Exception:
    class MockJoblib:
        @staticmethod
        def dump(obj, path: str):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        @staticmethod
        def load(path: str):
            return xgb.XGBClassifier(eval_metric="logloss")
    joblib = MockJoblib()


class XGBoostPredictor:
    """
    Production-grade XGBoost predictor with professional logging and walk-forward support.
    Uses the same public attribute name as LSTMPredictor: feature_columns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.enabled: bool = bool(config.get("enabled", False))
        if not self.enabled:
            return

        # Professional logger (inherits verbose level from main config)
        self.logger = setup_logging(self.cfg.get("verbose", 1))

        self.model_path: str = config["model_path"]
        self.retrain: bool = bool(config.get("retrain", False))
        self.n_lags: int = int(config.get("n_lags", 5))
        self.use_rolling_features: bool = bool(config.get("use_rolling_features", True))
        self.dropna: bool = bool(config.get("dropna", True))
        self.prediction_horizon: int = int(config["prediction_horizon_hours"])
        self.output_type: str = str(config.get("output_type", "class"))
        self.threshold: float = float(config.get("threshold", 0.5))
        self.params: Dict[str, Any] = dict(config.get("params", {}))
        self.params.pop("use_label_encoder", None)  # Deprecated
        self.eval_set_fraction: float = float(config.get("eval_set_fraction", 0.1))

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: Optional[List[str]] = None  # ← Unified name (same as LSTM)

        self.logger.info(
            f"XGBoost Predictor initialized | horizon: {self.prediction_horizon}h | path: {self.model_path}"
        )

    # --------------------------------------------------------------------- #
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lag in range(1, self.n_lags + 1):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

    def _prepare_features(self, df: pd.DataFrame, ensemble_config: Dict[str, Any]) -> pd.DataFrame:
        df_proc = generate_features(df.copy(), ensemble_config)
        if self.n_lags > 0:
            df_proc = self._add_lag_features(df_proc)
        if self.use_rolling_features:
            df_proc["rsi_approx_14"] = df_proc["close"].pct_change().rolling(14).mean()
            df_proc["volatility_20"] = df_proc["close"].pct_change().rolling(20).std()
        if self.dropna:
            df_proc = df_proc.dropna()
        return df_proc

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        h = self.prediction_horizon
        df["future_return"] = df["close"].shift(-h) / df["close"] - 1.0
        df = df.dropna(subset=["future_return"])
        df["target"] = (df["future_return"] > 0).astype(int)
        return df

    # --------------------------------------------------------------------- #
    def train(
        self,
        df: pd.DataFrame,
        ensemble_config: Dict[str, Any],
        fixed_feature_columns: Optional[List[str]] = None
    ):
        """
        Train XGBoost model.
        If fixed_feature_columns is provided (walk-forward steps >1), forces exact column match.
        """
        self.logger.info("\n--- Starting XGBoost Training ---")

        df_proc = self._prepare_features(df.copy(), ensemble_config)
        df_proc = self._create_target(df_proc)

        exclude = {"Date", "open", "high", "low", "close", "volume", "target", "future_return"}
        candidate_features = [c for c in df_proc.columns if c not in exclude and not c.startswith("signal")]

        if fixed_feature_columns is not None:
            self.logger.info(f"[XGBoost] Using FIXED feature set ({len(fixed_feature_columns)} columns) for walk-forward")
            features = fixed_feature_columns
            for col in features:
                if col not in df_proc.columns:
                    df_proc[col] = 0.0
                    self.logger.debug(f"Filling missing fixed feature '{col}' with 0.0")
        else:
            features = candidate_features
            if not features:
                raise RuntimeError("No features generated. Check feature pipeline.")

        X = df_proc[features].copy()
        y = df_proc["target"].astype(int)
        X = X.loc[y.index]

        pos_frac = y.mean() if len(y) > 0 else 0.0
        self.logger.info(f"Training rows: {len(X)} | positive fraction: {pos_frac:.3f} | features: {len(features)}")

        val_size = max(1, int(len(X) * self.eval_set_fraction))
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        self.params.setdefault("objective", "binary:logistic")
        self.params.setdefault("eval_metric", "logloss")

        self.model = xgb.XGBClassifier(**self.params)
        self.logger.info(f"Model initialized → {self.params}")
        self.logger.info(f"Validation set: {len(X_val)} rows ({self.eval_set_fraction*100:.0f}%)")

        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Unified attribute (same as LSTM)
        self.feature_columns = features.copy()

        # Backward compatibility: save as both keys
        packed = {
            "model": self.model,
            "feature_list": self.feature_columns,      # legacy key
            "feature_columns": self.feature_columns,   # new unified key
            "params": self.params
        }
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(packed, self.model_path)

        self.logger.info(f"Training complete → {self.model_path}")
        self.logger.info("-----------------------------------")

    # --------------------------------------------------------------------- #
    def _load_packed_model(self):
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(self.model_path)

        packed = joblib.load(self.model_path)

        if isinstance(packed, dict) and "model" in packed:
            self.model = packed["model"]
            # Prefer new key, fallback to legacy
            self.feature_columns = list(packed.get("feature_columns") or packed.get("feature_list", []))
            self.params = dict(packed.get("params", self.params))
        else:
            # Very old format
            self.model = packed
            self.feature_columns = None
            self.logger.warning("Legacy model loaded (no feature_columns saved)")

    # --------------------------------------------------------------------- #
    def predict(self, df: pd.DataFrame, ensemble_config: Dict[str, Any]) -> pd.DataFrame:
        if not self.enabled or self.model is None:
            try:
                self._load_packed_model()
            except Exception as e:
                self.logger.warning(f"Failed to load XGBoost model: {e} → returning zero signal")
                df["signal_xgboost"] = 0.0
                return df

        df_proc = self._prepare_features(df.copy(), ensemble_config)

        if self.feature_columns:
            self.logger.debug(f"[XGBoost] Predicting with {len(self.feature_columns)} locked features")
            X = pd.DataFrame(index=df_proc.index)
            for col in self.feature_columns:
                X[col] = df_proc.get(col, 0.0).fillna(0.0)
            X = X[self.feature_columns]
        else:
            exclude = {"open", "high", "low", "close", "volume"}
            cand = [c for c in df_proc.columns if c not in exclude and not c.startswith("signal")]
            if not cand:
                self.logger.warning("No candidate features for prediction → zero signal")
                df["signal_xgboost"] = 0.0
                return df
            X = df_proc[cand].fillna(0.0)
            self.logger.warning("Using best-effort features (no saved feature_columns)")

        try:
            proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            df["signal_xgboost"] = 0.0
            return df

        signal = pd.Series(proba, index=X.index)

        if self.output_type == "proba":
            df.loc[signal.index, "signal_xgboost"] = signal
        elif self.output_type in ("binary", "class"):
            df.loc[signal.index, "signal_xgboost"] = (signal > self.threshold).astype(float)
        elif self.output_type == "long_short":
            df.loc[signal.index, "signal_xgboost"] = np.where(signal > self.threshold, 1.0, -1.0)
        else:
            df.loc[signal.index, "signal_xgboost"] = signal

        df["signal_xgboost"] = df.get("signal_xgboost", pd.Series(index=df.index)).fillna(0.0)
        return df