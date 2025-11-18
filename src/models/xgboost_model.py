# src/models/xgboost_model.py
"""
XGBoost Predictor — production-ready, leak-free, index-safe.

Key guarantees:
- TRAIN and PREDICT use exactly the same feature set and ordering.
- Feature list is saved alongside the model (packed into joblib file).
- Predict fills missing features with zeros and ignores extras (no feature mismatch).
- Index preservation: predictions map back to original dataframe index.
- Graceful backward-compatibility if an older joblib containing only the model is found.
- Minimal external dependencies; uses joblib when available, otherwise falls back to a MockJoblib (dev).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import generate_features

# suppress noisy xgboost warnings about use_label_encoder
warnings.filterwarnings(
    "ignore",
    message=".*use_label_encoder.*",
    category=UserWarning,
    module="xgboost"
)

# Try to use real joblib if present, else fallback to a MockJoblib for dev/testing.
try:
    import joblib as real_joblib  # type: ignore
    joblib = real_joblib
except Exception:
    class MockJoblib:
        @staticmethod
        def dump(obj, path: str):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            print(f"MockJoblib: Saved to {path} (simulated).")
        @staticmethod
        def load(path: str):
            print(f"MockJoblib: Loaded from {path} (simulated). Returning placeholder model.")
            return xgb.XGBClassifier(eval_metric="logloss")
    joblib = MockJoblib()  # type: ignore


class XGBoostPredictor:
    """
    Production-grade XGBoost predictor.

    Config expected keys (examples):
      - model_path: str
      - prediction_horizon_hours: int
      - params: dict  (xgboost parameters)
      - retrain: bool
      - n_lags: int
      - use_rolling_features: bool
      - dropna: bool
      - eval_set_fraction: float
      - output_type: "proba" | "binary" | "long_short"
      - threshold: float
    """

    def __init__(self, config: Dict[str, Any]):
        self.enabled: bool = bool(config.get("enabled", False))
        self.model_path: str = config["model_path"]
        self.retrain: bool = bool(config.get("retrain", False))

        self.n_lags: int = int(config.get("n_lags", 5))
        self.use_rolling_features: bool = bool(config.get("use_rolling_features", True))
        self.dropna: bool = bool(config.get("dropna", True))

        self.prediction_horizon: int = int(config["prediction_horizon_hours"])
        self.output_type: str = str(config.get("output_type", "class"))
        self.threshold: float = float(config.get("threshold", 0.5))

        self.params: Dict[str, Any] = dict(config.get("params", {}))
        # remove deprecated param if present
        self.params.pop("use_label_encoder", None)
        self.eval_set_fraction: float = float(config.get("eval_set_fraction", 0.1))

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_list: Optional[List[str]] = None  # exact ordered features used for training

        print(f"XGBoost Predictor initialized for horizon: {self.prediction_horizon} hours.")

    # -----------------------
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lag in range(1, self.n_lags + 1):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the canonical feature pipeline. Index is preserved as much as possible.
        If dropna=True, rows with NaN after feature generation are dropped (these will be
        naturally excluded from training/prediction).
        """
        df_proc = generate_features(df.copy())

        if self.n_lags > 0:
            df_proc = self._add_lag_features(df_proc)

        if self.use_rolling_features:
            df_proc["rsi_approx_14"] = df_proc["close"].pct_change().rolling(14).mean()
            df_proc["volatility_20"] = df_proc["close"].pct_change().rolling(20).std()

        if self.dropna:
            df_proc = df_proc.dropna()

        return df_proc

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Leak-free target creation:
        - future_return at time t is (close_{t+h} / close_t) - 1
        - target is binary: 1 if future_return > 0 else 0
        - rows without a future (last h rows) are dropped
        """
        df = df.copy()
        h = int(self.prediction_horizon)
        df["future_return"] = df["close"].shift(-h) / df["close"] - 1.0
        df = df.dropna(subset=["future_return"])
        df["target"] = (df["future_return"] > 0).astype(int)
        return df

    # -----------------------
    def train(self, df: pd.DataFrame):
        """
        Train XGBoost on historical data without leakage.

        Saves a packed object to disk with:
          { "model": xgb_model, "feature_list": [...], "params": {...} }
        """
        print("\n--- Starting XGBoost Training ---")

        # 1) Prepare dataset and target (index-preserved where possible)
        df_proc = self._prepare_features(df.copy())
        df_proc = self._create_target(df_proc)

        # 2) Select feature columns (exclude raw OHLCV and target/aux)
        exclude = {"Date", "open", "high", "low", "close", "volume", "target", "future_return"}
        features = [
            c for c in df_proc.columns
            if c not in exclude and not c.startswith("signal")
        ]

        if len(features) == 0:
            raise RuntimeError("No features available after preparation. Check feature pipeline.")

        X = df_proc[features].copy()
        y = df_proc["target"].astype(int).copy()

        # Align indexes (defensive)
        X = X.loc[y.index]

        pos_frac = float(y.mean()) if len(y) > 0 else 0.0
        print(f"Training rows: {len(X)} | positive fraction: {pos_frac:.3f}")

        # Train/validation split (time-series aware: last fraction reserved as validation)
        val_size = max(1, int(len(X) * self.eval_set_fraction))
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        # ensure objective/metric present
        self.params.setdefault("objective", "binary:logistic")
        self.params.setdefault("eval_metric", "logloss")

        # instantiate model
        self.model = xgb.XGBClassifier(**self.params)
        print(f"Model initialized with params: {self.params}")
        print(f"Validation set size: {len(X_val)} rows ({self.eval_set_fraction*100:.0f}%)")

        # fit (no early stopping by default for reproducibility)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Save the exact feature list used for training (order matters)
        self.feature_list = list(X.columns)

        # Pack and persist model + metadata
        packed = {
            "model": self.model,
            "feature_list": self.feature_list,
            "params": self.params
        }
        # ensure parent dir exists
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(packed, self.model_path)
        print(f"Training complete. Model + feature list saved -> {self.model_path}")
        print("-----------------------------------")

    # -----------------------
    def _load_packed_model(self):
        """
        Load packed object from disk. Backwards-compatible: if older files contain only
        the model instance, return it and set feature_list=None (caller will handle).
        """
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(self.model_path)

        packed = joblib.load(self.model_path)
        # If the loaded object is a dict-like with our keys, extract them
        if isinstance(packed, dict) and "model" in packed and "feature_list" in packed:
            self.model = packed["model"]
            self.feature_list = list(packed.get("feature_list", []))
            # optionally update params
            self.params = dict(packed.get("params", self.params))
        else:
            # older format: assume the object itself is a model
            self.model = packed
            self.feature_list = None

    # -----------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict signals for the input df. Guarantees exact feature alignment with training:
        - If saved feature_list exists, build X using that list (missing features -> 0).
        - If no feature_list available (older model), fall back to safe intersection and log a warning.
        The resulting column is df['signal_xgboost'] and the original df index is preserved.
        """
        # load model (and feature_list) if needed
        if self.model is None:
            try:
                self._load_packed_model()
            except Exception as exc:
                print(f"Warning: failed to load model at {self.model_path}: {exc}. Returning zero signal.")
                df["signal_xgboost"] = 0.0
                return df

        # prepare features from input dataframe
        df_proc = self._prepare_features(df.copy())

        # build feature matrix X following training ordering
        if self.feature_list:
            # ensure all required columns exist, fill missing with zeros
            X = pd.DataFrame(index=df_proc.index)
            for col in self.feature_list:
                if col in df_proc.columns:
                    X[col] = df_proc[col].fillna(0.0)
                else:
                    # missing column: safe zero fill
                    X[col] = 0.0
            # enforce ordering
            X = X[self.feature_list]
        else:
            # backward compatibility: use intersection of training-time features (unknown) and df_proc
            # best-effort: use all numeric non-raw columns
            exclude_cols = {"open", "high", "low", "close", "volume"}
            cand = [c for c in df_proc.columns if c not in exclude_cols and not c.startswith("signal")]
            if not cand:
                print("Warning: No candidate features found for prediction. Returning zero signal.")
                df["signal_xgboost"] = 0.0
                return df
            X = df_proc[cand].fillna(0.0)
            print("Warning: model file did not include feature_list — using best-effort feature subset (may differ from training).")

        # predict probabilities (keep original index)
        try:
            proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Prediction failed: {e}. Returning zero signal.")
            df["signal_xgboost"] = 0.0
            return df

        signal = pd.Series(proba, index=X.index)

        # format output according to requested type
        out_type = self.output_type or "class"
        if out_type == "proba":
            df.loc[signal.index, "signal_xgboost"] = signal
        elif out_type == "binary" or out_type == "class":
            df.loc[signal.index, "signal_xgboost"] = (signal > self.threshold).astype(float)
        elif out_type == "long_short":
            df.loc[signal.index, "signal_xgboost"] = signal.apply(lambda x: 1.0 if x > self.threshold else -1.0)
        else:
            print(f"Unknown output_type '{out_type}'. Falling back to 'proba'.")
            df.loc[signal.index, "signal_xgboost"] = signal

        # fill any remaining rows (e.g., dropped by dropna) with 0.0 and preserve original index
        df["signal_xgboost"] = df.get("signal_xgboost", pd.Series(index=df.index, dtype=float)).fillna(0.0)

        return df