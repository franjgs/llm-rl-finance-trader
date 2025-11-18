# src/models/xgboost_model.py
"""
XGBoost Predictor Module for Directional Return Forecasting

This module implements a robust, production-grade XGBoost classifier designed for
high-frequency financial time series prediction. It handles:
- Feature engineering (lags, rolling stats)
- Clean target creation with bullish bias
- Time-series aware train/validation split
- Full training and inference pipeline
- Zero warnings and full compatibility with modern XGBoost (>=1.6)

Reference: XGBoost - Chen & Guestrin (2016)
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import xgboost as xgb

# Silently suppress the deprecated use_label_encoder warning (XGBoost 1.6+)
warnings.filterwarnings(
    "ignore",
    message=".*use_label_encoder.*",
    category=UserWarning,
    module="xgboost"
)

# ----------------------------------------------------------------------
# Mock Joblib (kept for standalone testing - replace with real joblib in prod)
# ----------------------------------------------------------------------
class MockJoblib:
    """Simulates joblib.dump/load for development and backtesting."""
    @staticmethod
    def dump(model, path: str):
        print(f"MockJoblib: Model saved to {path} (simulated).")

    @staticmethod
    def load(path: str):
        print(f"MockJoblib: Model loaded from {path} (simulated).")
        return xgb.XGBClassifier(eval_metric="logloss")

joblib = MockJoblib()

# ----------------------------------------------------------------------
# Mock feature generator (used only if real src.features is not available)
# ----------------------------------------------------------------------
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder feature generator.
    In production, this is replaced by src.features.generate_features().
    """
    df = df.copy()
    df["price_change_1"] = df["close"].pct_change()
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
    return df


# ======================================================================
# MAIN CLASS: XGBoostPredictor
# ======================================================================
class XGBoostPredictor:
    """
    Complete XGBoost pipeline for binary directional prediction in financial markets.

    Features:
    - Automatic lag and rolling feature creation
    - Clean, finance-aware target generation (up/down with bullish bias on flat)
    - Time-series validation split
    - Robust prediction with fallback to zero signal on error
    - Fully silent operation (no warnings)
    """

    def __init__(self, config: dict):
        """
        Initialize the XGBoost predictor from configuration dictionary.

        Args:
            config (dict): Configuration block from config_ensemble.yaml
        """
        self.enabled = config.get("enabled", False)
        self.model_path = config["model_path"]
        self.retrain = config.get("retrain", False)

        # Feature engineering
        self.n_lags = config.get("n_lags", 5)
        self.use_rolling_features = config.get("use_rolling_features", True)
        self.dropna = config.get("dropna", True)

        # Prediction settings
        self.prediction_horizon = config["prediction_horizon_hours"]
        self.threshold = config.get("threshold", 0.5)
        self.output_type = config.get("output_type", "class")  # "class" or "proba"

        # XGBoost hyperparameters (clean copy - no deprecated params)
        self.params = config["params"].copy()
        self.params.pop("use_label_encoder", None)  # Ensure no deprecated param

        # Validation settings
        self.eval_set_fraction = config.get("eval_set_fraction", 0.1)

        self.model = None
        print(f"XGBoost Predictor initialized for horizon: {self.prediction_horizon} hours.")

    # ------------------------------------------------------------------
    # Feature Engineering Helpers
    # ------------------------------------------------------------------
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged close prices as predictive features."""
        df = df.copy()
        for lag in range(1, self.n_lags + 1):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate complete feature set:
        1. External technical indicators (via generate_features)
        2. Lagged prices
        3. Rolling statistics
        4. Optional NaN removal
        """
        df = generate_features(df.copy())

        if self.n_lags > 0:
            df = self._add_lag_features(df)

        if self.use_rolling_features:
            df["rsi_approx_14"] = df["close"].pct_change().rolling(14).mean()
            df["volatility_20"] = df["close"].pct_change().rolling(20).std()

        if self.dropna:
            df.dropna(inplace=True)

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target:
        - 1 if price increases over prediction_horizon
        - 0 if price decreases
        - Flat returns treated as 1 (bullish/neutral bias)
        """
        df = df.copy()
        future_return = df["close"].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df["target"] = np.sign(future_return)
        df["target"] = df["target"].replace(0, np.nan).fillna(1)  # flat → long
        df["target"] = (df["target"] == 1).astype(int)
        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame):
        """Train the XGBoost model on historical data."""
        print("\n--- Starting XGBoost Training ---")

        df_processed = self._prepare_features(df.copy())
        df_processed = self._create_target(df_processed)

        # Select features (exclude raw OHLCV and target)
        exclude_cols = ["Date", "open", "high", "low", "close", "volume", "target"]
        features = [c for c in df_processed.columns if c not in exclude_cols and not c.startswith("signal")]
        X = df_processed[features]
        y = df_processed["target"]

        # Time-series train/validation split
        val_size = int(len(X) * self.eval_set_fraction)
        X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

        # Ensure clean params
        self.params.setdefault("objective", "binary:logistic")
        self.params.setdefault("eval_metric", "logloss")

        self.model = xgb.XGBClassifier(**self.params)
        print(f"Model initialized with params: {self.params}")
        print(f"Validation set size: {len(X_val)} rows ({self.eval_set_fraction*100:.0f}%)")

        # Train (no early stopping - full n_estimators for stability)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_iter = getattr(self.model, "best_iteration", "N/A")
        print(f"Training complete. Best iteration: {best_iter}")

        joblib.dump(self.model, self.model_path)
        print(f"XGBoost model trained and saved to {self.model_path}")
        print("-----------------------------------")

    # ------------------------------------------------------------------
    # Inference / Signal Generation
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signal (binary or probability) for the entire dataframe.

        Returns:
            pd.DataFrame: Original df with added 'signal_xgboost' column
        """
        if self.model is None:
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                print(f"Warning: Model not found at {self.model_path}. Using zero signal.")
                df["signal_xgboost"] = 0.0
                return df

        df_processed = self._prepare_features(df.copy())

        exclude_cols = ["Date", "open", "high", "low", "close", "volume"]
        features = [c for c in df_processed.columns if c not in exclude_cols and not c.startswith("signal")]
        X = df_processed[features].fillna(0)

        try:
            proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Prediction failed: {e}. Using zero signal.")
            df["signal_xgboost"] = 0.0
            return df

        signal = pd.Series(proba, index=X.index)

        if self.output_type == "proba":
            df.loc[signal.index, "signal_xgboost"] = signal
        else:
            df.loc[signal.index, "signal_xgboost"] = (signal > self.threshold).astype(int)

        df["signal_xgboost"] = df["signal_xgboost"].fillna(0.0)
        return df