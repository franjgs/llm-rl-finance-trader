# src/models/xgboost_model.py
"""
XGBoost classifier for financial return prediction.
Full version compatible with config_ensemble.yaml.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from src.features import generate_features


class XGBoostPredictor:
    def __init__(self, config):
        """Initialize XGBoost model using configuration dict."""
        self.enabled = config["enabled"]
        self.model_path = config["model_path"]
        self.retrain = config["retrain"]

        # parameters for feature engineering
        self.n_lags = config.get("n_lags", 5)
        self.use_rolling_features = config.get("use_rolling_features", True)
        self.dropna = config.get("dropna", True)

        # target horizon
        self.prediction_horizon = config["prediction_horizon_hours"]

        # CV params
        self.cv_splits = config.get("cv_splits", 3)
        self.test_size_ratio = config.get("test_size_ratio", 0.2)

        # XGBoost hyperparameters
        self.params = config["params"]

        # early stopping
        self.early_stopping_rounds = config.get("early_stopping_rounds", 30)
        self.eval_set_fraction = config.get("eval_set_fraction", 0.1)

        # output settings
        self.threshold = config.get("threshold", 0.5)
        self.output_type = config.get("output_type", "class")  # "class" or "proba"

        self.model = None

    # -----------------------------------------------------
    # Helper methods
    # -----------------------------------------------------

    def _add_lag_features(self, df):
        """Add lagged versions of all numeric columns."""
        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df["close"].shift(lag)
        return df

    def _prepare_features(self, df):
        """Generate all final features."""
        df = generate_features(df)

        if self.n_lags > 0:
            df = self._add_lag_features(df)

        if self.use_rolling_features:
            df["rsi_14"] = df["close"].pct_change().rolling(14).mean()
            df["volatility_20"] = df["close"].pct_change().rolling(20).std()

        if self.dropna:
            df = df.dropna()

        return df

    def _create_target(self, df):
        """Binary target: sign of return over prediction horizon."""
        df["target"] = np.sign(df["close"].pct_change(self.prediction_horizon).shift(-self.prediction_horizon))
        df["target"] = df["target"].replace(0, np.nan).fillna(1)  # avoid zeros
        return df

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    def train(self, df):
        """Train XGBoost model with time-series cross-validation."""
        df = self._prepare_features(df)
        df = self._create_target(df)

        # Features
        unwanted = ["Date", "close", "target"]
        features = [c for c in df.columns if c not in unwanted and "signal" not in c]

        X = df[features]
        y = df["target"]

        # time-series split
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        # Use last part of data for validation in early stopping
        valid_size = int(len(df) * self.eval_set_fraction)
        X_train, X_val = X.iloc[:-valid_size], X.iloc[-valid_size:]
        y_train, y_val = y.iloc[:-valid_size], y.iloc[-valid_size:]

        self.model = xgb.XGBClassifier(**self.params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )

        joblib.dump(self.model, self.model_path)

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    def predict(self, df):
        """Predict class or probability depending on config."""
        if self.model is None:
            self.model = joblib.load(self.model_path)

        df = self._prepare_features(df)

        features = [c for c in df.columns if c not in ["Date", "close"] and "signal" not in c]
        X = df[features].fillna(0)

        if self.output_type == "proba":
            df["signal_xgboost"] = self.model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict_proba(X)[:, 1]
            df["signal_xgboost"] = (proba > self.threshold).astype(int)

        return df