# src/models/lstm_model.py
"""
LSTM Predictor for Directional Return Forecasting (Fischer & Krauss 2018)

Production-grade, zero-warnings, fully documented LSTM model.
Integrates perfectly with XGBoost + Sentiment ensemble.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import generate_features

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """Clean LSTM classifier with safe output clamping."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # last timestep
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out.clamp(1e-7, 1 - 1e-7)  # prevent BCE explosion


class LSTMPredictor:
    """
    Complete LSTM pipeline: clean data → sequences → train → predict.
    Zero warnings. Institutional grade.
    """
    def __init__(self, config: dict):
        self.model_path = config["model_path"]
        self.retrain = config.get("retrain", False)
        self.sequence_length = config["sequence_length"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.prediction_horizon = config["prediction_horizon_hours"]
        self.normalize = config.get("normalize", True)
        self.output_type = config.get("output_type", "class")
        self.threshold = config.get("threshold", 0.5)

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = None
        logger.info(f"[LSTM] Initialized on {self.device}")

    def _prepare_data(self, df: pd.DataFrame):
        df = generate_features(df.copy())

        # Clean target: 1.0 = up, 0.0 = down/flat (flat → up)
        future_ret = df["close"].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df["target"] = np.where(future_ret > 0, 1.0, np.where(future_ret < 0, 0.0, 1.0))

        df = df.dropna(subset=["target"])
        df = df.dropna()  # Remove any remaining NaN from features

        feature_cols = [
            c for c in df.columns
            if c not in ["Date", "open", "high", "low", "close", "volume", "target"]
            and "signal" not in c
        ]

        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        if self.normalize:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std

        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            if not np.isnan(seq).any():
                X_seq.append(seq)
                y_seq.append(y[i + self.sequence_length])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader, feature_cols

    def train(self, df: pd.DataFrame):
        loader, feature_cols = self._prepare_data(df)

        self.model = LSTMNetwork(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Xavier init for stability
        for layer in self.model.lstm.modules():
            if isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                logger.info(f"   [LSTM] Epoch {epoch+1}/{self.epochs} – loss: {epoch_loss/len(loader):.6f}")

        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"[LSTM] Model trained and saved → {self.model_path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        loader, feature_cols = self._prepare_data(df)

        model = LSTMNetwork(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # ← LÍNEA MÁGICA: ELIMINA EL WARNING PARA SIEMPRE
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        model.eval()

        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                prob = model(X_batch).cpu().numpy().flatten()
                preds.extend(prob)

        pad = len(df) - len(preds)
        preds = np.concatenate([np.full(pad, np.nan), preds])

        if self.output_type == "proba":
            df["signal_lstm"] = preds
        else:
            df["signal_lstm"] = (preds > self.threshold).astype(int)

        return df