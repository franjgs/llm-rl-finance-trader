# src/models/lstm_model.py
"""
LSTM predictor — fixed prepare/predict mapping and safe load/save.

Key fixes:
- _prepare_data returns sequence start index so predictions align to original df index.
- predict maps predictions back to df using the correct index slice.
- safe torch.load / load_state_dict usage.
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
        out = out[:, -1, :]
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out.clamp(1e-7, 1 - 1e-7)


class LSTMPredictor:
    """
    LSTM pipeline: train / predict with correct index alignment.
    Config keys:
      - model_path, sequence_length, hidden_size, num_layers, dropout,
        epochs, batch_size, learning_rate, prediction_horizon_hours,
        normalize, output_type, threshold
    """
    def __init__(self, config: dict):
        self.model_path = config["model_path"]
        self.retrain = config.get("retrain", False)
        self.sequence_length = int(config["sequence_length"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.dropout = float(config["dropout"])
        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.learning_rate = float(config["learning_rate"])
        self.prediction_horizon = int(config["prediction_horizon_hours"])
        self.normalize = bool(config.get("normalize", True))
        self.output_type = config.get("output_type", "class")
        self.threshold = float(config.get("threshold", 0.5))

        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        logger.info(f"[LSTM] Initialized on {self.device}")

    def _prepare_data(self, df: pd.DataFrame):
        """
        Return:
           loader, feature_cols, seq_start_index (the index label corresponding to first prediction)
        """
        df_proc = generate_features(df.copy())

        # create target aligned to horizon
        future_ret = df_proc["close"].shift(-self.prediction_horizon) / df_proc["close"] - 1.0
        df_proc["target"] = np.where(future_ret > 0, 1.0, np.where(future_ret < 0, 0.0, 1.0))

        # drop rows at tail without target
        df_proc = df_proc.dropna(subset=["target"])
        df_proc = df_proc.dropna()

        feature_cols = [
            c for c in df_proc.columns
            if c not in ["Date", "open", "high", "low", "close", "volume", "target"]
            and "signal" not in c
        ]

        X = df_proc[feature_cols].values.astype(np.float32)
        y = df_proc["target"].values.astype(np.float32)

        if self.normalize and len(X) > 0:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std

        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            if not np.isnan(seq).any():
                X_seq.append(seq)
                y_seq.append(y[i + self.sequence_length])

        if len(X_seq) == 0:
            # empty dataset
            dataset = TensorDataset(torch.empty((0,)), torch.empty((0,)))
            loader = DataLoader(dataset, batch_size=self.batch_size)
            return loader, feature_cols, None

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # seq_start_idx: index label in df_proc that corresponds to the first prediction
        seq_start_pos = self.sequence_length
        seq_start_index = df_proc.index[seq_start_pos] if len(df_proc.index) > seq_start_pos else None

        return loader, feature_cols, seq_start_index

    def train(self, df: pd.DataFrame):
        loader, feature_cols, _ = self._prepare_data(df)
        if len(loader) == 0:
            logger.warning("[LSTM] No training sequences available.")
            return

        self.model = LSTMNetwork(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # init weights
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
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if n_batches > 0 and (epoch + 1) % 5 == 0:
                logger.info(f"   [LSTM] Epoch {epoch+1}/{self.epochs} – loss: {epoch_loss/max(1,n_batches):.6f}")

        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"[LSTM] Model trained and saved → {self.model_path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        loader, feature_cols, seq_start_index = self._prepare_data(df)
        # prepare model structure
        model = LSTMNetwork(
            input_size=max(1, len(feature_cols)),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # load weights (safe)
        try:
            state = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True
            )
            
            model.load_state_dict(state)
        except Exception as e:
            logger.warning(f"[LSTM] Could not load model at {self.model_path}: {e}. Returning zeros.")
            df["signal_lstm"] = 0.0
            return df

        model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                prob = model(X_batch).cpu().numpy().flatten()
                preds.extend(prob)

        if seq_start_index is None:
            # not enough data to create any sequence → all NaN
            df["signal_lstm"] = np.nan
            df["signal_lstm"] = df["signal_lstm"].fillna(0.0)
            return df

        # build a series aligned to df.index: predictions start at seq_start_index
        preds = np.asarray(preds, dtype=float)
        pred_index = df.loc[seq_start_index:].index[: len(preds)]
        proba_series = pd.Series(preds, index=pred_index)

        if self.output_type == "proba":
            df.loc[proba_series.index, "signal_lstm"] = proba_series
        else:
            df.loc[proba_series.index, "signal_lstm"] = (proba_series > self.threshold).astype(int)

        df["signal_lstm"] = df["signal_lstm"].fillna(0.0)
        return df