# src/models/lstm_model.py
"""
LSTM predictor for financial return direction.
Reference: Fischer & Krauss (2018).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import generate_features


# ---------------------------------------------------------
# LSTM Network
# ---------------------------------------------------------

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


# ---------------------------------------------------------
# Predictor class
# ---------------------------------------------------------

class LSTMPredictor:
    def __init__(self, config):
        self.model_path = config["model_path"]
        self.retrain = config["retrain"]

        self.sequence_length = config["sequence_length"]
        self.prediction_horizon = config["prediction_horizon_hours"]

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.normalize = config["normalize"]

        self.output_type = config["output_type"]
        self.threshold = config["threshold"]

        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Feature engineering + sequences
    # ---------------------------------------------------------

    def _prepare_data(self, df):
        df = generate_features(df)

        # target label
        df["target"] = np.sign(df["close"].pct_change(self.prediction_horizon).shift(-self.prediction_horizon))
        df["target"] = df["target"].replace(0, np.nan).fillna(1)

        df = df.dropna()

        feature_cols = [
            c for c in df.columns
            if c not in ["Date", "close", "target"] and "signal" not in c
        ]

        X = df[feature_cols].values
        y = df["target"].values

        # normalization
        if self.normalize:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            X = (X - mean) / std

        # sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])

        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return loader, feature_cols

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    def train(self, df):
        loader, self.feature_cols = self._prepare_data(df)

        self.model = LSTMNetwork(
            input_size=len(self.feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), self.model_path)

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(self, df):
        # load model
        loader, feature_cols = self._prepare_data(df)

        model = LSTMNetwork(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        all_preds = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                proba = model(X_batch).cpu().numpy()
                all_preds.extend(proba.flatten())

        # align output
        pad = len(df) - len(all_preds)
        preds = np.concatenate([np.full(pad, np.nan), np.array(all_preds)])

        if self.output_type == "proba":
            df["signal_lstm"] = preds
        else:
            df["signal_lstm"] = (preds > self.threshold).astype(int)

        return df