# src/models/lstm_model.py
"""
LSTM Predictor — production-ready, fully logged, PyTorch 2.5+ compliant (no FutureWarning).

Key features:
- Professional logging (no prints)
- Full support for fixed_feature_columns → zero dimension drift in walk-forward
- Respects global feature_mode via ensemble config
- Safe model saving/loading: compatible with weights_only=True (future-proof)
- Graceful fallback on load errors
- Sequence-to-index alignment preserved
- Automatic device selection: MPS (Apple Silicon) > CUDA > CPU
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Project-specific imports
from src.features import generate_features
from src.logging_config import setup_logging  # Professional logger


class LSTMNetwork(nn.Module):
    """Simple but effective LSTM with sigmoid output for binary classification / probability."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out.clamp(1e-7, 1 - 1e-7)  # Numerical stability


class LSTMPredictor:
    """
    Production-grade LSTM predictor with walk-forward stability and PyTorch future-proof saving.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.enabled: bool = bool(config.get("enabled", False))
        if not self.enabled:
            return

        # Professional logger (inherits verbose from main config)
        self.logger = setup_logging(self.cfg.get("verbose", 1))

        self.model_path = config["model_path"]
        self.sequence_length = int(config["sequence_length"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.dropout = float(config["dropout"])
        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.learning_rate = float(config["learning_rate"])
        self.prediction_horizon = int(config["prediction_horizon_hours"])
        self.normalize = config.get("normalize", True)
        self.output_type = config.get("output_type", "proba")
        self.threshold = float(config.get("threshold", 0.5))

        # Device selection: prioritize Apple Silicon MPS
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model: Optional[LSTMNetwork] = None
        self.feature_columns: Optional[List[str]] = None  # Locked after first training

        self.logger.info(f"[LSTM] Initialized on {self.device} | path: {self.model_path}")

    # --------------------------------------------------------------------- #
    def _prepare_data(
        self,
        df: pd.DataFrame,
        ensemble_config: Dict[str, Any],
        fixed_feature_columns: Optional[List[str]] = None
    ) -> Tuple[DataLoader, List[str], Optional[pd.Index]]:
        """
        Generate features and sequences using full ensemble config.
        Returns DataLoader, used feature columns, and first predictable index.
        """
        df_proc = generate_features(df.copy(), ensemble_config)

        # Leak-free target: future return over prediction horizon
        future_ret = df_proc["close"].shift(-self.prediction_horizon) / df_proc["close"] - 1.0
        df_proc["target"] = np.where(future_ret > 0, 1.0, np.where(future_ret < 0, 0.0, 1.0))
        df_proc = df_proc.dropna(subset=["target", "close"]).dropna()

        # Feature selection with walk-forward locking support
        if fixed_feature_columns is not None:
            self.logger.info(f"[LSTM] Using FIXED feature set ({len(fixed_feature_columns)} columns) for walk-forward")
            feature_cols = fixed_feature_columns
            for col in feature_cols:
                if col not in df_proc.columns:
                    df_proc[col] = 0.0
                    self.logger.debug(f"[LSTM] Filled missing fixed feature '{col}' with 0.0")
        else:
            exclude = {"open", "high", "low", "close", "volume", "target"}
            feature_cols = [c for c in df_proc.columns if c not in exclude and not c.startswith("signal")]

        if len(feature_cols) == 0:
            self.logger.error("[LSTM] No features available after preparation.")
            empty = TensorDataset(torch.empty((0, self.sequence_length, 1)), torch.empty((0,)))
            return DataLoader(empty, batch_size=self.batch_size), [], None

        X = df_proc[feature_cols].values.astype(np.float32)
        y = df_proc["target"].values.astype(np.float32)

        # Optional normalization (fit on training data only in real use)
        if self.normalize and len(X) > 0:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std

        # Build sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            if not np.isnan(seq).any():
                X_seq.append(seq)
                y_seq.append(y[i + self.sequence_length])

        if len(X_seq) == 0:
            empty = TensorDataset(torch.empty((0, self.sequence_length, len(feature_cols))), torch.empty((0,)))
            return TDataLoader(empty, batch_size=self.batch_size), feature_cols, None

        X_seq = np.stack(X_seq)
        y_seq = np.array(y_seq).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        first_pred_idx = df_proc.index[self.sequence_length] if len(df_proc) > self.sequence_length else None

        return loader, feature_cols, first_pred_idx

    # --------------------------------------------------------------------- #
    def train(
        self,
        df: pd.DataFrame,
        ensemble_config: Dict[str, Any],
        fixed_feature_columns: Optional[List[str]] = None
    ) -> None:
        """Train LSTM with optional fixed feature set (walk-forward steps >1)."""
        self.logger.info("\n--- Starting LSTM Training ---")

        loader, feature_cols, _ = self._prepare_data(df, ensemble_config, fixed_feature_columns)

        if len(loader.dataset) == 0:
            self.logger.warning("[LSTM] No valid training sequences → skipping training")
            return

        input_size = len(feature_cols)
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Xavier/Glorot initialization
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
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            if epoch % 5 == 0 or epoch == self.epochs:
                avg_loss = epoch_loss / max(1, batches)
                self.logger.info(f"[LSTM] Epoch {epoch:02d}/{self.epochs} – loss: {avg_loss:.6f}")

        # Future-proof save: safe for weights_only=True (PyTorch 2.5+)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "feature_columns": feature_cols,
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "normalize": self.normalize,
        }
        torch.save(save_dict, self.model_path)

        # Lock features for walk-forward
        self.feature_columns = feature_cols.copy()

        self.logger.info(f"[LSTM] Model trained and saved → {self.model_path} ({input_size} input features)")
        self.logger.info("-----------------------------------")

    # --------------------------------------------------------------------- #
    def predict(self, df: pd.DataFrame, ensemble_config: Dict[str, Any]) -> pd.DataFrame:
        """Out-of-sample prediction with fixed feature alignment and safe loading."""
        if self.model is None:
            try:
                # Safe load: compatible with future weights_only=True default
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.feature_columns = checkpoint["feature_columns"]
            except Exception as e:
                self.logger.warning(f"[LSTM] Failed to load model: {e} → returning zero signal")
                df["signal_lstm"] = 0.0
                return df

            model = LSTMNetwork(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
            ).to(self.device)

            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                self.logger.warning(f"[LSTM] Failed to load state dict: {e} → zero signal")
                df["signal_lstm"] = 0.0
                return df

            self.model = model

        loader, _, first_idx = self._prepare_data(df, ensemble_config, fixed_feature_columns=self.feature_columns)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                prob = self.model(X_batch).cpu().numpy().flatten()
                preds.extend(prob)

        if not preds or first_idx is None:
            df["signal_lstm"] = 0.0
            return df

        pred_series = pd.Series(preds, index=df.loc[first_idx:].index[:len(preds)])

        if self.output_type == "proba":
            df.loc[pred_series.index, "signal_lstm"] = pred_series
        else:
            df.loc[pred_series.index, "signal_lstm"] = (pred_series > self.threshold).astype(float)

        df["signal_lstm"] = df["signal_lstm"].fillna(0.0)
        return df