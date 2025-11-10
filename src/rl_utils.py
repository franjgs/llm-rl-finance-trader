# src/rl_utils.py
"""
Custom LSTM policy for Stable-Baselines3 PPO.
"""
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn


class CustomLstmPolicy(BaseFeaturesExtractor):
    """
    Custom LSTM feature extractor for PPO.
    """
    def __init__(self, observation_space, features_dim=64, lstm_hidden_size=64, n_lstm_layers=1):
        super().__init__(observation_space, features_dim)
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        n_input_features = observation_space.shape[1]
        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True
        )
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        x = observations.view(batch_size, seq_len, -1)

        h0 = th.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(observations.device)
        c0 = th.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size).to(observations.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.linear(lstm_out[:, -1, :])