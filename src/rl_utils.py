# src/rl_utils.py
"""
Utility classes for reinforcement learning in the walk-forward pipeline.

* ``CustomEvalMonitor`` – a ``Monitor`` that **does not overwrite** ``info["r"]``,
  allowing ``InfoRewardScaler`` to keep the scaled reward while still satisfying
  ``EvalCallback`` (episode length, total reward, time).  
  All monitor attributes are created **inside the wrapper**, never on the base
  environment.

* ``CustomLstmPolicy`` – LSTM feature extractor for PPO (kept for future use).

All docstrings and comments are in English.
"""

import time
from typing import Any

import torch as th
import torch.nn as nn

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# --------------------------------------------------------------------------- #
#                        CustomEvalMonitor
# --------------------------------------------------------------------------- #
class CustomEvalMonitor(Monitor):
    """
    Monitor wrapper **only** for the evaluation environment.

    It behaves exactly like ``stable_baselines3.common.monitor.Monitor`` but
    **never writes** ``info["r"] = reward``. This is required when an
    ``InfoRewardScaler`` (or any reward-modifying wrapper) is placed *after* the
    monitor – otherwise ``EvalCallback`` would see the **unscaled** reward and
    report a near-zero mean reward.

    The wrapper still records:
      * episode length
      * total (scaled) episode reward
      * elapsed real time
    which silences the ``EvalCallback`` warning.
    """

    def __init__(self, env: Any):
        """
        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        """
        # ``Monitor.__init__`` expects the environment and optional filename.
        # We pass only the env – no CSV logging is needed for evaluation.
        super().__init__(env)

        # ----------------------------------------------------------------- #
        # IMPORTANT: ``Monitor`` normally creates these attributes **on the
        # wrapped environment** (self.env).  That caused the
        # ``AttributeError: 'TradingEnv' object has no attribute ...``.
        # We create them **here** inside the wrapper so the base env stays clean.
        # ----------------------------------------------------------------- #
        self.current_episode_reward: float = 0.0
        self.episode_lengths: list[int] = [0]
        self.episode_rewards: list[float] = []
        self.episode_times: list[float] = []
        self.t_start: float = time.time()

    # --------------------------------------------------------------------- #
    def step(self, action: Any):
        """
        Execute one environment step and update episode bookkeeping.

        Parameters
        ----------
        action : Any
            Action supplied by the agent.

        Returns
        -------
        obs, reward, terminated, truncated, info
            Standard Gymnasium step return values.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode stats only if the inner env did not already provide them.
        if "episode" not in info:
            self._update_episode_stats(reward, terminated or truncated)

        # **Never** overwrite ``info["r"]`` – preserves scaled reward.
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    def _update_episode_stats(self, reward: float, done: bool) -> None:
        """
        Update internal episode statistics without touching ``info["r"]``.

        Parameters
        ----------
        reward : float
            Reward returned by the environment (may be scaled).
        done : bool
            ``True`` if the episode has ended (terminated or truncated).
        """
        self.current_episode_reward += reward
        self.episode_lengths[-1] += 1

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(0)
            self.current_episode_reward = 0.0
            self.episode_times.append(time.time() - self.t_start)
            self.t_start = time.time()


# --------------------------------------------------------------------------- #
#                        CustomLstmPolicy (kept for future use)
# --------------------------------------------------------------------------- #
class CustomLstmPolicy(BaseFeaturesExtractor):
    """
    LSTM feature extractor for PPO (or any SB3 algorithm that accepts a
    ``features_extractor_class``).

    Observations must have shape ``(batch, window_size, n_features)``.
    The LSTM processes the temporal dimension and a small linear head
    produces the requested ``features_dim``.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 64,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
    ):
        """
        Parameters
        ----------
        observation_space : gym.spaces.Box
            Observation space of the environment.
        features_dim : int, default 64
            Dimension of the feature vector returned by the extractor.
        lstm_hidden_size : int, default 64
            Hidden size of the LSTM cell.
        n_lstm_layers : int, default 1
            Number of stacked LSTM layers.
        """
        super().__init__(observation_space, features_dim)

        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        n_input_features = observation_space.shape[1]  # features per time-step

        self.lstm = nn.LSTM(
            input_size=n_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the LSTM extractor.

        Parameters
        ----------
        observations : torch.Tensor
            Shape ``(batch, window_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(batch, features_dim)``.
        """
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]

        x = observations.view(batch_size, seq_len, -1)

        device = observations.device
        h0 = th.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = th.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        return self.linear(lstm_out[:, -1, :])