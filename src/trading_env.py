# src/trading_env.py
"""
Gymnasium-compatible trading environment.

Features
--------
- Discrete actions: 0=hold, 1=buy, 2=sell
- Optional sentiment feature
- Configurable initial balance
- Windowed observations for LSTM (window_size > 1)
- Robust reset/step with index safety
- Compatible with Stable-Baselines3 + LSTM policy
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agents.

    Observation
    -----------
    - window_size == 1 → [open, high, low, close, volume, (sentiment)]
    - window_size > 1 → stacked history of above (shape: (window_size, n_features))

    Action Space
    ------------
    Discrete(3):
        0 → Hold
        1 → Buy (all available cash)
        2 → Sell (all held shares)

    Reward
    ------
    Profit from sell actions only.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        initial_balance: float = 10_000.0,
        window_size: int = 1,
    ) -> None:
        """
        Initialize the trading environment.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ['Date', 'open', 'high', 'low', 'close', 'volume']
            Optional: 'sentiment'
            Index will be ignored; Date used for logging only.
        use_sentiment : bool
            Include sentiment in observation if available.
        initial_balance : float
            Starting cash.
        window_size : int
            Number of past steps to include in observation (for LSTM).
        """
        super().__init__()

        if "Date" not in df.columns:
            raise ValueError("DataFrame must have 'Date' column")
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError("DataFrame missing required price/volume columns")

        # Clean and reset
        self.df = df.dropna().reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after dropping NaN")

        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)

        # State
        self.balance: float = self.initial_balance
        self.shares_held: float = 0.0
        self.current_step: int = 0
        self.max_steps: int = len(self.df) - 1

        # Features per step
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        if self.use_sentiment:
            self.feature_cols.append("sentiment")
        self.n_features = len(self.feature_cols)

        # Observation space
        if self.window_size == 1:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
            )
        else:
            low = np.full((self.window_size, self.n_features), -np.inf, dtype=np.float32)
            high = np.full((self.window_size, self.n_features), np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        # History buffer for windowed obs
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        info : dict
            Empty info dict.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.window_buffer.clear()

        # Initialize buffer with first row(s)
        first_row = self.df.iloc[0]
        first_obs = np.array([first_row[col] for col in self.feature_cols], dtype=np.float32)

        if self.window_size == 1:
            self.window_buffer.append(first_obs)
        else:
            zero_obs = np.zeros(self.n_features, dtype=np.float32)
            for _ in range(self.window_size - 1):
                self.window_buffer.append(zero_obs)
            self.window_buffer.append(first_obs)

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Return current observation (windowed or single)."""
        if self.window_size == 1:
            return self.window_buffer[0]
        else:
            return np.stack(list(self.window_buffer)).astype(np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : int
            0=hold, 1=buy, 2=sell

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        current_price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0

        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.balance -= cost

        elif action == 2:  # Sell
            revenue = self.shares_held * current_price
            self.balance += revenue
            reward = revenue  # Reward = profit from selling
            self.shares_held = 0.0

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Update observation buffer
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            obs_vec = np.array([row[col] for col in self.feature_cols], dtype=np.float32)
            self.window_buffer.append(obs_vec)

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        """Render current state (console only)."""
        if mode != "human":
            return
        price = self.df.iloc[self.current_step]["close"]
        net_worth = self.balance + self.shares_held * price
        print(f"Step: {self.current_step} | Price: {price:,.2f} | "
              f"Shares: {self.shares_held} | Balance: ${self.balance:,.2f} | "
              f"Net Worth: ${net_worth:,.2f}")

    def close(self) -> None:
        """Cleanup (no-op)."""
        pass