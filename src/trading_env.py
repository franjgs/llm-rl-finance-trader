"""
Simple Gymnasium trading environment for educational purposes (TFG).

Features
--------
- Discrete actions: 0 = hold, 1 = buy all cash, 2 = sell all shares
- Optional sentiment in observations
- Initial balance configurable
- No commissions (kept simple for learning)
- Safe indexing (handles episode end correctly)
- Compatible with Stable-Baselines3 PPO
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agents.

    The agent starts with cash and can buy/sell shares of one stock.
    Observation: price data (open, high, low, close, volume) + optional sentiment.
    Actions: hold, buy all, sell all.
    Reward: profit when selling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        initial_balance: float = 10_000.0,
    ) -> None:
        """
        Initialize the environment.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: ['open', 'high', 'low', 'close', 'volume']
            Optional: 'sentiment'
            Index will be reset to 0..N for safe iloc access.
        use_sentiment : bool
            Include sentiment in observations.
        initial_balance : float
            Starting cash (e.g., 10,000 €).
        """
        super().__init__()

        # Clean data and reset index for safe iloc access
        self.df = df.dropna().reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after dropping NaN")

        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)

        # State variables
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = 0

        # Observation: 5 price features + optional sentiment
        n_features = 5 + (1 if self.use_sentiment else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state (day 0, full cash, no shares)."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Return current day's data as observation.
        Safe: when episode ends, returns the last valid day.
        """
        # Clamp index to last row when episode is finished
        idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx]

        obs = [
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
        ]
        if self.use_sentiment:
            obs.append(row.get("sentiment", 0.0))

        return np.array(obs, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading day.

        Actions:
            0 → Hold
            1 → Buy (use all available cash)
            2 → Sell (sell all shares)

        Reward: profit when selling.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        current_price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0

        if action == 1:  # Buy all
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.balance -= cost

        elif action == 2:  # Sell all
            revenue = self.shares_held * current_price
            self.balance += revenue
            reward = revenue  # Simple reward = money received
            self.shares_held = 0.0

        # Advance to next day
        self.current_step += 1

        # Termination: end of data
        terminated = self.current_step >= len(self.df)
        truncated = False

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        """Print current portfolio status."""
        if mode != "human":
            return
        price = self.df.iloc[self.current_step - 1]["close"] if self.current_step > 0 else self.df.iloc[0]["close"]
        net_worth = self.balance + self.shares_held * price
        print(
            f"Day {self.current_step:3d} | Price: {price:8.2f} | "
            f"Shares: {self.shares_held:6.0f} | Cash: {self.balance:10.2f} | "
            f"Net Worth: {net_worth:10.2f}"
        )

    def close(self) -> None:
        """Cleanup (no-op)."""
        pass