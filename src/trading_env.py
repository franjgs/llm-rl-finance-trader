"""
Gymnasium-compatible trading environment.

Features
--------
- Discrete actions: 0=hold, 1=buy (all cash), 2=sell (all shares)
- Optional sentiment feature
- Configurable initial balance
- Windowed observations (window_size > 1) for LSTM
- Preserves original Date index (no reset_index)
- Handles optional 'news' column (filled with '')
- Safe indexing with assertions
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
    - window_size > 1 → stacked history (shape: (window_size, n_features))

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
            Must have 'Date' as index and columns:
            ['open', 'high', 'low', 'close', 'volume']
            Optional: 'sentiment', 'news'
        use_sentiment : bool
            Include sentiment in observation if available.
        initial_balance : float
            Starting cash.
        window_size : int
            Number of past steps to include in observation (for LSTM).
        """
        super().__init__()

        # --- 1. Validate and preserve Date index ---
        if df.index.name != "Date":
            if "Date" in df.columns:
                df = df.set_index("Date")
            else:
                raise ValueError("df must have 'Date' as index or column")

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        self.original_df = df.copy()
        self.df = df.copy()

        # --- 2. Clean data safely (no reset_index) ---
        # Fill 'news' with empty string
        if "news" in self.df.columns:
            self.df["news"] = self.df["news"].fillna("")

        # Fill 'sentiment' with 0.0
        if "sentiment" in self.df.columns:
            self.df["sentiment"] = self.df["sentiment"].fillna(0.0)

        # Forward/backward fill price/volume columns
        price_cols = ["open", "high", "low", "close", "volume"]
        self.df[price_cols] = self.df[price_cols].ffill().bfill()

        # Final dropna (should keep all rows now)
        # before = len(self.df)
        self.df = self.df.dropna().copy()
        after = len(self.df)
        if after == 0:
            raise ValueError("DataFrame is empty after cleaning")
        # print(f"TradingEnv: {before} → {after} rows after cleaning")  # Debug

        # --- 3. Config ---
        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)

        # --- 4. State ---
        self.balance: float = self.initial_balance
        self.shares_held: float = 0.0
        self.current_step: int = 0

        # --- 5. Features ---
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        if self.use_sentiment:
            self.feature_cols.append("sentiment")
        self.n_features = len(self.feature_cols)

        # --- 6. Observation space ---
        if self.window_size == 1:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
            )
        else:
            low = np.full((self.window_size, self.n_features), -np.inf, dtype=np.float32)
            high = np.full((self.window_size, self.n_features), np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        # --- 7. History buffer ---
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.window_buffer.clear()

        # Initialize buffer
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
        - Uses self.current_step BEFORE increment
        - Safe indexing with assertion
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # --- 1. Use current_step BEFORE increment ---
        assert self.current_step < len(self.df), (
            f"current_step {self.current_step} out of bounds (len(df)={len(self.df)})"
        )
        current_price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0

        # --- 2. Execute action ---
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.balance -= cost
        elif action == 2:  # Sell
            revenue = self.shares_held * current_price
            self.balance += revenue
            reward = revenue
            self.shares_held = 0.0

        # --- 3. Advance step AFTER using price ---
        self.current_step += 1

        # --- 4. Termination ---
        terminated = self.current_step >= len(self.df)
        truncated = False

        # --- 5. Update observation buffer ---
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            obs_vec = np.array([row[col] for col in self.feature_cols], dtype=np.float32)
            self.window_buffer.append(obs_vec)

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        """Render current state (console only)."""
        if mode != "human":
            return
        price = (
            self.df.iloc[self.current_step - 1]["close"]
            if self.current_step > 0
            else self.df.iloc[0]["close"]
        )
        net_worth = self.balance + self.shares_held * price
        print(
            f"Step: {self.current_step:3d} | "
            f"Price: {price:8.2f} | "
            f"Shares: {self.shares_held:6.2f} | "
            f"Balance: ${self.balance:10.2f} | "
            f"Net Worth: ${net_worth:10.2f}"
        )

    def close(self) -> None:
        """Cleanup (no-op)."""
        pass