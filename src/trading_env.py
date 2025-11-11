"""
Gymnasium-compatible trading environment.

Features
--------
- Discrete actions: 0=hold, 1=buy (all cash), 2=sell (all shares)
- Optional sentiment & news
- Commission per trade (read from config.yaml)
- Partial buys/sells (30%, 60%) → commented but ready to activate
- Preserves original Date index
- Safe indexing + assertions
- Compatible with Stable-Baselines3 + LSTM
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Dict, Any


import os

# --- Load commission from config.yaml (correct path) ---
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

COMMISSION_RATE = CONFIG.get("commission", 0.001)  # Default: 0.1%

class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agents.

    Think of this as a **stock market simulator** where an AI learns to trade.

    Observation
    -----------
    - window_size == 1 → [open, high, low, close, volume, (sentiment)]
    - window_size > 1 → stacked history (like short-term memory)

    Action Space
    ------------
    Discrete(3):
        0 → Hold (do nothing)
        1 → Buy (use ALL available cash)
        2 → Sell (sell ALL held shares)

    Reward
    ------
    Net profit from sell actions (after commission).

    Commission
    ----------
    Every buy/sell costs a % of the transaction value (e.g., 0.1%).
    Read from config.yaml → realistic broker fee.
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
        Initialize the trading simulator.

        Parameters
        ----------
        df : pd.DataFrame
            Must have 'Date' as index.
            Required columns: ['open', 'high', 'low', 'close', 'volume']
            Optional: 'sentiment' (news mood), 'news' (text)
        use_sentiment : bool
            Include sentiment in observation.
        initial_balance : float
            Starting cash (e.g., 10,000 €).
        window_size : int
            How many past days the AI "remembers" (for LSTM).
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
        # Fill missing news with empty string
        if "news" in self.df.columns:
            self.df["news"] = self.df["news"].fillna("")

        # Fill missing sentiment with 0 (neutral)
        if "sentiment" in self.df.columns:
            self.df["sentiment"] = self.df["sentiment"].fillna(0.0)

        # Forward/backward fill prices (handle gaps)
        price_cols = ["open", "high", "low", "close", "volume"]
        self.df[price_cols] = self.df[price_cols].ffill().bfill()

        # Final cleanup
        self.df = self.df.dropna().copy()
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after cleaning")

        # --- 3. Config ---
        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.commission_rate = COMMISSION_RATE  # e.g., 0.001 = 0.1%

        # --- 4. State (what changes during simulation) ---
        self.balance: float = self.initial_balance      # Cash in hand
        self.shares_held: float = 0.0                   # Number of shares owned
        self.current_step: int = 0                     # Current day index

        # --- 5. Features (what the AI sees) ---
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        if self.use_sentiment:
            self.feature_cols.append("sentiment")
        self.n_features = len(self.feature_cols)

        # --- 6. Observation space (what the AI can observe) ---
        if self.window_size == 1:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
            )
        else:
            low = np.full((self.window_size, self.n_features), -np.inf, dtype=np.float32)
            high = np.full((self.window_size, self.n_features), np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- 7. Action space ---
        # Current: 3 actions (hold, buy all, sell all)
        self.action_space = spaces.Discrete(3)

        # --- 8. History buffer (short-term memory) ---
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to day 1 with full cash and zero shares."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.window_buffer.clear()

        # Fill memory buffer with first day (or zeros for past)
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
        """Return current market data (single day or window)."""
        if self.window_size == 1:
            return self.window_buffer[0]
        else:
            return np.stack(list(self.window_buffer)).astype(np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading day.

        Actions:
            0 → Hold
            1 → Buy (use ALL cash)
            2 → Sell (sell ALL shares)

        Commission is applied on both buy and sell.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # --- 1. Get current price (before moving to next day) ---
        assert self.current_step < len(self.df), (
            f"Step {self.current_step} out of bounds (len={len(self.df)})"
        )
        current_price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0

        # --- 2. Execute action ---
        if action == 1:  # Buy (ALL-IN)
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                commission = cost * self.commission_rate
                total_cost = cost + commission
                if total_cost <= self.balance:
                    self.shares_held += shares_to_buy
                    self.balance -= total_cost

        elif action == 2:  # Sell (ALL-OUT)
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                commission = revenue * self.commission_rate
                net_revenue = revenue - commission
                self.balance += net_revenue
                reward = net_revenue  # Profit after commission
                self.shares_held = 0.0

        # --- 3. Advance to next day ---
        self.current_step += 1

        # --- 4. Check if simulation ends ---
        terminated = self.current_step >= len(self.df)
        truncated = False

        # --- 5. Update observation (next day's data) ---
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            obs_vec = np.array([row[col] for col in self.feature_cols], dtype=np.float32)
            self.window_buffer.append(obs_vec)

        return self._get_observation(), float(reward), terminated, truncated, {}

        # ------------------------------------------------------------------
        # FUTURE: Partial buys/sells (uncomment to activate)
        # ------------------------------------------------------------------
        # elif action == 3:  # Buy 30%
        #     cash_to_use = self.balance * 0.3
        #     shares_to_buy = cash_to_use // current_price
        #     if shares_to_buy > 0:
        #         cost = shares_to_buy * current_price
        #         commission = cost * self.commission_rate
        #         total_cost = cost + commission
        #         if total_cost <= self.balance:
        #             self.shares_held += shares_to_buy
        #             self.balance -= total_cost
        #
        # elif action == 4:  # Buy 60%
        #     cash_to_use = self.balance * 0.6
        #     shares_to_buy = cash_to_use // current_price
        #     # ... same as above ...
        #
        # elif action == 5:  # Sell 50%
        #     shares_to_sell = self.shares_held * 0.5
        #     if shares_to_sell > 0:
        #         revenue = shares_to_sell * current_price
        #         commission = revenue * self.commission_rate
        #         net_revenue = revenue - commission
        #         self.balance += net_revenue
        #         self.shares_held -= shares_to_sell
        #         reward = net_revenue

    def render(self, mode: str = "human") -> None:
        """Print current portfolio status (like a bank statement)."""
        if mode != "human":
            return
        price = (
            self.df.iloc[self.current_step - 1]["close"]
            if self.current_step > 0
            else self.df.iloc[0]["close"]
        )
        net_worth = self.balance + self.shares_held * price
        print(
            f"Day {self.current_step:3d} | "
            f"Price: {price:8.2f} € | "
            f"Shares: {self.shares_held:6.2f} | "
            f"Cash: {self.balance:10.2f} € | "
            f"Net Worth: {net_worth:10.2f} € | "
            f"Commission: {self.commission_rate*100:.3f}%"
        )

    def close(self) -> None:
        """Cleanup (nothing to do)."""
        pass