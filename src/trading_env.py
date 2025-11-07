# src/trading_env.py
"""
PPO-Compatible Trading Environment for Single-Asset Trading with Sentiment.
Features:
- Continuous action space: -1 (full short) to +1 (full long)
- Fractional position management
- Commission cost (0.05% default)
- Reward: PnL - commission
- Optional sentiment signal
- Real-time rendering with Matplotlib
- Compatible with Gymnasium, PPO, and Stable-Baselines3
- Date as column (required)
- net_worth calculated internally
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for financial trading with sentiment.

    Action Space:
        Box(-1, 1) → target position (-1 = full short, +1 = full long)

    Observation Space:
        [close, volume, sentiment, open, high, low] (if use_sentiment=True)

    Reward:
        PnL from position change - commission cost
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        commission: float = 0.0005,
        initial_balance: float = 10_000.0
    ):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): Must have columns ['Date', 'open', 'high', 'low', 'close', 'volume', 'sentiment']
            use_sentiment (bool): Include sentiment in observation
            commission (float): Trading commission per unit of position change
            initial_balance (float): Starting cash
        """
        super().__init__()
        if "Date" not in df.columns:
            raise ValueError("DataFrame must have 'Date' as a column (not index)")
        self.df = df.dropna().reset_index(drop=True)
        self.use_sentiment = use_sentiment
        self.commission = commission
        self.initial_balance = initial_balance

        # State variables
        self.balance = initial_balance
        self.position = 0.0  # Current position (-1 to +1)
        self.net_worth = initial_balance
        self.current_step = 0
        self.max_steps = len(self.df) - 1  # Need next price for PnL

        # Observation: close, volume, sentiment, open, high, low
        obs_dim = 6 if use_sentiment else 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Continuous action: target position
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Rendering
        self.fig = None
        self.history = []  # For backtesting

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            obs (np.ndarray): Initial observation
            info (dict): Empty info dict
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.history = []

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.

        Returns:
            np.ndarray: [close, volume, sentiment, open, high, low] or subset
        """
        idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        obs = [
            row["close"],
            row["volume"],
            row.get("sentiment", 0.0) if self.use_sentiment else 0.0,
            row["open"],
            row["high"],
            row["low"]
        ]
        return np.array(obs[:6 if self.use_sentiment else 5], dtype=np.float32)

    def step(self, action):
        """
        Execute one time step.

        Args:
            action: Target position. Accepts float or np.ndarray.

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Accept float or array
        if isinstance(action, np.ndarray):
            action = float(np.clip(action[0], -1, 1))
        else:
            action = float(np.clip(action, -1, 1))

        current_price = self.df.iloc[self.current_step]["close"]

        # Next price for PnL
        next_price = (
            self.df.iloc[self.current_step + 1]["close"]
            if self.current_step + 1 < len(self.df)
            else current_price
        )

        # Trade execution
        trade = action - self.position
        commission_cost = abs(trade) * current_price * self.commission
        self.balance -= commission_cost
        self.position = action

        # PnL calculation
        pnl = self.position * (next_price - current_price)
        self.balance += pnl
        self.net_worth = self.balance + self.position * next_price
        reward = pnl - commission_cost

        # Step forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Record history (safe index)
        history_step = min(self.current_step, len(self.df) - 1)
        self.history.append({
            "step": self.current_step,
            "date": self.df.iloc[history_step]["Date"],
            "action": action,
            "price": current_price,
            "net_worth": self.net_worth,
            "sentiment": self.df.iloc[history_step].get("sentiment", 0.0),
            "pnl": pnl,
            "commission": commission_cost
        })

        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode: str = "human"):
        """
        Render the environment (Matplotlib).

        Args:
            mode (str): Only "human" supported
        """
        if mode != "human":
            return

        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            plt.ion()

        plt.clf()

        steps = min(self.current_step + 1, len(self.df))
        dates = self.df["Date"].iloc[:steps]
        prices = self.df["close"].iloc[:steps]

        # Price + Position
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(dates, prices, label="Close Price", color="blue", linewidth=1.5)
        if self.current_step < len(self.df):
            color = "green" if self.position > 0 else "red" if self.position < 0 else "gray"
            ax1.scatter(
                dates.iloc[self.current_step],
                prices.iloc[self.current_step],
                color=color, s=80, marker="o", zorder=5
            )
        ax1.set_title(f"Step {self.current_step} | Net Worth: ${self.net_worth:,.2f} | Position: {self.position:+.2f}")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Net Worth
        ax2 = plt.subplot(2, 1, 2)
        nw_history = [self.initial_balance] + [h["net_worth"] for h in self.history]
        ax2.plot(range(len(nw_history)), nw_history, label="Net Worth", color="purple")
        ax2.set_ylabel("Net Worth ($)")
        ax2.set_xlabel("Step")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        """Close the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None