"""
Gymnasium-compatible trading environment.

Features
--------
- Discrete actions: 0=hold, 1=buy (all cash), 2=sell (all shares), 
  3=buy (30% cash), 4=buy (60% cash), 5=sell (50% shares)
- Optional sentiment & news
- Commission per trade (read from config.yaml)
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
    # Fallback for environments where the config path is tricky
    COMMISSION_RATE = 0.001
else:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
        COMMISSION_RATE = CONFIG.get("commission", 0.001)  # Default: 0.1%


class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agents.

    Observation
    -----------
    - All features (open, high, low, close, volume, sentiment) are normalized to [0, 1].
    - window_size == 1 → [normalized features]
    - window_size > 1 → stacked history (like short-term memory)

    Action Space
    ------------
    Discrete(6):
        0 → Hold (do nothing)
        1 → Buy (use ALL available cash)
        2 → Sell (sell ALL held shares)
        3 → Buy (use 30% available cash)
        4 → Buy (use 60% available cash)
        5 → Sell (sell 50% held shares)

    Reward
    ------
    Daily % change in portfolio net worth **using previous day's close price** as baseline.
    This allows the agent to profit from price movements even if it only holds.

    Commission
    ----------
    Same % fee applied on **both buy and sell** (e.g., 0.1%).
    Read from config.yaml → realistic broker fee.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        initial_balance: float = 10_000.0,
        window_size: int = 1,
        commission: Optional[float] = None,
        prediction_horizon: int = 1
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
        commission : float, optional
            Transaction fee rate. If None, uses value from config.yaml.
        prediction_horizon : int
            Number of days to predict (used in eval env).
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

        # --- 2. Clean data safely ---
        if "news" in self.df.columns:
            self.df["news"] = self.df["news"].fillna("")
        if "sentiment" in self.df.columns:
            self.df["sentiment"] = self.df["sentiment"].fillna(0.0)

        price_cols = ["open", "high", "low", "close", "volume"]
        self.df[price_cols] = self.df[price_cols].ffill().bfill()
        self.df = self.df.dropna().copy()

        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after cleaning")

        # --- 3. Config ---
        self.use_sentiment = use_sentiment and "sentiment" in self.df.columns
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.commission_rate = commission if commission is not None else COMMISSION_RATE
        self.prediction_horizon = prediction_horizon

        # --- 4. State ---
        self.balance: float = self.initial_balance
        self.shares_held: float = 0.0
        self.current_step: int = 0
        self.net_worth: float = self.initial_balance

        # --- 5. Features ---
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        if self.use_sentiment:
            self.feature_cols.append("sentiment")
        self.n_features = len(self.feature_cols)

        # --- 6. Normalization (per-environment, no leakage) ---
        data_to_normalize = self.df[self.feature_cols].values
        self.data_min = np.min(data_to_normalize, axis=0)
        self.data_max = np.max(data_to_normalize, axis=0)
        self.data_range = self.data_max - self.data_min
        self.data_range[self.data_range == 0] = 1.0e-8  # avoid div by zero

        # --- 7. Observation space ---
        if self.window_size == 1:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32
            )
        else:
            low = np.zeros((self.window_size, self.n_features), dtype=np.float32)
            high = np.ones((self.window_size, self.n_features), dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- 8. Action space (Updated to Discrete(6) for partial actions) ---
        self.action_space = spaces.Discrete(6)

        # --- 9. History buffer ---
        self.window_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def _normalize_data(self, data_row: np.ndarray) -> np.ndarray:
        """MinMax normalization using training slice stats."""
        normalized = (data_row - self.data_min) / self.data_range
        return np.clip(normalized, 0.0, 1.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to day 1 with full cash and zero shares."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.window_buffer.clear()

        first_row_data = self.df.iloc[0][self.feature_cols].values.astype(np.float32)
        first_obs = self._normalize_data(first_row_data)

        if self.window_size == 1:
            self.window_buffer.append(first_obs)
        else:
            zero_obs = np.zeros(self.n_features, dtype=np.float32)
            for _ in range(self.window_size - 1):
                self.window_buffer.append(zero_obs)
            self.window_buffer.append(first_obs)

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Return current observation (window), float32 for MPS."""
        if self.window_size == 1:
            obs = self.window_buffer[0]
        else:
            obs = np.stack(list(self.window_buffer)).astype(np.float32)
        return obs.astype(np.float32)

    def _execute_buy(self, current_price: float, percentage: float = 1.0) -> None:
        """
        Execute a buy order using a given percentage of available cash.
    
        Parameters
        ----------
        current_price : float
            Current market close price of the asset.
        percentage : float, default 1.0
            Fraction of available cash to spend (0.0 → 1.0).
    
        Notes
        -----
        * Allows **fractional shares** – essential for walk-forward training.
        * Commission is applied on the **gross** trade value (standard broker model).
        * The commission is deducted from the cash allocated before calculating shares,
          guaranteeing that the total cost never exceeds the intended amount.
        """
        cash_to_use = self.balance * percentage
        if cash_to_use <= 0.0:
            return
    
        # Cash left after paying commission (standard broker calculation)
        cash_after_commission = cash_to_use * (1.0 - self.commission_rate)
    
        # Fractional shares are allowed → the agent can always execute the order
        shares_to_buy = cash_after_commission / current_price
    
        if shares_to_buy > 0.0:
            cost = shares_to_buy * current_price
            commission_amount = cost * self.commission_rate
            total_cost = cost + commission_amount
    
            # Tiny tolerance for floating-point rounding
            if total_cost <= self.balance + 1e-8:
                self.shares_held += shares_to_buy
                self.balance -= total_cost
    
    
    def _execute_sell(self, current_price: float, percentage: float = 1.0) -> None:
        """
        Execute a sell order for a given percentage of currently held shares.
    
        Parameters
        ----------
        current_price : float
            Current market close price of the asset.
        percentage : float, default 1.0
            Fraction of held shares to sell (0.0 → 1.0).
    
        Notes
        -----
        * Allows **fractional shares** – keeps the environment consistent with the buy side.
        * Commission is applied on the **gross revenue** of the sale (standard broker model).
        * Prevents tiny negative share dust due to floating-point errors.
        """
        shares_to_sell = self.shares_held * percentage
    
        if shares_to_sell <= 0.0:
            return
    
        revenue = shares_to_sell * current_price
        commission_amount = revenue * self.commission_rate
        net_revenue = revenue - commission_amount
    
        self.balance += net_revenue
        self.shares_held -= shares_to_sell
    
        # Clean up tiny negative values caused by rounding errors
        if self.shares_held < 1e-8:
            self.shares_held = 0.0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute **one trading day**.

        **Reward** = % change in portfolio value from **previous day’s close price**.
        This gives credit for holding during price rises.

        **Commission** = same % on **both buy and sell**.
        """
        # --------------------------------------------------------------------- #
        # 1. Validate action
        # --------------------------------------------------------------------- #
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # --------------------------------------------------------------------- #
        # 2. Current close price
        # --------------------------------------------------------------------- #
        assert self.current_step < len(self.df), (
            f"Step {self.current_step} out of bounds (len={len(self.df)})"
        )
        current_price = float(self.df.iloc[self.current_step]["close"])

        # --------------------------------------------------------------------- #
        # 3. Previous close price (baseline for reward)
        # --------------------------------------------------------------------- #
        prev_price = (
            float(self.df.iloc[self.current_step - 1]["close"])
            if self.current_step > 0
            else current_price
        )

        # --------------------------------------------------------------------- #
        # 4. Portfolio value using previous price
        # --------------------------------------------------------------------- #
        prev_portfolio_value = self.balance + self.shares_held * prev_price

        # --------------------------------------------------------------------- #
        # 5. Execute action (Updated to handle 6 actions)
        # --------------------------------------------------------------------- #
        if action == 1:  # BUY – ALL-IN
            self._execute_buy(current_price, percentage=1.0)

        elif action == 2:  # SELL – ALL-OUT
            self._execute_sell(current_price, percentage=1.0)
            
        elif action == 3:  # BUY – 10% Partial
            self._execute_buy(current_price, percentage=0.10)

        elif action == 4:  # BUY – 30% Partial
            self._execute_buy(current_price, percentage=0.30)
            
        elif action == 5:  # SELL – 50% Partial
            self._execute_sell(current_price, percentage=0.50)

        # Action 0 (Hold) does nothing here, which is correct.

        # --------------------------------------------------------------------- #
        # 6. Current portfolio value
        # --------------------------------------------------------------------- #
        current_portfolio_value = self.balance + self.shares_held * current_price
        self.net_worth = current_portfolio_value

        # --------------------------------------------------------------------- #
        # 7. Reward = Logarithmic return (scale-invariant & PPO-friendly)
        # --------------------------------------------------------------------- #
        # We use **log returns** instead of raw € or percentage change.
        # Why this is the gold-standard in financial RL:
        #   • Stationary magnitude: a +5 % day always gives ~0.048 reward,
        #     regardless of portfolio size → value function never explodes.
        #   • Naturally handles compounding and is mathematically sound.
        #   • PPO (and all policy-gradient methods) converge dramatically faster
        #     and more stably with this reward shape.
        #   • Used in virtually every published paper on deep RL for trading.
        if prev_portfolio_value > 0.0:
            reward = np.log(current_portfolio_value / prev_portfolio_value) * 100  # +1% → +1.0 reward
        else:
            reward = 0.0

        # Optional tiny exposure bonus (tie-breaker)
        #   • Prevents the agent from being completely indifferent between
        #     holding cash and being fully invested when expected return is zero.
        #   • Very small magnitude (≈ 0.0001 per day when 100 % invested) so it
        #     never dominates the main log-return signal.
        exposure = (
            (self.shares_held * current_price) / current_portfolio_value
            if current_portfolio_value > 0.0
            else 0.0
        )
        reward += 0.003 * exposure
        # --------------------------------------------------------------------- #
        # --------------------------------------------------------------------- #
        # 8. Advance day
        # --------------------------------------------------------------------- #
        self.current_step += 1

        # --------------------------------------------------------------------- #
        # 9. Termination
        # --------------------------------------------------------------------- #
        terminated = self.current_step >= len(self.df)
        truncated = self.current_step >= self.prediction_horizon

        # --------------------------------------------------------------------- #
        # 10. Next observation
        # --------------------------------------------------------------------- #
        if self.current_step < len(self.df):
            row_data = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
            obs_vec = self._normalize_data(row_data)
            self.window_buffer.append(obs_vec)

        # --------------------------------------------------------------------- #
        # 11. Debug print – shows the REAL situation every step
        # --------------------------------------------------------------------- #
        """
        print(
            f"[SIM] Step {self.current_step-1:2d} | "
            f"Price: {current_price:7.2f} € | "
            f"Action: {action} | "
            f"Shares: {self.shares_held:8.3f} | "
            f"Cash: {self.balance:10.2f} € | "
            f"Net Worth: {self.net_worth:10.2f} €"
        )
        """
        # --------------------------------------------------------------------- #
        # 12. Return step info
        # --------------------------------------------------------------------- #
        info = {"net_worth": self.net_worth}
        return self._get_observation(), float(reward), terminated, truncated, info


    def render(self, mode: str = "human") -> None:
        """Print current portfolio status using the CORRECT current price."""
        if mode != "human":
            return

        # Use the price of the day we just finished (current_step points to next day)
        price = (
            self.df.iloc[self.current_step - 1]["close"]
            if self.current_step > 0
            else self.df.iloc[0]["close"]
        )

        print(
            f"[RENDER] Day {self.current_step:3d} | "
            f"Price: {price:8.2f} € | "
            f"Shares: {self.shares_held:8.3f} | "
            f"Cash: {self.balance:10.2f} € | "
            f"Net Worth: {self.net_worth:10.2f} € | "
            f"Commission: {self.commission_rate*100:.3f}%"
        )

    def close(self) -> None:
        """Cleanup."""
        pass