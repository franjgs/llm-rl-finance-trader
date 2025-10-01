import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """Custom Gym environment for stock trading with optional sentiment data."""
    
    def __init__(self, df, use_sentiment=True):
        """Initialize the trading environment.

        Args:
            df (pd.DataFrame): DataFrame with stock data (open, high, low, close, volume, sentiment).
            use_sentiment (bool): Whether to include sentiment in the observation space.
        """
        super(TradingEnv, self).__init__()
        # Clean and reset DataFrame index
        self.df = df.dropna().reset_index(drop=True)
        self.use_sentiment = use_sentiment
        self.current_step = 0
        self.balance = 10000  # Initial balance
        self.shares_held = 0  # Initial shares held
        self.max_steps = len(self.df)  # Set max_steps to DataFrame length

        # Define observation space (5 or 6 features depending on use_sentiment)
        obs_shape = 6 if self.use_sentiment else 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional reset options.

        Returns:
            tuple: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return self._get_observation(), {}

    def _get_observation(self):
        """Get the current observation from stock data.

        Returns:
            np.array: Array with [open, high, low, close, volume, (sentiment)].
        """
        # Handle out-of-bounds steps
        if self.current_step >= len(self.df):
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.current_step]
        obs = [
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ]
        if self.use_sentiment:
            obs.append(row['sentiment'])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action (int): Action to take (0: hold, 1: buy, 2: sell).

        Returns:
            tuple: (observation, reward, done, truncated, info).
        """
        # Get current price
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]['close']
        reward = 0
        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.shares_held += shares_bought
            self.balance -= shares_bought * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            reward = self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        """Render the environment (not implemented)."""
        pass