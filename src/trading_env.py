import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.dropna().reset_index(drop=True)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.max_steps = len(self.df)  # Mantener max_steps = len(self.df)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return self._get_observation(), {}

    def _get_observation(self):
        # Evitar acceso fuera de límites
        if self.current_step >= len(self.df):
            # Devolver la última observación disponible
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.current_step]
        obs = np.array([
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume'],
            row['sentiment']
        ], dtype=np.float32)
        return obs

    def step(self, action):
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
        pass