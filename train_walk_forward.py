#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_walk_forward.py
Created on Thu Nov 13 09:09:35 2025
@author: fran
Walk-forward training (1 day ahead) with full logging, evaluation, and comparison.
Supports:
- sentiment_mode: "with", "without", "both"
- LSTM policy
- Any SB3 algorithm (PPO default)
- TensorBoard logging
- EvalCallback with best model saving
- Sharpe ratio, outperformance vs Buy & Hold
- Learning curves per day
- Final walk-forward equity curve
"""
import argparse
import yaml
import pandas as pd
import logging
import torch
import os
import numpy as np
import random
import time
import hashlib
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.trading_env import TradingEnv
from src.plot_utils import plot_results
from src.rl_utils import CustomLstmPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# =============================================================================
# 0. CONFIG & HELPERS
# =============================================================================
def load_config(config_path):
    """
    Load the YAML configuration file.
    Args:
        config_path: Path to the YAML file.
    Returns:
        Dictionary with the parsed configuration.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise
def calculate_sharpe_ratio(net_worth):
    """
    Compute the annualized Sharpe ratio (risk-free rate = 0).
    Args:
        net_worth: Array-like of portfolio net worth.
    Returns:
        Sharpe ratio. Returns 0.0 if invalid.
    """
    if len(net_worth) < 2:
        return 0.0
    returns = np.diff(net_worth) / net_worth[:-1]
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return np.mean(returns) / std_return * np.sqrt(252)
# =============================================================================
# 1. CLI & CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config_walk_forward.yaml", help="Path to config file")
parser.add_argument("--random-seed", action="store_true", help="Use random seed")
args = parser.parse_args()
config = load_config(args.config)
verbose = config.get("verbose", 0)
# --- Core parameters ---
symbol = config["stock_symbol"]
initial_balance = config.get("initial_balance", 10000)
raw_dir = config["raw_dir"]
processed_dir = config["processed_dir"]
sentiment_mode = config.get("sentiment_mode", "individual")
sentiment_source = config.get("sentiment_source", "finnhub_orig")
algo_name = config.get("algo", "PPO")
use_lstm = config.get("use_lstm", False)
lstm_window = config.get("lstm_window", 32)
lstm_hidden_size = config.get("lstm_hidden_size", 64)
timesteps = config["timesteps"]
walk_forward = config.get("walk_forward", True)
prediction_horizon = config.get("prediction_horizon", 1)
tb_log_base_path = config.get("tensorboard_log_dir", "logs/tensorboard_walk_forward")

device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Seed handling ---
config_seed = config.get("seed")
entropy_str = f"{time.time()}_{os.getpid()}_{random.getrandbits(64)}"
global_entropy_seed = int(hashlib.sha256(entropy_str.encode()).hexdigest(), 16) % (2**32)
if args.random_seed:
    base_seed = random.randint(0, 999999)
    logger.info(f"CLI override → Using RANDOM seed: {base_seed}")
elif config_seed is None:
    base_seed = random.randint(0, 999999)
    logger.info(f"Config seed is null → Using RANDOM seed: {base_seed}")
else:
    base_seed = config_seed
    logger.info(f"Using FIXED seed from config: {base_seed}")
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(base_seed)

# === DATA PATH ===
raw_csv = os.path.join(raw_dir, f"{symbol}_raw.csv")
processed_csv = os.path.join(processed_dir, f"{symbol}_sentiment_{sentiment_source if sentiment_mode == 'individual' else 'combined'}.csv")
data_path = processed_csv if os.path.exists(processed_csv) else raw_csv
logger.info(f"Using data: {data_path}")

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)
df = df.dropna()
logger.info(f"Data: {len(df)} days from {df.index.min()} to {df.index.max()}")

# =============================================================================
# 2. SIMULATE FUNCTION
# =============================================================================
def simulate(
    model,
    use_sentiment: bool,
    df: pd.DataFrame,
    initial_balance: float,
    window_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run a full trading episode on a **specific DataFrame** (train or test).
    Parameters
    ----------
    model : stable_baselines3.PPO
        Trained PPO model (with or without sentiment).
    use_sentiment : bool
        ``True`` → include sentiment in observations; ``False`` → ignore it.
    df : pd.DataFrame
        Data slice to simulate on. **Must have ``Date`` as index** and columns
        ``['open','high','low','close','volume']`` (``sentiment`` optional).
    initial_balance : float
        Starting cash (e.g. 10 000 €).
    window_size : int
        Number of past days the agent remembers (1 for MLP, >1 for LSTM).
    seed : int
        Random seed for environment reset → reproducible simulations.
    Returns
    -------
    pd.DataFrame
        Index = original ``Date``; columns = ``['net_worth','action']``.
        ``net_worth`` = cash + shares × current close price.
        ``action`` = 0 (hold), 1 (buy all), 2 (sell all).
    """
    logger.info(
        f"simulate() → {len(df)} rows | "
        f"Date range: {df.index[0].date()} → {df.index[-1].date()} | "
        f"Sentiment={'ON' if use_sentiment else 'OFF'}"
    )
    env = TradingEnv(
        df=df,
        use_sentiment=use_sentiment,
        initial_balance=initial_balance,
        window_size=window_size,
    )
    obs, _ = env.reset(seed=seed)
    sim_df = pd.DataFrame(
        index=df.index,
        columns=["net_worth", "action"],
        dtype=float,
    )
    for step in range(len(df)):
        current_price = float(df.iloc[step]["close"])
        net_worth = env.balance + env.shares_held * current_price
        action, _ = model.predict(obs, deterministic=False)
        action = int(action.item()) if hasattr(action, "item") else int(action)
        sim_df.loc[df.index[step], "net_worth"] = net_worth
        sim_df.loc[df.index[step], "action"] = action
        obs, reward, terminated, truncated, _ = env.step(action)
    logger.info(f"Simulation finished → {len(sim_df)} rows")
    return sim_df

# =============================================================================
# 3. CALLBACKS
# =============================================================================
class RewardLogger(BaseCallback):
    """Log episode rewards for learning curve."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0 and "episode" in self.locals["infos"][0]:
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info["r"])
            self.episode_lengths.append(info["l"])

            if "net_worth" in self.locals["infos"][0]:
                self.logger.record("eval/net_worth", self.locals["infos"][0]["net_worth"])
        return True

# --- Flush TensorBoard every 100 steps ---
class FlushTensorBoard(BaseCallback):
    """Force TensorBoard to write events periodically."""
    def __init__(self, flush_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.flush_freq = flush_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.flush_freq == 0:
            self.logger.dump(self.num_timesteps)
        return True
    
# =============================================================================
# 4. ALGORITHM SETUP
# =============================================================================
common_kwargs = {
    "seed": base_seed,
    "device": device,
    "verbose": 0 # TensorBoard handles logging
}
if algo_name == "PPO":
    AlgoClass = PPO
    algo_kwargs = {
        "learning_rate": config.get("ppo_learning_rate", 1e-4),
        "gamma": config.get("ppo_gamma", 0.99),
        "n_steps": (lstm_window if use_lstm else config.get("ppo_n_steps", 256)),
        "batch_size": (lstm_window if use_lstm else config.get("ppo_batch_size", 128)),
        "gae_lambda": config.get("ppo_gae_lambda", 0.95),
        "clip_range": config.get("ppo_clip_range", 0.1),
        "ent_coef": config.get("ppo_ent_coef", 0.001),
        **common_kwargs
    }
elif algo_name == "SAC":
    AlgoClass = SAC
    algo_kwargs = {
        "buffer_size": config.get("sac_buffer_size", 1000000),
        "learning_starts": config.get("sac_learning_starts", 1000),
        "batch_size": config.get("sac_batch_size", 256),
        "ent_coef": config.get("sac_ent_coef", "auto"),
        "learning_rate": config.get("ppo_learning_rate", 1e-4),
        **common_kwargs
    }
elif algo_name == "TD3":
    AlgoClass = TD3
    algo_kwargs = {
        "buffer_size": config.get("td3_buffer_size", 1000000),
        "learning_starts": config.get("td3_learning_starts", 1000),
        "train_freq": config.get("td3_train_freq", 1),
        "gradient_steps": config.get("td3_gradient_steps", 1),
        "learning_rate": config.get("ppo_learning_rate", 1e-4),
        **common_kwargs
    }
elif algo_name == "A2C":
    AlgoClass = A2C
    algo_kwargs = {
        "n_steps": config.get("a2c_n_steps", 5),
        "gamma": config.get("a2c_gamma", 0.99),
        "ent_coef": config.get("a2c_ent_coef", 0.01),
        "learning_rate": config.get("ppo_learning_rate", 1e-4),
        **common_kwargs
    }
else:
    raise ValueError(f"Unknown algorithm: {algo_name}")
logger.info(f"Selected algorithm: {algo_name} with params: {algo_kwargs}")

# =============================================================================
# 5. WALK-FORWARD LOOP
# =============================================================================
os.makedirs("results/walk_forward", exist_ok=True)
os.makedirs("models/best_walk_forward", exist_ok=True)
os.makedirs("logs/tensorboard_walk_forward", exist_ok=True)
modes = []
if sentiment_mode == "with": modes = [True]
elif sentiment_mode == "without": modes = [False]
elif sentiment_mode == "both": modes = [True, False]
else: raise ValueError("sentiment_mode must be: with, without, both")
all_results = {True: [], False: []}
min_train_days = 100
start_idx = min_train_days
end_idx = len(df) - prediction_horizon
logger.info(f"Walk-forward: {start_idx} → {end_idx} (predicting {prediction_horizon} day(s) ahead)")
for pred_idx in range(start_idx, end_idx + 1):
    train_slice = df.iloc[:pred_idx]
    test_slice = df.iloc[pred_idx:pred_idx + prediction_horizon]
    pred_date = test_slice.index[-1].date()
    for use_sentiment in modes:
        label = "with" if use_sentiment else "without"
        logger.info(f"Predicting {pred_date} | Train: {len(train_slice)} days | {label}")
        # --- Seed per day ---
        day_entropy = f"{pred_idx}_{use_sentiment}_{global_entropy_seed}".encode()
        day_seed = int(hashlib.sha256(day_entropy).hexdigest(), 16) % (2**32)
        # --- Environment ---
        window_size = lstm_window if use_lstm else 1
        env_train = Monitor(TradingEnv(train_slice, use_sentiment, initial_balance, window_size))
        env_eval = Monitor(TradingEnv(test_slice, use_sentiment, initial_balance, window_size))
        vec_train = DummyVecEnv([lambda: env_train])
        vec_eval = DummyVecEnv([lambda: env_eval])
        # --- WARM START: Load previous best model ---
        prev_date = (pd.Timestamp(pred_date) - pd.Timedelta(days=1)).date()
        model_path = f"models/best_walk_forward/{symbol}_{prev_date}_{label}/best_model.zip"
        if config.get("warm_start", False) and pred_idx > start_idx and os.path.exists(model_path):
            logger.info(f"Warm start: loading model from {model_path}")
            model = AlgoClass.load(model_path, env=vec_train, **algo_kwargs, weights_only=True)
            train_steps = config.get("incremental_timesteps", 1000)
        else:
            # --- Model ---
            policy_kwargs = None
            if use_lstm and algo_name == "PPO":
                policy_kwargs = dict(
                    features_extractor_class=CustomLstmPolicy,
                    features_extractor_kwargs=dict(lstm_hidden_size=lstm_hidden_size, n_lstm_layers=1),
                    net_arch=[],
                )
            model = AlgoClass(
                "MlpPolicy", vec_train,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"{tb_log_base_path}/{symbol}_{pred_date}_{label}",
                **algo_kwargs
            )
            train_steps = timesteps
        # --- Callbacks ---
        reward_logger = RewardLogger()
        eval_callback = EvalCallback(
            vec_eval,
            best_model_save_path=f"models/best_walk_forward/{symbol}_{pred_date}_{label}",
            log_path=f"logs/eval_walk_forward/{symbol}_{pred_date}_{label}",
            eval_freq=max(1000, train_steps // 50),
            deterministic=True,
            render=False,
            n_eval_episodes=1,
            verbose=0
        )
        flush_callback = FlushTensorBoard(flush_freq=100)  

        # --- Train ---
        model.learn(
            total_timesteps=train_steps,
            callback=[reward_logger, eval_callback, flush_callback],  # ← AÑADIDO
            tb_log_name=f"{label}_sentiment",
            reset_num_timesteps=False
        )
        # --- Simulate ---
        sim = simulate(model, use_sentiment, test_slice, initial_balance, window_size, day_seed)
        final_nw = sim["net_worth"].iloc[-1]
        # --- Save learning curve ---
        lc_df = pd.DataFrame({
            "episode": range(len(reward_logger.episode_rewards)),
            "reward": reward_logger.episode_rewards
        })
        lc_path = f"results/walk_forward/{symbol}_{pred_date}_{label}_learning_curve.csv"
        lc_df.to_csv(lc_path, index=False)
        # --- Save result ---
        all_results[use_sentiment].append({
            "date": pred_date,
            "net_worth": final_nw,
            "action": int(sim["action"].iloc[-1])
        })
        
# =============================================================================
# 6. SAVE & PLOT
# =============================================================================
net_worth_with_mean = None
net_worth_without_mean = None
if True in modes:
    df_with = pd.DataFrame(all_results[True])
    csv_with = f"results/walk_forward/{symbol}_walk_forward_1day_with.csv"
    df_with.to_csv(csv_with, index=False)
    logger.info(f"Saved: {csv_with}")
    net_worth_with_mean = pd.Series(df_with["net_worth"].values, index=pd.to_datetime(df_with["date"]))
if False in modes:
    df_without = pd.DataFrame(all_results[False])
    csv_without = f"results/walk_forward/{symbol}_walk_forward_1day_without.csv"
    df_without.to_csv(csv_without, index=False)
    logger.info(f"Saved: {csv_without}")
    net_worth_without_mean = pd.Series(df_without["net_worth"].values, index=pd.to_datetime(df_without["date"]))
# === FINAL PLOT ===
plot_results(
    walk_forward=True,
    symbol=symbol,
    seed=base_seed,
    use_lstm=use_lstm,
    net_worth_with_mean=net_worth_with_mean,
    net_worth_without_mean=net_worth_without_mean
)
logger.info("Walk-forward training completed.")