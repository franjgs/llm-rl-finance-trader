# train_model.py
"""
Train PPO agents with/without sentiment using config_beta.yaml.
Supports:
- Replicates with different seeds
- LSTM policy
- Any algo from config (PPO, A2C, etc.)
- Same behavior as original when replicates=1
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
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import TradingEnv
from src.plot_utils import plot_results
from src.rl_utils import CustomLstmPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            # logger.info(f"YAML content:\n{content}")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

def calculate_sharpe_ratio(net_worth):
    if len(net_worth) < 2:
        logger.warning("Net worth series too short to calculate Sharpe Ratio")
        return 0.0
    returns = np.diff(net_worth) / net_worth[:-1]
    std_return = np.std(returns)
    if std_return == 0:
        logger.warning("Zero volatility in returns, Sharpe Ratio undefined")
        return 0.0
    return np.mean(returns) / std_return * np.sqrt(252)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
parser.add_argument("--random-seed", action="store_true", help="Use random seed")
args = parser.parse_args()

config = load_config(args.config)
verbose = config.get("verbose", 0)
if verbose:
    logger.info(f"Loaded config: {config}")

symbol = config["stock_symbol"]
initial_balance = config.get("initial_balance", 10000)
start_date = config["start_date"]
end_date = config["end_date"]
raw_dir = config["raw_dir"]
processed_dir = config["processed_dir"]
sentiment_mode = config.get("sentiment_mode", "individual")
sentiment_source = config.get("sentiment_source", "finnhub_orig")
algo_name = config.get("algo", "PPO")
use_lstm = config.get("use_lstm", False)
lstm_window = config.get("lstm_window", 32)
lstm_hidden_size = config.get("lstm_hidden_size", 64)
train_test_split = config.get("train_test_split", 0.7)  # ← NUEVO
replicates = config.get("replicates", 1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# === SEED HANDLING ===
config_seed = config.get("seed")
entropy_str = f"{time.time()}_{os.getpid()}_{random.getrandbits(64)}"
global_entropy_seed = int(hashlib.sha256(entropy_str.encode()).hexdigest(), 16) % (2**32)

if args.random_seed:
    seed = random.randint(0, 999999)
    logger.info(f"CLI override → Using RANDOM seed: {seed}")
elif config_seed is None:
    seed = random.randint(0, 999999)
    logger.info(f"Config seed is null → Using RANDOM seed: {seed}")
else:
    seed = config_seed
    logger.info(f"Using FIXED seed from config: {seed}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)

# === DATA PATH ===
raw_csv = os.path.join(raw_dir, f"{symbol}_raw.csv")
processed_csv = os.path.join(processed_dir, f"{symbol}_sentiment_{sentiment_source if sentiment_mode == 'individual' else 'combined'}.csv")
data_path = processed_csv if os.path.exists(processed_csv) else raw_csv
logger.info(f"Using data: {data_path}")

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)
df.name = symbol
logger.info(f"Filtered data: {len(df)} rows from {df.index.min()} to {df.index.max()}")

# === TRAIN / TEST SPLIT (CONFIGURABLE) ===
split_idx = int(train_test_split * len(df))
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

logger.info(f"TRAIN: {len(train_df)} days ({train_test_split*100:.0f}%) | TEST: {len(test_df)} days ({(1-train_test_split)*100:.0f}%)")

# === SIMULATE FUNCTION (GLOBAL) ===
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

    Notes
    -----
    * The environment is **re‑created** for every call → isolates train / test.
    * No early termination: the loop runs over **all rows** of ``df``.
    * Uses ``df.iloc[step]`` → safe even if the index is non‑sequential.
    """
    logger.info(
        f"simulate() → {len(df)} rows | "
        f"Date range: {df.index[0].date()} → {df.index[-1].date()} | "
        f"Sentiment={'ON' if use_sentiment else 'OFF'}"
    )

    # ------------------------------------------------------------------ #
    # 1. Build a fresh environment for the given slice
    # ------------------------------------------------------------------ #
    env = TradingEnv(
        df=df,
        use_sentiment=use_sentiment,
        initial_balance=initial_balance,
        window_size=window_size,
    )
    obs, _ = env.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # 2. Prepare result container (keeps original Date index)
    # ------------------------------------------------------------------ #
    sim_df = pd.DataFrame(
        index=df.index,
        columns=["net_worth", "action"],
        dtype=float,
    )

    # ------------------------------------------------------------------ #
    # 3. Episode loop
    # ------------------------------------------------------------------ #
    for step in range(len(df)):
        # ---- current market price (close of *this* day) ----------------
        current_price = float(df.iloc[step]["close"])

        # ---- portfolio value before the agent acts --------------------
        net_worth = env.balance + env.shares_held * current_price

        # ---- agent decides --------------------------------------------
        action, _ = model.predict(obs, deterministic=False)
        action = int(action.item()) if hasattr(action, "item") else int(action)

        # ---- record state ---------------------------------------------
        sim_df.loc[df.index[step], "net_worth"] = net_worth
        sim_df.loc[df.index[step], "action"] = action

        # ---- environment advances to the *next* day -------------------
        obs, reward, terminated, truncated, _ = env.step(action)

        # (no early break – we always want the full slice)

    logger.info(f"Simulation finished → {len(sim_df)} rows")
    return sim_df
    
# === HYPERPARAMETERS ===
base_kwargs = {
    "learning_rate": config.get("ppo_lr", 0.0001),
    "gamma": config.get("ppo_gamma", 0.99),
    "seed": seed,
    "device": device,
    "verbose": verbose
}
ppo_specific = {
    "n_steps": lstm_window if use_lstm else config.get("ppo_n_steps", 256),
    "batch_size": lstm_window if use_lstm else config.get("ppo_batch_size", 128),
    "gae_lambda": config.get("ppo_gae_lambda", 0.95),
    "clip_range": config.get("ppo_clip_range", 0.1),
    "ent_coef": config.get("ppo_ent_coef", 0.001)
}
algo_kwargs = {**base_kwargs, **ppo_specific}


# === SINGLE RUN (replicates == 1) ===
if replicates == 1:
    window_size = lstm_window if use_lstm else 1
    
    # Train on train_df
    env_with = TradingEnv(train_df, use_sentiment=True, initial_balance=initial_balance, window_size=window_size)
    env_without = TradingEnv(train_df, use_sentiment=False, initial_balance=initial_balance, window_size=window_size)
    vec_env_with = DummyVecEnv([lambda: env_with])
    vec_env_without = DummyVecEnv([lambda: env_without])
    
    if use_lstm:
        policy_kwargs = dict(
            features_extractor_class=CustomLstmPolicy,
            features_extractor_kwargs=dict(lstm_hidden_size=lstm_hidden_size, n_lstm_layers=1),
            net_arch=[]
        )
        model_with = PPO("MlpPolicy", vec_env_with, policy_kwargs=policy_kwargs, **algo_kwargs)
        model_without = PPO("MlpPolicy", vec_env_without, policy_kwargs=policy_kwargs, **algo_kwargs)
    else:
        model_with = PPO("MlpPolicy", vec_env_with, **algo_kwargs)
        model_without = PPO("MlpPolicy", vec_env_without, **algo_kwargs)
    
    logger.info(f"Training {algo_name} {'+ LSTM' if use_lstm else ''} with sentiment")
    model_with.learn(total_timesteps=config["timesteps"])
    model_with.save(f"models/model_{'lstm' if use_lstm else 'mlp'}_with_sentiment")
    logger.info(f"Training {algo_name} {'+ LSTM' if use_lstm else ''} without sentiment")
    model_without.learn(total_timesteps=config["timesteps"])
    model_without.save(f"models/model_{'lstm' if use_lstm else 'mlp'}_without_sentiment")
  
    # Evaluate on test_df
    sim_with    = simulate(model_with,    True,  test_df, initial_balance, window_size, seed)
    sim_without = simulate(model_without, False, test_df, initial_balance, window_size, seed)

    sharpe_with = calculate_sharpe_ratio(sim_with['net_worth'].values)
    sharpe_without = calculate_sharpe_ratio(sim_without['net_worth'].values)

    plot_results(
        test_df, sim_with['net_worth'], sim_with['action'],
        sim_without['net_worth'], sim_without['action'],
        sharpe_with, sharpe_without, symbol, seed, use_lstm
    )
# === REPLICATION MODE ===
else:
    logger.info(f"REPLICATION MODE: {replicates} runs")
    os.makedirs("results/replicates", exist_ok=True)
    all_sharpe_with = []
    all_sharpe_without = []

    for rep in range(replicates):
        rep_entropy = f"{time.time()}_{os.getpid()}_{rep}_{global_entropy_seed}".encode()
        rep_seed = int(hashlib.sha256(rep_entropy).hexdigest(), 16) % (2**32)
        logger.info(f"--- Replication {rep+1}/{replicates} | Seed: {rep_seed} ---")
        random.seed(rep_seed)
        np.random.seed(rep_seed)
        torch.manual_seed(rep_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(rep_seed)

        window_size = lstm_window if use_lstm else 1
        
        # Train on train_df
        env_with = TradingEnv(train_df, True, initial_balance, window_size)
        env_without = TradingEnv(train_df, False, initial_balance, window_size)
        vec_env_with = DummyVecEnv([lambda: env_with])
        vec_env_without = DummyVecEnv([lambda: env_without])

        if use_lstm:
            policy_kwargs = dict(
                features_extractor_class=CustomLstmPolicy,
                features_extractor_kwargs=dict(lstm_hidden_size=lstm_hidden_size, n_lstm_layers=1),
                net_arch=[]
            )
            model_with = PPO("MlpPolicy", vec_env_with, policy_kwargs=policy_kwargs, **algo_kwargs)
            model_without = PPO("MlpPolicy", vec_env_without, policy_kwargs=policy_kwargs, **algo_kwargs)
        else:
            model_with = PPO("MlpPolicy", vec_env_with, **algo_kwargs)
            model_without = PPO("MlpPolicy", vec_env_without, **algo_kwargs)

        model_with.learn(total_timesteps=config["timesteps"])
        model_without.learn(total_timesteps=config["timesteps"])
        
        # Evaluate on test_df
        sim_with    = simulate(model_with,    True,  test_df, initial_balance, window_size, seed)
        sim_without = simulate(model_without, False, test_df, initial_balance, window_size, seed)

        sharpe_with = calculate_sharpe_ratio(sim_with['net_worth'].values)
        sharpe_without = calculate_sharpe_ratio(sim_without['net_worth'].values)
        all_sharpe_with.append(sharpe_with)
        all_sharpe_without.append(sharpe_without)

        rep_csv = f"results/replicates/{symbol}_rep_{rep+1:03d}_seed_{rep_seed}.csv"
        pd.DataFrame({
            "Date": test_df.index,
            "Net_Worth_With": sim_with['net_worth'],
            "Net_Worth_Without": sim_without['net_worth']
        }).to_csv(rep_csv, index=False)

    logger.info(f"TEST MEAN Sharpe (With): {np.mean(all_sharpe_with):.4f} ± {np.std(all_sharpe_with):.4f}")
    logger.info(f"TEST MEAN Sharpe (Without): {np.mean(all_sharpe_without):.4f} ± {np.std(all_sharpe_without):.4f}")