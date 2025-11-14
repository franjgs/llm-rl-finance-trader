#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_walk_forward.py
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
from src.gen_utils import load_config
from src.plot_utils import plot_results
from src.rl_utils import CustomLstmPolicy, CustomEvalMonitor
from src.metrics import log_daily_performance  # ← CORRECTED: use shared metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Reward amplification wrapper that preserves info (critical for EvalCallback) ---
from gymnasium import RewardWrapper


class InfoRewardScaler(RewardWrapper):
    """
    Amplifies reward magnitude while preserving the original ``info`` dictionary.
    Required so ``EvalCallback`` can access ``net_worth`` from the environment.
    """
    def __init__(self, env, scale: float = 100.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


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
commission = config.get("commission", 0.001)
raw_dir = config["raw_dir"]
processed_dir = config["processed_dir"]
sentiment_mode = config.get("sentiment_mode", "individual")
sentiment_source = config.get("sentiment_source", "finnhub_orig")
algo_name = config.get("algo", "PPO")
use_lstm = config.get("use_lstm", False)
lstm_window = config.get("lstm_window", 32)
lstm_hidden_size = config.get("lstm_hidden_size", 64)
timesteps = config["timesteps"]
# ADDED: Load incremental timesteps outside the loop for cleaner logic
incremental_timesteps = config.get("incremental_timesteps", 1000)
walk_forward = config.get("walk_forward", True)
prediction_horizon = config.get("prediction_horizon", 1)
tb_log_base_path = config.get("tensorboard_log_dir", "logs/tensorboard_walk_forward")

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu" # FIXED: Force CPU to avoid low-utilization GPU/MPS warning for MlpPolicy
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
if device == "mps":  # torch.backends.mps.is_available():
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

# === FILTER BY DATE RANGE FROM CONFIG ===
if "start_date" in config and config["start_date"]:
    start_date = pd.to_datetime(config["start_date"])
    df = df.loc[start_date:]
    logger.info(f"Filtered start_date: {start_date.date()}")
if "end_date" in config and config["end_date"]:
    end_date = pd.to_datetime(config["end_date"])
    df = df.loc[:end_date]
    logger.info(f"Filtered end_date: {end_date.date()}")
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
        commission=commission  # ← Pass commission to env
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
        # --- PREDICTION ---
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item()) if hasattr(action, "item") else int(action)
        
        # --- LOGS ---
        print(f"[SIM] Step {step} | Price: {current_price:,.2f} | "
              f"Action: {action} | Shares: {env.shares_held} | "
              f"Net Worth: {net_worth:,.2f}")
        
        sim_df.loc[df.index[step], "net_worth"] = net_worth
        sim_df.loc[df.index[step], "action"] = action
        obs, reward, terminated, truncated, _ = env.step(action)
    logger.info(f"Simulation finished → {len(sim_df)} rows")
    return sim_df


# =============================================================================
# 3. CALLBACKS
# =============================================================================
class RewardLogger(BaseCallback):
    """
    Logs training episode rewards and episode lengths.
    Compatible with Stable-Baselines3 monitor format.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info and "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])
                # Write explicitly for TensorBoard
                self.logger.record("train/episode_reward", ep["r"])
                self.logger.record("train/episode_length", ep["l"])
            if info and "net_worth" in info:
                self.logger.record("train/net_worth", info["net_worth"])
        return True


class EvalNetWorthLogger(BaseCallback):
    """
    Logs net worth step-by-step during evaluation episodes.
    Helps debugging agent behavior in EvalCallback.
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info and "net_worth" in info:
                self.logger.record("eval/net_worth_step", info["net_worth"])
        return True


# === POST-EVALUATION CALLBACK: LOG NET WORTH & REWARD ===
class NetWorthLogger(BaseCallback):
    """
    Logs the **final** portfolio net worth and mean reward after each evaluation episode.
    Designed for any ``prediction_horizon`` (1-day, 5-day, …).
    Uses the **last** entry of ``infos`` (``infos[-1]``) → state at the end of the episode.
    Pushes the following tags to TensorBoard:
    - ``eval/net_worth_final`` – portfolio value after the full horizon
    - ``eval/mean_reward`` – average daily reward over the horizon
    - ``eval/episode_length`` – number of days in the evaluation episode (== horizon)
    All docstrings and comments are in **English**.
    """
    def __init__(self, verbose: int = 0):
        """
        Parameters
        ----------
        verbose : int, optional
            Verbosity level (passed to ``BaseCallback``). Default = 0.
        """
        super().__init__(verbose)

    # Required by BaseCallback – we do not need it
    def _on_step(self) -> bool:
        """Return ``True`` to continue training."""
        return True

    # Called **once** after a complete evaluation episode
    def _on_eval_end(self) -> None:
        """
        Extract and log final metrics from the evaluation run.
        ``self.locals`` contains:
            - ``infos`` : list of info dicts (one per step)
            - ``mean_reward`` : average reward of the episode
            - ``episode_lengths``: length of the episode (in steps)
        """
        infos = self.locals.get("infos", [])
        if not infos:
            return

        # ----------------------------------------------------------------- #
        # 1. Final portfolio value (last step of the episode)
        # ----------------------------------------------------------------- #
        final_info = infos[-1]
        if "net_worth" in final_info:
            net_worth = final_info["net_worth"]
            self.logger.record("eval/net_worth_final", net_worth)

            # OPTIONAL: also print to console (visible even with DummyVecEnv)
            if self.verbose > 0:
                print(f"[EVAL] net_worth_final = {net_worth:,.2f}")

        # ----------------------------------------------------------------- #
        # 2. Mean reward over the whole horizon
        # ----------------------------------------------------------------- #
        if "mean_reward" in self.locals:
            mean_reward = self.locals["mean_reward"]
            self.logger.record("eval/mean_reward", mean_reward)

            if self.verbose > 0:
                print(f"[EVAL] mean_reward = {mean_reward:.6f}")

        # ----------------------------------------------------------------- #
        # 3. Episode length (number of days predicted)
        # ----------------------------------------------------------------- #
        if "episode_lengths" in self.locals:
            ep_len = self.locals["episode_lengths"][0]
            self.logger.record("eval/episode_length", ep_len)

            if self.verbose > 0:
                print(f"[EVAL] episode_length = {ep_len}")

        # ----------------------------------------------------------------- #
        # 4. Force TensorBoard flush so the scalars appear instantly
        # ----------------------------------------------------------------- #
        self.logger.dump(self.num_timesteps)

# =============================================================================
# 4. ALGORITHM SETUP
# =============================================================================
common_kwargs = {
    "seed": base_seed,
    "device": device,
    "verbose": 0  # TensorBoard handles logging
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
if sentiment_mode == "with":
    modes = [True]
elif sentiment_mode == "without":
    modes = [False]
elif sentiment_mode == "both":
    modes = [True, False]
else:
    raise ValueError("sentiment_mode must be: with, without, both")
all_results = {True: [], False: []}
min_train_days = config.get("min_train_days", 100)
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

        def make_train_env():
            """
            Factory for the **training** environment.
            Returns a fresh ``TradingEnv`` wrapped with ``InfoRewardScaler``.
            """
            env = TradingEnv(
                df=train_slice,
                use_sentiment=use_sentiment,
                initial_balance=initial_balance,
                window_size=window_size,
                commission=commission
            )
            return InfoRewardScaler(env, scale=1)

        def make_eval_env():
            """
            Factory for the **evaluation** environment.
            Returns a fresh ``TradingEnv`` (with the prediction horizon) wrapped
            with ``InfoRewardScaler`` and ``CustomEvalMonitor``.
            """
            env = TradingEnv(
                df=test_slice,
                use_sentiment=use_sentiment,
                initial_balance=initial_balance,
                window_size=window_size,
                commission=commission,
                prediction_horizon=prediction_horizon
            )
            env = InfoRewardScaler(env, scale=1)
            return CustomEvalMonitor(env)

        vec_train = DummyVecEnv([make_train_env])
        vec_eval = DummyVecEnv([make_eval_env])

        # -----------------------------------------------------------------
        # DYNAMIC PPO HYPER-PARAMETERS (re-computed every day)
        # -----------------------------------------------------------------
        # PPO: n_steps must be divisible by batch_size, and batch_size must be >= 2.
        max_n_steps = config.get("ppo_n_steps", 64)
        max_batch = config.get("ppo_batch_size", 32)
        
        # 1. Calculate the initial n_steps, ensuring a minimum of 8 for stability
        initial_n_steps = min(max_n_steps, len(train_slice) // 2)
        n_steps = max(8, initial_n_steps) # Use 8 as a safe minimum (divisible by 2 and 4)
        
        # 2. Determine the desired batch size (e.g., 1/4 of n_steps, min 2)
        # We calculate the desired size and ensure it's at least 2
        desired_batch_size = n_steps // 4
        batch_size = max(2, desired_batch_size)
        
        # 3. CRITICAL STEP: Adjust n_steps down to be a multiple of the final batch_size.
        # This eliminates the Stable-Baselines3 UserWarning.
        if n_steps % batch_size != 0:
            # Round n_steps down to the nearest multiple of batch_size
            n_steps = (n_steps // batch_size) * batch_size
            # Ensure the new n_steps is still at least batch_size (which is >= 2)
            n_steps = max(batch_size, n_steps) 
        
        # Final Sanity Check (Should never trigger)
        if n_steps < batch_size or batch_size < 2:
            batch_size = 2
            n_steps = 2 # Worst case scenario
            
        # Copy static kwargs and override the dynamic ones
        algo_kwargs_dynamic = algo_kwargs.copy()
        algo_kwargs_dynamic.update({
            "n_steps": n_steps,
            "batch_size": batch_size,
        })
        logger.info(
            f"Dynamic PPO → n_steps={n_steps}, batch_size={batch_size}, "
            f"train_days={len(train_slice)}, "
            f"rollouts≈{incremental_timesteps//n_steps}"
        )
        # -----------------------------------------------------------------

        # --- WARM START: Load previous best model ---
        prev_date = (pd.Timestamp(pred_date) - pd.Timedelta(days=1)).date()
        model_path = f"models/best_walk_forward/{symbol}_{prev_date}_{label}/best_model.zip"
        if config.get("warm_start", False) and pred_idx > start_idx and os.path.exists(model_path):
            logger.info(f"Warm start: loading model from {model_path}")
            model = AlgoClass.load(model_path, env=vec_train, **algo_kwargs_dynamic)
            train_steps = incremental_timesteps
        else:
            # --- Model creation ---
            policy_kwargs = None
            if use_lstm and algo_name == "PPO":
                policy_kwargs = dict(
                    features_extractor_class=CustomLstmPolicy,
                    features_extractor_kwargs=dict(
                        lstm_hidden_size=lstm_hidden_size,
                        n_lstm_layers=1
                    ),
                    net_arch=[],
                )
            model = AlgoClass(
                "MlpPolicy", vec_train,
                normalize_advantage=True,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"{tb_log_base_path}/{symbol}_{pred_date}_{label}",
                **algo_kwargs_dynamic          # use dynamic values
            )
            train_steps = timesteps
        # --- Callbacks ---
        reward_logger = RewardLogger()  # Logs episode rewards for learning curves

        eval_networth_logger = EvalNetWorthLogger()
        
        eval_callback = None
        if prediction_horizon > 1:
            eval_freq = max(1, len(train_slice) // 5)
            eval_callback = EvalCallback(
                eval_env=vec_eval,
                best_model_save_path=f"models/best_walk_forward/{symbol}_{pred_date}_{label}",
                log_path=f"logs/eval_walk_forward/{symbol}_{pred_date}_{label}",
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=1,
                verbose=1,
                callback_after_eval=NetWorthLogger(verbose=1)
            )
        else:
            logger.info(f"prediction_horizon=1 → Skipping EvalCallback (no evaluation needed)")
        
        # --- CALLBACKS ---
        callbacks = [reward_logger, eval_networth_logger]
        if eval_callback is not None:
            callbacks.append(eval_callback)

        # --- Train ---
        model.learn(
            total_timesteps=train_steps,
            callback=callbacks,
            tb_log_name=f"{label}_sentiment",
            reset_num_timesteps=False
        )
        # --- Simulate ---
        sim = simulate(model, use_sentiment, test_slice, initial_balance, window_size, day_seed)
        final_nw = sim["net_worth"].iloc[-1]
        # --- Buy & Hold value for this day ---
        test_price_start = test_slice["close"].iloc[0]
        test_price_end = test_slice["close"].iloc[-1]
        shares_bh = initial_balance / test_price_start
        bh_value = shares_bh * test_price_end
        
        # --- Log performance ---
        log_daily_performance(pred_date, final_nw, test_slice, initial_balance, bh_value)

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
# === PREPARE BUY & HOLD (CORRECTED: starts on first training day) ===
first_train_date = df.index[min_train_days]  # Day 100: when agent starts learning
start_price = df.loc[first_train_date, "close"]
shares = initial_balance / start_price
buy_and_hold = shares * df["close"].loc[first_train_date:]
buy_and_hold.name = "Buy & Hold"
# === FINAL PLOT ===
plot_results(
    walk_forward=True,
    symbol=symbol,
    seed=base_seed,
    use_lstm=use_lstm,
    net_worth_with_mean=net_worth_with_mean,
    net_worth_without_mean=net_worth_without_mean,
    buy_and_hold=buy_and_hold,
    initial_balance=initial_balance
)
logger.info("Walk-forward training completed.")