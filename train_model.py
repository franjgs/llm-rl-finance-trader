# train_model.py
# Run in Spyder → df, simulation_df_with_sentiment, simulation_df_without_sentiment visible
# Final plot: results/<symbol>_trading_results_comparison_<source>_seed_<seed>.png
"""
Train two PPO agents (with / without news sentiment) on a stock trading environment.
Key features:
- All parameters (including `initial_balance` and `replicates`) are read from `configs/config.yaml`.
- `replicates == 1` → identical behavior to the original script.
- `replicates > 1` → runs independent replications and saves a statistical summary.
- Profit percentages are computed using `initial_balance` from the config.
- Full English documentation and logging.
"""
import argparse
import yaml
import pandas as pd
import logging
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.trading_env import TradingEnv

# --------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------- #
def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file.
    Args:
        config_path: Path to the YAML file.
    Returns:
        Dictionary with the parsed configuration.
    Raises:
        yaml.YAMLError: If the YAML is malformed.
        Exception: For any other file-related error.
    """
    try:
        with open(config_path, "r") as f:
            content = f.read()
            logger.info(f"YAML content:\n{content}")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise


def calculate_sharpe_ratio(net_worth: np.ndarray) -> float:
    """
    Compute the annualized Sharpe ratio (risk-free rate = 0).
    Args:
        net_worth: Time series of portfolio net worth.
    Returns:
        Sharpe ratio. Returns 0.0 if the series is too short or has zero volatility.
    """
    if len(net_worth) < 2:
        logger.warning("Net worth series too short to calculate Sharpe Ratio")
        return 0.0
    returns = np.diff(net_worth) / net_worth[:-1]  # daily returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        logger.warning("Zero volatility in returns, Sharpe Ratio undefined")
        return 0.0
    sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 252 trading days
    return sharpe_ratio


def plot_results(
    df: pd.DataFrame,
    net_worth_with: pd.Series,
    actions_with: pd.Series,
    net_worth_without: pd.Series,
    actions_without: pd.Series,
    sharpe_with: float,
    sharpe_without: float,
    seed_used: int,
) -> None:
    """
    Plot price + buy/sell actions (with sentiment) and net-worth curves for both agents.
    Args:
        df: Stock data (index = Date, must contain `close` column).
        net_worth_with: Net-worth series for the sentiment agent.
        actions_with: Action series for the sentiment agent (0=hold, 1=buy, 2=sell).
        net_worth_without: Net-worth series for the no-sentiment agent.
        actions_without: Action series for the no-sentiment agent.
        sharpe_with: Sharpe ratio of the sentiment agent.
        sharpe_without: Sharpe ratio of the no-sentiment agent.
        seed_used: Random seed used for the run (shown in titles).
    """
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top panel: price + actions (with sentiment)
    ax1.plot(df.index, df["close"], label="Close Price", color="blue")
    ax1.set_title(f"{symbol} Close Price and Trading Actions (With Sentiment)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    buy_points = df[actions_with >= 0.5]   # position > 0.5 → buy
    sell_points = df[actions_with <= -0.5]  # position < -0.5 → sell
    ax1.scatter(buy_points.index, buy_points["close"], color="green", marker="^", label="Buy")
    ax1.scatter(sell_points.index, sell_points["close"], color="red", marker="v", label="Sell")
    ax1.legend()

    # Bottom panel: net-worth comparison
    ax2.plot(df.index, net_worth_with,
             label=f"Net Worth (With Sentiment, Sharpe: {sharpe_with:.2f})", color="purple")
    ax2.plot(df.index, net_worth_without,
             label=f"Net Worth (Without Sentiment, Sharpe: {sharpe_without:.2f})", color="orange", linestyle="--")
    ax2.set_title(f"Portfolio Net Worth Comparison (Seed: {seed_used})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Net Worth ($)")
    min_nw = min(net_worth_with.min(), net_worth_without.min())
    max_nw = max(net_worth_with.max(), net_worth_without.max())
    if np.std(net_worth_with) < 1e-6 and np.std(net_worth_without) < 1e-6:
        ax2.set_ylim(min_nw - 100, max_nw + 100)
    else:
        ax2.set_ylim(min_nw * 0.95, max_nw * 1.05)
    ax2.legend()
    logger.info(f"Net Worth (With): min={net_worth_with.min():.2f}, max={net_worth_with.max():.2f}")
    logger.info(f"Net Worth (Without): min={net_worth_without.min():.2f}, max={net_worth_without.max():.2f}")

    plt.tight_layout()
    plt.show()
    os.makedirs("results", exist_ok=True)
    plot_filename = f"results/{symbol}_trading_results_comparison_{source}_seed_{seed_used}.png"
    plt.savefig(plot_filename)
    plt.close(fig)
    logger.info(f"Saved plot to {plot_filename}")


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train RL trading model with/without sentiment")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
parser.add_argument("--random-seed", action="store_true", help="Use random seed instead of config seed")
args = parser.parse_args()


# --------------------------------------------------------------------- #
# Load configuration
# --------------------------------------------------------------------- #
config = load_config(args.config)
logger.info(f"Loaded config: {config}")
verbose = config["verbose"]
symbol = config["stock_symbol"]
mode = config.get("sentiment_mode", "individual").lower()
source = "combined" if mode == "combined" else config.get("sentiment_source", "finnhub").lower()
replicates = config.get("replicates", 1)
initial_balance = config.get("initial_balance", 10000)  # Configurable starting capital


# --------------------------------------------------------------------- #
# Seed handling with entropy injection
# --------------------------------------------------------------------- #
config_seed = config.get("seed")
# Inject entropy to avoid identical seeds on rapid executions
random.seed(int(time.time() * 1e6) % 2**32)
np.random.seed(int(time.time() * 1e6) % 2**32)
torch.manual_seed(int(time.time() * 1e6) % 2**32)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(int(time.time() * 1e6) % 2**32)

if args.random_seed:
    seed = random.randint(0, 999999)
    logger.info(f"CLI override → Using RANDOM seed: {seed}")
elif config_seed is None:
    seed = random.randint(0, 999999)
    logger.info(f"Config seed is null → Using RANDOM seed: {seed}")
else:
    seed = config_seed
    logger.info(f"Using FIXED seed from config: {seed}")

# Apply final seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)


# --------------------------------------------------------------------- #
# Build file paths
# --------------------------------------------------------------------- #
raw_csv = os.path.join(config["raw_dir"], f"{symbol}_raw.csv")
processed_csv = os.path.join(config["processed_dir"], f"{symbol}_sentiment_{source}.csv")
data_path = processed_csv if os.path.exists(processed_csv) else raw_csv
logger.info(f"Using data: {data_path}")


# --------------------------------------------------------------------- #
# Load and filter price data
# --------------------------------------------------------------------- #
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= config["start_date"]) & (df["Date"] <= config["end_date"])].copy()
df = df.sort_values("Date").reset_index(drop=True)  # ← Limpio y secuencial
df.set_index("Date", inplace=True)  # Required by plot_results
df.name = symbol
logger.info(f"Filtered data: {len(df)} rows from {df.index.min()} to {df.index.max()}")


# --------------------------------------------------------------------- #
# PPO hyper-parameters
# --------------------------------------------------------------------- #
ppo_kwargs = {
    "learning_rate": config.get("ppo_lr", 0.0003),
    "clip_range": config.get("ppo_clip_range", 0.2),
    "n_steps": config.get("ppo_n_steps", 256),
    "batch_size": config.get("ppo_batch_size", 128),
    "gamma": config.get("ppo_gamma", 0.99),
    "gae_lambda": config.get("ppo_gae_lambda", 0.95),
    "seed": config.get("seed", 42),
    "ent_coef": 0.001,  # Prevent policy collapse
}
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")


# --------------------------------------------------------------------- #
# SINGLE RUN (replicates == 1) – Original behavior
# --------------------------------------------------------------------- #
if replicates == 1:
    # Vectorized environments
    vec_env_with = make_vec_env(lambda: TradingEnv(df.reset_index(), use_sentiment=True, initial_balance=initial_balance), n_envs=1, seed=seed)
    vec_env_without = make_vec_env(lambda: TradingEnv(df.reset_index(), use_sentiment=False, initial_balance=initial_balance), n_envs=1, seed=seed)

    # Train with sentiment
    model_with_sentiment = PPO("MlpPolicy", vec_env_with, verbose=verbose, device=device, **ppo_kwargs)
    logger.info(f"Training PPO with sentiment for {config['timesteps']} timesteps")
    model_with_sentiment.learn(total_timesteps=config["timesteps"])
    model_with_sentiment.save("models/trading_model_with_sentiment")

    # Train without sentiment
    model_without_sentiment = PPO("MlpPolicy", vec_env_without, verbose=verbose, device=device, **ppo_kwargs)
    logger.info(f"Training PPO without sentiment for {config['timesteps']} timesteps")
    model_without_sentiment.learn(total_timesteps=config["timesteps"])
    model_without_sentiment.save("models/trading_model_without_sentiment")

    # Simulation DataFrames
    simulation_df_with_sentiment = pd.DataFrame(index=df.index, columns=["net_worth", "action"])
    simulation_df_without_sentiment = pd.DataFrame(index=df.index, columns=["net_worth", "action"])

    # Simulate with sentiment
    env = TradingEnv(df.reset_index(), use_sentiment=True, initial_balance=initial_balance)
    obs, _ = env.reset(seed=seed)
    for step in range(len(df)):
        action, _ = model_with_sentiment.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        net_worth = env.net_worth
        date = df.index[step]
        simulation_df_with_sentiment.loc[date, "net_worth"] = net_worth
        simulation_df_with_sentiment.loc[date, "action"] = action
        if done or truncated:
            missing = df.index[step + 1:].tolist()
            logger.warning(f"Simulation (with) stopped early at step {step}, date={date}. Filling {len(missing)} dates.")
            for md in missing:
                simulation_df_with_sentiment.loc[md, "net_worth"] = net_worth
                simulation_df_with_sentiment.loc[md, "action"] = action
            break

    # Simulate without sentiment
    env = TradingEnv(df.reset_index(), use_sentiment=False, initial_balance=initial_balance)
    obs, _ = env.reset(seed=seed)
    for step in range(len(df)):
        action, _ = model_without_sentiment.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        net_worth = env.net_worth
        date = df.index[step]
        simulation_df_without_sentiment.loc[date, "net_worth"] = net_worth
        simulation_df_without_sentiment.loc[date, "action"] = action
        if done or truncated:
            missing = df.index[step + 1:].tolist()
            logger.warning(f"Simulation (without) stopped early at step {step}, date={date}. Filling {len(missing)} dates.")
            for md in missing:
                simulation_df_without_sentiment.loc[md, "net_worth"] = net_worth
                simulation_df_without_sentiment.loc[md, "action"] = action
            break

    # Verify alignment
    if not simulation_df_with_sentiment.index.equals(df.index):
        raise ValueError("Index misalignment in simulation_df_with_sentiment")
    if not simulation_df_without_sentiment.index.equals(df.index):
        raise ValueError("Index misalignment in simulation_df_without_sentiment")
    if simulation_df_with_sentiment["net_worth"].isna().any():
        raise ValueError("Missing net_worth values in simulation_df_with_sentiment")
    if simulation_df_without_sentiment["net_worth"].isna().any():
        raise ValueError("Missing net_worth values in simulation_df_without_sentiment")
    logger.info("Date indices and data aligned successfully")

    # Sharpe ratios
    sharpe_with_sentiment = calculate_sharpe_ratio(simulation_df_with_sentiment["net_worth"].values)
    sharpe_without_sentiment = calculate_sharpe_ratio(simulation_df_without_sentiment["net_worth"].values)
    logger.info(f"Sharpe Ratio (With Sentiment): {sharpe_with_sentiment:.4f}")
    logger.info(f"Sharpe Ratio (Without Sentiment): {sharpe_without_sentiment:.4f}")

    # Final profit percentages
    final_with = simulation_df_with_sentiment["net_worth"].iloc[-1]
    final_without = simulation_df_without_sentiment["net_worth"].iloc[-1]
    profit_with_pct = (final_with - initial_balance) / initial_balance * 100
    profit_without_pct = (final_without - initial_balance) / initial_balance * 100
    logger.info(f"Final Net Worth (With): ${final_with:,.2f} → {profit_with_pct:+.2f}%")
    logger.info(f"Final Net Worth (Without): ${final_without:,.2f} → {profit_without_pct:+.2f}%")

    # Action distribution
    logger.info(f"Actions (with): {Counter(simulation_df_with_sentiment['action'].round(2))}")
    logger.info(f"Actions (without): {Counter(simulation_df_without_sentiment['action'].round(2))}")

    # Save results CSV
    results_csv = f"results/{symbol}_trading_results_{source}_seed_{seed}.csv"
    results_df = pd.DataFrame({
        "Date": df.index,
        "Net_Worth_With_Sentiment": simulation_df_with_sentiment["net_worth"],
        "Net_Worth_Without_Sentiment": simulation_df_without_sentiment["net_worth"],
        "Actions_With_Sentiment": simulation_df_with_sentiment["action"],
        "Actions_Without_Sentiment": simulation_df_without_sentiment["action"],
    })
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved results → {results_csv}")

    # Plot
    plot_results(
        df,
        simulation_df_with_sentiment["net_worth"],
        simulation_df_with_sentiment["action"],
        simulation_df_without_sentiment["net_worth"],
        simulation_df_without_sentiment["action"],
        sharpe_with_sentiment,
        sharpe_without_sentiment,
        seed,
    )


# --------------------------------------------------------------------- #
# REPLICATION MODE (replicates > 1)
# --------------------------------------------------------------------- #
else:
    logger.info(f"REPLICATION MODE: Running {replicates} replications")
    os.makedirs("results/replicates", exist_ok=True)
    all_results = []

    for rep in range(replicates):
        rep_seed = random.randint(0, 999999)
        logger.info(f"--- Replication {rep+1}/{replicates} | Seed: {rep_seed} ---")

        # Re-seed
        random.seed(rep_seed)
        np.random.seed(rep_seed)
        torch.manual_seed(rep_seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(rep_seed)

        # Environments
        vec_env_with = make_vec_env(lambda: TradingEnv(df.reset_index(), use_sentiment=True, initial_balance=initial_balance), n_envs=1, seed=rep_seed)
        vec_env_without = make_vec_env(lambda: TradingEnv(df.reset_index(), use_sentiment=False, initial_balance=initial_balance), n_envs=1, seed=rep_seed)

        # Train (quiet)
        model_with = PPO("MlpPolicy", vec_env_with, verbose=verbose,
                         device="mps" if torch.backends.mps.is_available() else "cpu", **ppo_kwargs)
        model_with.learn(total_timesteps=config["timesteps"])
        model_without = PPO("MlpPolicy", vec_env_without, verbose=verbose,
                             device="mps" if torch.backends.mps.is_available() else "cpu", **ppo_kwargs)
        model_without.learn(total_timesteps=config["timesteps"])

        # Simulate with sentiment
        sim_with = pd.DataFrame(index=df.index, columns=["net_worth", "action"])
        env = TradingEnv(df.reset_index(), use_sentiment=True, initial_balance=initial_balance)
        obs, _ = env.reset(seed=rep_seed)
        for step in range(len(df)):
            action, _ = model_with.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            net_worth = env.net_worth
            date = df.index[step]
            sim_with.loc[date, "net_worth"] = net_worth
            sim_with.loc[date, "action"] = action
            if done or truncated:
                for md in df.index[step + 1:]:
                    sim_with.loc[md, "net_worth"] = net_worth
                    sim_with.loc[md, "action"] = action
                break

        # Simulate without sentiment
        sim_without = pd.DataFrame(index=df.index, columns=["net_worth", "action"])
        env = TradingEnv(df.reset_index(), use_sentiment=False, initial_balance=initial_balance)
        obs, _ = env.reset(seed=rep_seed)
        for step in range(len(df)):
            action, _ = model_without.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            net_worth = env.net_worth
            date = df.index[step]
            sim_without.loc[date, "net_worth"] = net_worth
            sim_without.loc[date, "action"] = action
            if done or truncated:
                for md in df.index[step + 1:]:
                    sim_without.loc[md, "net_worth"] = net_worth
                    sim_without.loc[md, "action"] = action
                break

        # Save per-replication CSV
        rep_csv = f"results/replicates/{symbol}_rep_{rep+1:03d}_{source}_seed_{rep_seed}.csv"
        pd.DataFrame({
            "Date": df.index,
            "Net_Worth_With": sim_with["net_worth"],
            "Net_Worth_Without": sim_without["net_worth"],
        }).to_csv(rep_csv, index=False)

        # Metrics
        final_with = sim_with["net_worth"].iloc[-1]
        final_without = sim_without["net_worth"].iloc[-1]
        sharpe_with = calculate_sharpe_ratio(sim_with["net_worth"].values)
        sharpe_without = calculate_sharpe_ratio(sim_without["net_worth"].values)
        all_results.append({
            "replication": rep + 1,
            "seed": rep_seed,
            "final_with": final_with,
            "final_without": final_without,
            "sharpe_with": sharpe_with,
            "sharpe_without": sharpe_without,
            "profit_with_%": (final_with - initial_balance) / initial_balance * 100,
            "profit_without_%": (final_without - initial_balance) / initial_balance * 100,
        })

    # Summary
    summary_df = pd.DataFrame(all_results)
    summary_path = f"results/summary_{symbol}_{source}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path}")
    logger.info(f"MEAN PROFIT WITH: {summary_df['profit_with_%'].mean():.2f}% ± {summary_df['profit_with_%'].std():.2f}%")
    logger.info(f"MEAN PROFIT WITHOUT: {summary_df['profit_without_%'].mean():.2f}% ± {summary_df['profit_without_%'].std():.2f}%")
    logger.info(f"WIN RATE (WITH > WITHOUT): {(summary_df['profit_with_%'] > summary_df['profit_without_%']).mean()*100:.1f}%")


# --------------------------------------------------------------------- #
# Variables visible in Spyder (only in single-run mode)
# --------------------------------------------------------------------- #
# df, simulation_df_with_sentiment, simulation_df_without_sentiment,
# model_with_sentiment, model_without_sentiment,
# sharpe_with_sentiment, sharpe_without_sentiment, seed