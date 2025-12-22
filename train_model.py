# train_model.py
"""
Simple Reinforcement Learning (RL) trading script for educational purposes.

What this script does (in simple terms):
1. Loads historical stock data (prices + optional sentiment from news).
2. Trains two PPO agents:
   - One that uses price data + sentiment
   - One that uses only price data
3. Simulates trading day-by-day with both agents on the same data.
4. Calculates the Sharpe Ratio (risk-adjusted return).
5. Plots:
   - Stock price + buy/sell signals (with sentiment agent)
   - Portfolio net worth for both agents
6. Saves results as CSV and PNG.

- Code is linear and easy to follow in Spyder (all variables visible).
- Compares "with sentiment" vs "without sentiment".
- No complex features – focus on understanding RL basics.
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

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.trading_env import TradingEnv

# --------------------------------------------------------------------- #
# Logging configuration (shows info in console)
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
    Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML file (e.g., configs/config.yaml).

    Returns:
        Dictionary with all configuration parameters.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise

def calculate_sharpe_ratio(net_worth: np.ndarray) -> float:
    """
    Calculate the annualized Sharpe Ratio (risk-free rate = 0).

    The Sharpe Ratio measures return per unit of risk.
    Higher = better (more profit with less volatility).

    Args:
        net_worth: Array of portfolio values over time.

    Returns:
        Sharpe Ratio (float). Returns 0.0 if not enough data or zero volatility.
    """
    if len(net_worth) < 2:
        logger.warning("Not enough data to calculate Sharpe Ratio")
        return 0.0

    returns = np.diff(net_worth) / net_worth[:-1]  # Daily returns
    if np.std(returns) == 0:
        logger.warning("Zero volatility → Sharpe Ratio undefined")
        return 0.0

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # 252 trading days/year
    return float(sharpe)

def plot_results(
    df: pd.DataFrame,
    net_worth_with: pd.Series,
    actions_with: pd.Series,
    net_worth_without: pd.Series,
    actions_without: pd.Series,
    sharpe_with: float,
    sharpe_without: float,
    symbol: str,
) -> None:
    """
    Create two plots:
    1. Stock price + buy/sell signals (only with sentiment agent)
    2. Net worth comparison for both agents

    Saves the figure to results/ and shows it (Spyder compatible).
    """
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # --- Plot 1: Price + Actions (with sentiment) ---
    ax1.plot(df.index, df["close"], label="Close Price", color="steelblue", linewidth=1.5)
    ax1.set_title(f"{symbol} Price and Trading Signals (With Sentiment)")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)

    # Buy/Sell markers
    buy_dates = df.index[actions_with == 1]
    sell_dates = df.index[actions_with == 2]
    ax1.scatter(buy_dates, df.loc[buy_dates, "close"], marker="^", color="green", s=80, label="Buy")
    ax1.scatter(sell_dates, df.loc[sell_dates, "close"], marker="v", color="red", s=80, label="Sell")
    ax1.legend()

    # --- Plot 2: Net Worth comparison ---
    ax2.plot(df.index, net_worth_with,
             label=f"With Sentiment (Sharpe: {sharpe_with:.2f})", color="purple", linewidth=2)
    ax2.plot(df.index, net_worth_without,
             label=f"Without Sentiment (Sharpe: {sharpe_without:.2f})", color="orange", linestyle="--", linewidth=2)
    ax2.set_title("Portfolio Net Worth Comparison")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Net Worth ($)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save and show
    os.makedirs("results", exist_ok=True)
    plot_path = f"results/{symbol.lower()}_trading_results_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()  # Visible in Spyder
    logger.info(f"Plot saved to {plot_path}")

# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train simple RL trading agents")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# --------------------------------------------------------------------- #
# Load configuration
# --------------------------------------------------------------------- #
config = load_config(args.config)

symbol = config["stock_symbol"]
initial_balance = config.get("initial_balance", 10000)

# --------------------------------------------------------------------- #
# Load and filter data
# --------------------------------------------------------------------- #
raw_csv = os.path.join(config["raw_dir"], f"{symbol}_raw.csv")
processed_csv = os.path.join(config["processed_dir"], f"{symbol}_sentiment_{config.get('sentiment_source', 'finnhub')}.csv")
data_path = processed_csv if os.path.exists(processed_csv) else raw_csv
logger.info(f"Loading data from {data_path}")

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= config["start_date"]) & (df["Date"] <= config["end_date"])].copy()
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)

logger.info(f"Data loaded: {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")

# --------------------------------------------------------------------- #
# Create environments
# --------------------------------------------------------------------- #
env_with = lambda: TradingEnv(df, use_sentiment=True, initial_balance=initial_balance)
env_without = lambda: TradingEnv(df, use_sentiment=False, initial_balance=initial_balance)

vec_env_with = make_vec_env(env_with, n_envs=1)
vec_env_without = make_vec_env(env_without, n_envs=1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --------------------------------------------------------------------- #
# Train PPO agents
# --------------------------------------------------------------------- #
ppo_kwargs = {
    "learning_rate": config.get("ppo_lr", 0.0003),
    "n_steps": config.get("ppo_n_steps", 2048),
    "batch_size": config.get("ppo_batch_size", 64),
    "gamma": config.get("ppo_gamma", 0.99),
    "gae_lambda": config.get("ppo_gae_lambda", 0.95),
    "clip_range": config.get("ppo_clip_range", 0.2),
    "ent_coef": 0.01,
    "verbose": 1,
    "device": device,
}

logger.info("Training agent WITH sentiment...")
model_with = PPO("MlpPolicy", vec_env_with, **ppo_kwargs)
model_with.learn(total_timesteps=config["timesteps"])
model_with.save("models/trading_model_with_sentiment")

logger.info("Training agent WITHOUT sentiment...")
model_without = PPO("MlpPolicy", vec_env_without, **ppo_kwargs)
model_without.learn(total_timesteps=config["timesteps"])
model_without.save("models/trading_model_without_sentiment")

# --------------------------------------------------------------------- #
# Simulation (run agents on data)
# --------------------------------------------------------------------- #
simulation_with = pd.DataFrame(index=df.index, columns=["net_worth", "action"])
simulation_without = pd.DataFrame(index=df.index, columns=["net_worth", "action"])

def run_simulation(model, use_sentiment: bool, sim_df: pd.DataFrame):
    """Run one agent on the full dataset and fill sim_df."""
    env = TradingEnv(df, use_sentiment=use_sentiment, initial_balance=initial_balance)
    obs = env.reset()[0]

    for step in range(len(df)):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item())
        obs, reward, done, truncated, info = env.step(action)

        price = df.iloc[step]["close"]
        net_worth = env.balance + env.shares_held * price
        date = df.index[step]

        sim_df.loc[date, "net_worth"] = net_worth
        sim_df.loc[date, "action"] = action

        if done or truncated:
            # Fill remaining days with last values (rare case)
            for remaining_date in df.index[step + 1:]:
                sim_df.loc[remaining_date, "net_worth"] = net_worth
                sim_df.loc[remaining_date, "action"] = 0
            break

    return sim_df

logger.info("Simulating agent WITH sentiment...")
simulation_with = run_simulation(model_with, True, simulation_with)

logger.info("Simulating agent WITHOUT sentiment...")
simulation_without = run_simulation(model_without, False, simulation_without)

# --------------------------------------------------------------------- #
# Metrics and results
# --------------------------------------------------------------------- #
sharpe_with = calculate_sharpe_ratio(simulation_with["net_worth"].values)
sharpe_without = calculate_sharpe_ratio(simulation_without["net_worth"].values)

logger.info(f"Sharpe Ratio (With Sentiment): {sharpe_with:.4f}")
logger.info(f"Sharpe Ratio (Without Sentiment): {sharpe_without:.4f}")

# Save CSV
results_df = pd.DataFrame({
    "Date": df.index,
    "Net_Worth_With_Sentiment": simulation_with["net_worth"],
    "Actions_With_Sentiment": simulation_with["action"],
    "Net_Worth_Without_Sentiment": simulation_without["net_worth"],
    "Actions_Without_Sentiment": simulation_without["action"],
})
os.makedirs("results", exist_ok=True)
results_df.to_csv(f"results/{symbol.lower()}_trading_results.csv", index=False)
logger.info("Results saved to CSV")

# Plot
plot_results(
    df,
    simulation_with["net_worth"],
    simulation_with["action"],
    simulation_without["net_worth"],
    simulation_without["action"],
    sharpe_with,
    sharpe_without,
    symbol
)

# --------------------------------------------------------------------- #
# Variables visible in Spyder
# --------------------------------------------------------------------- #
# df
# model_with, model_without
# simulation_with, simulation_without
# sharpe_with, sharpe_without