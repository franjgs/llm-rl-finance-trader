import argparse
import yaml
import pandas as pd
import logging
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.trading_env import TradingEnv
import os
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.

    Raises:
        yaml.YAMLError: If the YAML file is invalid.
        Exception: For other file loading errors.
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            logger.info(f"YAML content:\n{content}")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

def plot_results(df, net_worth, actions):
    """Generate and save plots for stock prices, trading actions, and portfolio net worth.

    Args:
        df (pd.DataFrame): DataFrame with stock data (Date, close, etc.).
        net_worth (list): List of portfolio net worth values.
        actions (list): List of trading actions (0: hold, 1: buy, 2: sell).
    """
    # Ensure lengths match
    min_length = min(len(df['Date']), len(net_worth), len(actions))
    if min_length < len(df['Date']):
        logger.warning(f"Truncating df['Date'] from {len(df['Date'])} to {min_length} to match net_worth and actions")
        df = df.iloc[:min_length]
    if min_length < len(net_worth):
        logger.warning(f"Truncating net_worth from {len(net_worth)} to {min_length}")
        net_worth = net_worth[:min_length]
    if min_length < len(actions):
        logger.warning(f"Truncating actions from {len(actions)} to {min_length}")
        actions = actions[:min_length]

    # Convert 'Date' to datetime to avoid Matplotlib warnings
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clear Matplotlib state
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot stock prices and trading actions
    ax1.plot(df['Date'], df['close'], label='Close Price', color='blue')
    ax1.set_title('AAPL Close Price and Trading Actions')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    
    buy_points = df.iloc[[i for i, a in enumerate(actions) if a == 1]]
    sell_points = df.iloc[[i for i, a in enumerate(actions) if a == 2]]
    ax1.scatter(buy_points['Date'], buy_points['close'], color='green', marker='^', label='Buy')
    ax1.scatter(sell_points['Date'], sell_points['close'], color='red', marker='v', label='Sell')
    ax1.legend()
    
    # Plot portfolio net worth
    if not np.all(np.isfinite(net_worth)):
        logger.warning("net_worth contains invalid values (NaN or inf)")
        net_worth = [10000 if not np.isfinite(x) else x for x in net_worth]
    ax2.plot(df['Date'], net_worth, label='Portfolio Net Worth', color='purple')
    ax2.set_title('Portfolio Net Worth')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    if np.std(net_worth) < 1e-6:
        logger.warning("net_worth is constant, adjusting y-axis")
        ax2.set_ylim(min(net_worth) - 100, max(net_worth) + 100)
    else:
        ax2.set_ylim(min(net_worth) * 0.95, max(net_worth) * 1.05)
    ax2.legend()
    logger.info(f"Net Worth plot: min={min(net_worth)}, max={max(net_worth)}, std={np.std(net_worth)}")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/aapl_trading_results.png')
    plt.show()
    plt.close(fig)

    # Save separate net worth plot for debugging
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Date'], net_worth, label='Portfolio Net Worth', color='purple')
    ax.set_title('Portfolio Net Worth (Separate)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Net Worth ($)')
    if np.std(net_worth) < 1e-6:
        ax.set_ylim(min(net_worth) - 100, max(net_worth) + 100)
    else:
        ax.set_ylim(min(net_worth) * 0.95, max(net_worth) * 1.05)
    ax.legend()
    plt.savefig('results/aapl_net_worth_separate.png')
    plt.close(fig)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL trading model")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger.info(f"Loaded config: {config}")

# Load data
df = pd.read_csv(config['data_path'])
logger.info(f"Loaded data from {config['data_path']} with {len(df)} rows")

# Initialize environment
env = lambda: TradingEnv(df)
vec_env = make_vec_env(env, n_envs=1)
logger.info("Environment initialized")

# Set device (MPS or CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Train PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, device=device, learning_rate=0.0001, clip_range=0.1)
logger.info(f"Training PPO for {config['timesteps']} timesteps")
model.learn(total_timesteps=config['timesteps'])

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/trading_model")
logger.info("Model saved to models/trading_model")

# Simulate trading for visualization
env = TradingEnv(df)
obs = env.reset()[0]
net_worth = []
actions = []
done = False
while not done:
    action, _ = model.predict(obs)
    action = action.item()  # Convert ndarray to integer
    obs, reward, done, truncated, _ = env.step(action)
    current_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]['close']
    net_worth_value = env.balance + env.shares_held * current_price
    net_worth.append(net_worth_value)
    actions.append(action)
    # Log simulation step details
    logger.debug(f"Step {env.current_step}: action={action}, balance={env.balance}, shares_held={env.shares_held}, current_price={current_price}, net_worth={net_worth_value}")
    # Stop early if max_steps reached
    if env.current_step >= len(df) - 1:
        done = True

# Log data lengths
logger.info(f"Length of df['Date']: {len(df['Date'])}, Length of net_worth: {len(net_worth)}, Length of actions: {len(actions)}")

# Log net worth and actions distribution
logger.info(f"net_worth sample: {net_worth[:5]}...{net_worth[-5:]}")
logger.info(f"Actions distribution: {Counter(actions)}")
if not np.all(np.isfinite(net_worth)):
    logger.error("net_worth contains invalid values (NaN or inf)")

# Generate plots
plot_results(df, net_worth, actions)