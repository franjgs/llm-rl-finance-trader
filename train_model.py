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

def plot_results(df, net_worth_with_sentiment, actions_with_sentiment, net_worth_without_sentiment, actions_without_sentiment):
    """Generate and save plots for stock prices, trading actions, and portfolio net worth with/without sentiment.

    Args:
        df (pd.DataFrame): DataFrame with stock data (Date, close, etc.).
        net_worth_with_sentiment (list): Portfolio net worth with sentiment.
        actions_with_sentiment (list): Trading actions with sentiment (0: hold, 1: buy, 2: sell).
        net_worth_without_sentiment (list): Portfolio net worth without sentiment.
        actions_without_sentiment (list): Trading actions without sentiment.
    """
    # Ensure lengths match
    min_length = min(len(df['Date']), len(net_worth_with_sentiment), len(actions_with_sentiment),
                     len(net_worth_without_sentiment), len(actions_without_sentiment))
    if min_length < len(df['Date']):
        logger.warning(f"Truncating df['Date'] from {len(df['Date'])} to {min_length}")
        df = df.iloc[:min_length]
    if min_length < len(net_worth_with_sentiment):
        logger.warning(f"Truncating net_worth_with_sentiment from {len(net_worth_with_sentiment)} to {min_length}")
        net_worth_with_sentiment = net_worth_with_sentiment[:min_length]
    if min_length < len(actions_with_sentiment):
        logger.warning(f"Truncating actions_with_sentiment from {len(actions_with_sentiment)} to {min_length}")
        actions_with_sentiment = actions_with_sentiment[:min_length]
    if min_length < len(net_worth_without_sentiment):
        logger.warning(f"Truncating net_worth_without_sentiment from {len(net_worth_without_sentiment)} to {min_length}")
        net_worth_without_sentiment = net_worth_without_sentiment[:min_length]
    if min_length < len(actions_without_sentiment):
        logger.warning(f"Truncating actions_without_sentiment from {len(actions_without_sentiment)} to {min_length}")
        actions_without_sentiment = actions_without_sentiment[:min_length]

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clear Matplotlib state
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot stock prices and trading actions (with sentiment)
    ax1.plot(df['Date'], df['close'], label='Close Price', color='blue')
    ax1.set_title('AAPL Close Price and Trading Actions (With Sentiment)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    
    buy_points = df.iloc[[i for i, a in enumerate(actions_with_sentiment) if a == 1]]
    sell_points = df.iloc[[i for i, a in enumerate(actions_with_sentiment) if a == 2]]
    ax1.scatter(buy_points['Date'], buy_points['close'], color='green', marker='^', label='Buy (With Sentiment)')
    ax1.scatter(sell_points['Date'], sell_points['close'], color='red', marker='v', label='Sell (With Sentiment)')
    ax1.legend()
    
    # Plot portfolio net worth
    ax2.plot(df['Date'], net_worth_with_sentiment, label='Net Worth (With Sentiment)', color='purple')
    ax2.plot(df['Date'], net_worth_without_sentiment, label='Net Worth (Without Sentiment)', color='orange', linestyle='--')
    ax2.set_title('Portfolio Net Worth Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    min_net_worth = min(min(net_worth_with_sentiment), min(net_worth_without_sentiment))
    max_net_worth = max(max(net_worth_with_sentiment), max(net_worth_without_sentiment))
    if np.std(net_worth_with_sentiment) < 1e-6 and np.std(net_worth_without_sentiment) < 1e-6:
        ax2.set_ylim(min_net_worth - 100, max_net_worth + 100)
    else:
        ax2.set_ylim(min_net_worth * 0.95, max_net_worth * 1.05)
    ax2.legend()
    logger.info(f"Net Worth (With Sentiment): min={min(net_worth_with_sentiment)}, max={max(net_worth_with_sentiment)}")
    logger.info(f"Net Worth (Without Sentiment): min={min(net_worth_without_sentiment)}, max={max(net_worth_without_sentiment)}")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/aapl_trading_results_comparison.png')
    plt.show()
    plt.close(fig)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL trading model with/without sentiment")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger.info(f"Loaded config: {config}")

# Load data
df = pd.read_csv(config['data_path'])
logger.info(f"Loaded data from {config['data_path']} with {len(df)} rows")

# Initialize environments
env_with_sentiment = lambda: TradingEnv(df, use_sentiment=True)
vec_env_with_sentiment = make_vec_env(env_with_sentiment, n_envs=1)
env_without_sentiment = lambda: TradingEnv(df, use_sentiment=False)
vec_env_without_sentiment = make_vec_env(env_without_sentiment, n_envs=1)
logger.info("Environments initialized")

# Set device (MPS or CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Train PPO model with sentiment
model_with_sentiment = PPO("MlpPolicy", vec_env_with_sentiment, verbose=1, device=device, learning_rate=0.0001, clip_range=0.1)
logger.info(f"Training PPO with sentiment for {config['timesteps']} timesteps")
model_with_sentiment.learn(total_timesteps=config['timesteps'])
model_with_sentiment.save("models/trading_model_with_sentiment")
logger.info("Model with sentiment saved to models/trading_model_with_sentiment")

# Train PPO model without sentiment
model_without_sentiment = PPO("MlpPolicy", vec_env_without_sentiment, verbose=1, device=device, learning_rate=0.0001, clip_range=0.1)
logger.info(f"Training PPO without sentiment for {config['timesteps']} timesteps")
model_without_sentiment.learn(total_timesteps=config['timesteps'])
model_without_sentiment.save("models/trading_model_without_sentiment")
logger.info("Model without sentiment saved to models/trading_model_without_sentiment")

# Simulate trading for visualization (with sentiment)
env = TradingEnv(df, use_sentiment=True)
obs = env.reset()[0]
net_worth_with_sentiment = []
actions_with_sentiment = []
done = False
while not done:
    action, _ = model_with_sentiment.predict(obs)
    action = action.item()
    obs, reward, done, truncated, _ = env.step(action)
    current_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]['close']
    net_worth_value = env.balance + env.shares_held * current_price
    net_worth_with_sentiment.append(net_worth_value)
    actions_with_sentiment.append(action)
    logger.debug(f"Step {env.current_step} (with sentiment): action={action}, balance={env.balance}, shares_held={env.shares_held}, net_worth={net_worth_value}")
    if env.current_step >= len(df) - 1:
        done = True

# Simulate trading for visualization (without sentiment)
env = TradingEnv(df, use_sentiment=False)
obs = env.reset()[0]
net_worth_without_sentiment = []
actions_without_sentiment = []
done = False
while not done:
    action, _ = model_without_sentiment.predict(obs)
    action = action.item()
    obs, reward, done, truncated, _ = env.step(action)
    current_price = env.df.iloc[min(env.current_step, len(env.df) - 1)]['close']
    net_worth_value = env.balance + env.shares_held * current_price
    net_worth_without_sentiment.append(net_worth_value)
    actions_without_sentiment.append(action)
    logger.debug(f"Step {env.current_step} (without sentiment): action={action}, balance={env.balance}, shares_held={env.shares_held}, net_worth={net_worth_value}")
    if env.current_step >= len(df) - 1:
        done = True

# Log data lengths and actions distribution
logger.info(f"Length of df['Date']: {len(df['Date'])}, Length of net_worth_with_sentiment: {len(net_worth_with_sentiment)}, Length of actions_with_sentiment: {len(actions_with_sentiment)}")
logger.info(f"Length of net_worth_without_sentiment: {len(net_worth_without_sentiment)}, Length of actions_without_sentiment: {len(actions_without_sentiment)}")
logger.info(f"Actions distribution (with sentiment): {Counter(actions_with_sentiment)}")
logger.info(f"Actions distribution (without sentiment): {Counter(actions_without_sentiment)}")

# Generate comparison plots
plot_results(df, net_worth_with_sentiment, actions_with_sentiment, net_worth_without_sentiment, actions_without_sentiment)