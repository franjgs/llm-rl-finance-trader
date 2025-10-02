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

def calculate_sharpe_ratio(net_worth):
    """Calculate the Sharpe Ratio for a given net worth time series.

    Args:
        net_worth (list or np.ndarray): Time series of portfolio net worth.

    Returns:
        float: Annualized Sharpe Ratio (assuming zero risk-free rate).
    """
    if len(net_worth) < 2:
        logger.warning("Net worth series too short to calculate Sharpe Ratio")
        return 0.0
    returns = np.diff(net_worth) / net_worth[:-1]  # Daily returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        logger.warning("Zero volatility in returns, Sharpe Ratio undefined")
        return 0.0
    sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
    return sharpe_ratio

def plot_results(df, net_worth_with_sentiment, actions_with_sentiment, net_worth_without_sentiment, actions_without_sentiment, sharpe_with_sentiment, sharpe_without_sentiment):
    """Generate and save plots for stock prices, trading actions, and portfolio net worth with/without sentiment.

    Args:
        df (pd.DataFrame): DataFrame with stock data (index as Date, close, etc.).
        net_worth_with_sentiment (pd.Series): Portfolio net worth with sentiment.
        actions_with_sentiment (pd.Series): Trading actions with sentiment (0: hold, 1: buy, 2: sell).
        net_worth_without_sentiment (pd.Series): Portfolio net worth without sentiment.
        actions_without_sentiment (pd.Series): Trading actions without sentiment.
        sharpe_with_sentiment (float): Sharpe Ratio for strategy with sentiment.
        sharpe_without_sentiment (float): Sharpe Ratio for strategy without sentiment.
    """
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot stock prices and trading actions (with sentiment)
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.set_title('AAPL Close Price and Trading Actions (With Sentiment)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    
    buy_points = df[actions_with_sentiment == 1]
    sell_points = df[actions_with_sentiment == 2]
    ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', label='Buy (With Sentiment)')
    ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', label='Sell (With Sentiment)')
    ax1.legend()
    
    # Plot portfolio net worth with Sharpe Ratios in legend
    ax2.plot(df.index, net_worth_with_sentiment, label=f'Net Worth (With Sentiment, Sharpe: {sharpe_with_sentiment:.2f})', color='purple')
    ax2.plot(df.index, net_worth_without_sentiment, label=f'Net Worth (Without Sentiment, Sharpe: {sharpe_without_sentiment:.2f})', color='orange', linestyle='--')
    ax2.set_title('Portfolio Net Worth Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    min_net_worth = min(net_worth_with_sentiment.min(), net_worth_without_sentiment.min())
    max_net_worth = max(net_worth_with_sentiment.max(), net_worth_without_sentiment.max())
    if np.std(net_worth_with_sentiment) < 1e-6 and np.std(net_worth_without_sentiment) < 1e-6:
        ax2.set_ylim(min_net_worth - 100, max_net_worth + 100)
    else:
        ax2.set_ylim(min_net_worth * 0.95, max_net_worth * 1.05)
    ax2.legend()
    logger.info(f"Net Worth (With Sentiment): min={net_worth_with_sentiment.min()}, max={net_worth_with_sentiment.max()}")
    logger.info(f"Net Worth (Without Sentiment): min={net_worth_without_sentiment.min()}, max={net_worth_without_sentiment.max()}")

    plt.tight_layout()
    plt.show()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/aapl_trading_results_comparison.png')
    plt.close(fig)
    logger.info("Saved plot to results/aapl_trading_results_comparison.png")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL trading model with/without sentiment")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger.info(f"Loaded config: {config}")

# Load data and convert Date to datetime
df = pd.read_csv(config['data_path'])
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)  # Set Date as index
logger.info(f"Loaded data from {config['data_path']} with {len(df)} rows")
logger.info(f"Data date range: {df.index.min()} to {df.index.max()}")

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

# Initialize simulation DataFrames with Date as index
simulation_df_with_sentiment = pd.DataFrame(index=df.index, columns=['net_worth', 'action'])
simulation_df_without_sentiment = pd.DataFrame(index=df.index, columns=['net_worth', 'action'])

# Simulate trading (with sentiment)
env = TradingEnv(df, use_sentiment=True)
obs = env.reset()[0]
for step in range(len(df)):
    action, _ = model_with_sentiment.predict(obs)
    action = action.item()
    obs, reward, done, truncated, info = env.step(action)
    current_price = df.iloc[step]['close']
    net_worth_value = env.balance + env.shares_held * current_price
    date = df.index[step]
    simulation_df_with_sentiment.loc[date, 'net_worth'] = net_worth_value
    simulation_df_with_sentiment.loc[date, 'action'] = action
    logger.debug(f"Step {step} (with sentiment): action={action}, balance={env.balance}, shares_held={env.shares_held}, net_worth={net_worth_value}, date={date}")
    if done or truncated:
        missing_dates = df.index[step + 1:].tolist()
        logger.warning(f"Simulation (with sentiment) stopped early at step {step}, date={date}. Missing dates: {missing_dates}")
        # Fill remaining rows with last values
        last_net_worth = net_worth_value
        last_action = 0  # Hold
        for missing_date in missing_dates:
            simulation_df_with_sentiment.loc[missing_date, 'net_worth'] = last_net_worth
            simulation_df_with_sentiment.loc[missing_date, 'action'] = last_action
        break

# Simulate trading (without sentiment)
env = TradingEnv(df, use_sentiment=False)
obs = env.reset()[0]
for step in range(len(df)):
    action, _ = model_without_sentiment.predict(obs)
    action = action.item()
    obs, reward, done, truncated, info = env.step(action)
    current_price = df.iloc[step]['close']
    net_worth_value = env.balance + env.shares_held * current_price
    date = df.index[step]
    simulation_df_without_sentiment.loc[date, 'net_worth'] = net_worth_value
    simulation_df_without_sentiment.loc[date, 'action'] = action
    logger.debug(f"Step {step} (without sentiment): action={action}, balance={env.balance}, shares_held={env.shares_held}, net_worth={net_worth_value}, date={date}")
    if done or truncated:
        missing_dates = df.index[step + 1:].tolist()
        logger.warning(f"Simulation (without sentiment) stopped early at step {step}, date={date}. Missing dates: {missing_dates}")
        # Fill remaining rows with last values
        last_net_worth = net_worth_value
        last_action = 0  # Hold
        for missing_date in missing_dates:
            simulation_df_without_sentiment.loc[missing_date, 'net_worth'] = last_net_worth
            simulation_df_without_sentiment.loc[missing_date, 'action'] = last_action
        break

# Verify alignment and data completeness
if not simulation_df_with_sentiment.index.equals(df.index):
    logger.error("Index misalignment in simulation_df_with_sentiment")
    raise ValueError("Index misalignment in simulation_df_with_sentiment")
if not simulation_df_without_sentiment.index.equals(df.index):
    logger.error("Index misalignment in simulation_df_without_sentiment")
    raise ValueError("Index misalignment in simulation_df_without_sentiment")
if simulation_df_with_sentiment['net_worth'].isna().any():
    logger.error("Missing net_worth values in simulation_df_with_sentiment")
    raise ValueError("Missing net_worth values in simulation_df_with_sentiment")
if simulation_df_without_sentiment['net_worth'].isna().any():
    logger.error("Missing net_worth values in simulation_df_without_sentiment")
    raise ValueError("Missing net_worth values in simulation_df_without_sentiment")
logger.info("Date indices and data aligned successfully")

# Calculate Sharpe Ratios
sharpe_with_sentiment = calculate_sharpe_ratio(simulation_df_with_sentiment['net_worth'])
sharpe_without_sentiment = calculate_sharpe_ratio(simulation_df_without_sentiment['net_worth'])
logger.info(f"Sharpe Ratio (With Sentiment): {sharpe_with_sentiment:.4f}")
logger.info(f"Sharpe Ratio (Without Sentiment): {sharpe_without_sentiment:.4f}")

# Log data lengths and actions distribution
logger.info(f"Length of df: {len(df)}, Length of simulation_df_with_sentiment: {len(simulation_df_with_sentiment)}, Length of simulation_df_without_sentiment: {len(simulation_df_without_sentiment)}")
logger.info(f"Actions distribution (with sentiment): {Counter(simulation_df_with_sentiment['action'])}")
logger.info(f"Actions distribution (without sentiment): {Counter(simulation_df_without_sentiment['action'])}")

# Save results
results_df = pd.DataFrame({
    'Date': df.index,
    'Net_Worth_With_Sentiment': simulation_df_with_sentiment['net_worth'],
    'Net_Worth_Without_Sentiment': simulation_df_without_sentiment['net_worth'],
    'Actions_With_Sentiment': simulation_df_with_sentiment['action'],
    'Actions_Without_Sentiment': simulation_df_without_sentiment['action']
})
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/aapl_trading_results.csv', index=False)
logger.info("Saved trading results to results/aapl_trading_results.csv")

# Generate comparison plots
plot_results(df, simulation_df_with_sentiment['net_worth'], simulation_df_with_sentiment['action'], simulation_df_without_sentiment['net_worth'], simulation_df_without_sentiment['action'], sharpe_with_sentiment, sharpe_without_sentiment)