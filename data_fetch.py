import yfinance as yf
import pandas as pd
import argparse
import yaml
import logging
import os
from datetime import datetime, timedelta

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
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data using yfinance.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: Stock data with Date, open, high, low, close, volume.

    Raises:
        Exception: If data fetching fails.
    """
    try:
        # Extend end_date by one day to ensure inclusion
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date_dt.strftime('%Y-%m-%d'))
        if df.empty:
            logger.error(f"No data retrieved for {symbol} from {start_date} to {end_date}")
            raise ValueError("No data retrieved from yfinance")
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['Date', 'open', 'high', 'low', 'close', 'volume']
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        logger.info(f"Fetched {len(df)} rows for {symbol} from {start_date} to {end_date}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch stock data using yfinance")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger.info(f"Loaded config: {config}")

# Fetch stock data
df = fetch_stock_data(config['stock_symbol'], config['start_date'], config['end_date'])

# Save data
os.makedirs('data/raw', exist_ok=True)
output_path = f"data/raw/{config['stock_symbol']}.csv"
df.to_csv(output_path, index=False)
logger.info(f"Saved stock data to {output_path}")