import argparse
import pandas as pd
import yfinance as yf
import logging
import yaml
import os
from datetime import datetime

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

def fetch_stock_data(symbol, start_date, end_date, output_path):
    """Fetch stock data using yfinance and save to CSV.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL').
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        output_path (str): Path to save the CSV file.
    """
    try:
        # Adjust end_date to avoid future dates
        today = datetime.now().strftime('%Y-%m-%d')
        if end_date > today:
            logger.warning(f"end_date {end_date} is in the future; adjusting to {today}")
            end_date = today
        # Download data
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            logger.error(f"No data retrieved for {symbol} from {start_date} to {end_date}")
            raise ValueError("Empty DataFrame")
        # Reset index to have Date as a column
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['Date', 'open', 'high', 'low', 'close', 'volume']
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Fetched {len(df)} rows for {symbol} from {start_date} to {end_date}")
        logger.info(f"Stock data date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved stock data to {output_path}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch stock data using yfinance")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
symbol = config['stock_symbol']
start_date = config['start_date']
end_date = config['end_date']
output_path = f"data/raw/{symbol}.csv"

# Fetch and save stock data
fetch_stock_data(symbol, start_date, end_date, output_path)