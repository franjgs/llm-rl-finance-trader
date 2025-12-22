# data_fetch.py
# Run in Spyder → df visible
# Output: <raw_dir>/<stock_symbol>_raw.csv

import yfinance as yf
import pandas as pd
import argparse
import yaml
import logging
import os
from datetime import datetime, timedelta

# --------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Load configuration
# --------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary with configuration parameters.
    """
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Configuration loaded: {cfg}")
        return cfg
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {path}: {e}")
        raise


# --------------------------------------------------------------------- #
# Fetch stock data
# --------------------------------------------------------------------- #
def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with columns: Date, open, high, low, close, volume.
    """
    try:
        # Extend end date by one day to include the last trading day
        end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end_dt)

        if df.empty:
            raise ValueError(f"No data retrieved for {symbol}")

        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Date", "open", "high", "low", "close", "volume"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        logger.info(f"Fetched {len(df)} rows for {symbol} from {start} to {end}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Fetch raw stock data using yfinance")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()


# --------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------- #
cfg = load_config(args.config)

symbol = cfg["stock_symbol"]
raw_path = f"{cfg['raw_dir']}/{symbol}_raw.csv"

df = fetch_stock_data(
    symbol=symbol,
    start=cfg["start_date"],
    end=cfg["end_date"]
)

os.makedirs(cfg["raw_dir"], exist_ok=True)
df.to_csv(raw_path, index=False)
logger.info(f"Saved raw stock data → {raw_path}")