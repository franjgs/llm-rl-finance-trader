# data_fetch.py
# Run in Spyder → df, cfg, output_path visible in Variable Explorer
# Output: <raw_dir>/<stock_symbol>_raw.csv

import argparse
import yaml
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import yfinance as yf

# --------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# 1. Load configuration
# --------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Configuration loaded: {cfg}")
        return cfg
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        raise


# --------------------------------------------------------------------- #
# 2. Fetch stock data from Yahoo Finance
# --------------------------------------------------------------------- #
def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical OHLCV data using yfinance.

    Args:
        symbol: Stock ticker (e.g., "AAPL").
        start_date: Start date in "YYYY-MM-DD".
        end_date: End date in "YYYY-MM-DD" (inclusive).

    Returns:
        DataFrame with columns: Date, open, high, low, close, volume.

    Raises:
        ValueError: If no data is returned.
    """
    try:
        # Extend end_date by 1 day to include the final trading day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_str = end_dt.strftime("%Y-%m-%d")

        logger.info(f"Downloading {symbol} from {start_date} to {end_date}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_str, interval="1d")

        if df.empty:
            raise ValueError(f"No data for {symbol} in date range")

        # Clean and standardize
        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Date", "open", "high", "low", "close", "volume"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        logger.info(f"Retrieved {len(df)} trading days for {symbol}")
        return df
    except Exception as e:
        logger.error(f"yfinance error for {symbol}: {e}")
        raise


# --------------------------------------------------------------------- #
# 3. Argument parsing
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Fetch historical stock data from Yahoo Finance"
)
parser.add_argument(
    "--config", default="configs/config.yaml", help="Path to config file"
)
args = parser.parse_args()


# --------------------------------------------------------------------- #
# 4. Main execution
# --------------------------------------------------------------------- #
cfg = load_config(args.config)

symbol = cfg["stock_symbol"]
raw_dir = cfg["raw_dir"]

# Auto-generated output path: <raw_dir>/<symbol>_raw.csv
output_path = f"{raw_dir}/{symbol}_raw.csv"

# Fetch data
df = fetch_stock_data(
    symbol=symbol,
    start_date=cfg["start_date"],
    end_date=cfg["end_date"],
)

# Save raw data
os.makedirs(raw_dir, exist_ok=True)
df.to_csv(output_path, index=False)
logger.info(f"Stock data saved → {output_path}")


# --------------------------------------------------------------------- #
# Variables available in Spyder
# --------------------------------------------------------------------- #
# df, cfg, output_path