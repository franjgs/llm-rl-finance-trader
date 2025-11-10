# data_fetch.py
# Run in Spyder → df, cfg, output_path, source
# Output: <raw_dir>/<stock_symbol>_raw.csv
# Fuente: yfinance (sin session manual) + fallback a Yahoo CSV

import argparse
import yaml
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests

# --------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("data_fetch")

# --------------------------------------------------------------------- #
# 1. Load configuration
# --------------------------------------------------------------------- #
def load_config(path):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")
        return cfg
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        raise

# --------------------------------------------------------------------- #
# 2. Fetch con yfinance (sin session manual)
# --------------------------------------------------------------------- #
def fetch_yfinance(symbol, start_date, end_date):
    """
    Fetch using yfinance with NO session. Let yfinance handle curl_cffi.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"[yfinance] Attempt {attempt + 1}: {symbol} {start_date} → {end_date}")
            df = yf.download(
                tickers=symbol,
                start=start_date,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False  # Silencia barra
                # NO session=...
            )
            if df.empty:
                logger.warning("[yfinance] Empty response, retrying...")
                time.sleep(3)
                continue

            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["Date", "open", "high", "low", "close", "volume"]
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            logger.info(f"[yfinance] Success: {len(df)} rows")
            return df, "yfinance"

        except Exception as e:
            logger.error(f"[yfinance] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None, "yfinance"

    return None, "yfinance"

# --------------------------------------------------------------------- #
# 3. Fallback: Yahoo CSV directo (si yfinance sigue fallando)
# --------------------------------------------------------------------- #
def fetch_yahoo_csv(symbol, start_date, end_date):
    """
    Direct CSV download from Yahoo (bypass yfinance entirely).
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_ts = int(end_dt.timestamp())

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    try:
        logger.info(f"[Yahoo CSV] Fetching {symbol}...")
        response = session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            logger.error(f"[Yahoo CSV] HTTP {response.status_code}")
            return None, "Yahoo CSV"

        df = pd.read_csv(pd.compat.StringIO(response.text))
        if df.empty:
            return None, "Yahoo CSV"

        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Date", "open", "high", "low", "close", "volume"]
        logger.info(f"[Yahoo CSV] Success: {len(df)} rows")
        return df, "Yahoo CSV"

    except Exception as e:
        logger.error(f"[Yahoo CSV] Error: {e}")
        return None, "Yahoo CSV"

# --------------------------------------------------------------------- #
# 4. Main fetcher
# --------------------------------------------------------------------- #
def fetch_stock_data(symbol, start_date, end_date, cfg):
    # 1. yfinance (sin session)
    df, source = fetch_yfinance(symbol, start_date, end_date)
    if df is not None:
        return df, source

    # 2. Fallback: CSV directo
    logger.warning("yfinance failed. Trying direct CSV...")
    df, source = fetch_yahoo_csv(symbol, start_date, end_date)
    if df is not None:
        return df, source

    raise ValueError(f"Failed to fetch {symbol} from Yahoo. Try different network.")


# --------------------------------------------------------------------- #
# 5. Execution
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()  # ← ¡AHORA SÍ!

cfg = load_config(args.config)
symbol = cfg["stock_symbol"]
raw_dir = cfg["raw_dir"]
output_path = f"{raw_dir}/{symbol}_raw.csv"

df, source = fetch_stock_data(symbol, cfg["start_date"], cfg["end_date"], cfg)

os.makedirs(raw_dir, exist_ok=True)
df.to_csv(output_path, index=False)
logger.info(f"Saved {output_path} → Source: {source}")

# Spyder variables
# df, cfg, output_path, source