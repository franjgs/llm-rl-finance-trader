# ensemble.py
"""
Main ensemble execution script.
- English comments inside code (for consistency with your preferences).
- Designed to run interactively in Spyder (no `if __name__ == "__main__"` required).
- Robust logging, data checks, basic fallbacks.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Your existing utilities - keep interfaces stable ===
from src.gen_utils import load_config
from src.logging_config import setup_logging
from src.intraday_utils import adjust_config_for_interval
from src.ensemble_core import build_ensemble
from src.metrics import sharpe_ratio, max_drawdown, annualized_return
from src.plot_utils import plot_results

# Optional auto-download
try:
    import yfinance as yf
except Exception:
    yf = None

# ----------------------------- #
# 1) Load configuration & logging
# ----------------------------- #
try:
    cfg = load_config("configs/config_ensemble.yaml")
except Exception as e:
    raise RuntimeError(f"Failed to load config: {e}")

logger = setup_logging(cfg.get("verbose", 1))
logger.info("Configuration loaded successfully")

# ----------------------------- #
# 2) Core parameters
# ----------------------------- #
symbol = cfg.get("stock_symbol")
interval = cfg.get("data_interval", "1d")
start_date = cfg.get("start_date")
end_date = cfg.get("end_date")
initial_cap = float(cfg.get("initial_balance", 10_000))

logger.info(f"Ensemble execution started for {symbol} [{interval}] from {start_date} to {end_date}")

# adjust config for intraday if needed
cfg = adjust_config_for_interval(cfg, interval)

# Paths
paths = cfg.get("paths", {})
raw_path = Path(paths.get("raw_dir", "data/raw")) / f"{symbol}_{interval}_raw.csv"
processed_path = Path(paths.get("processed_dir", "data/processed")) / f"{symbol}_sentiment_combined.csv"

# ----------------------------- #
# 3) Load or download price data
# ----------------------------- #
def load_price_data(path: Path, symbol: str, start: str, end: str, interval: str):
    """Load price data from CSV or download with yfinance if available."""
    if path.exists():
        logger.info(f"Loading cached price data -> {path}")
        df = pd.read_csv(path, parse_dates=True)
        # Try to detect date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            logger.warning("No explicit date column detected in price CSV; attempting index parse.")
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "date"})
        return df
    else:
        logger.warning(f"Raw data not found: {path}")
        if yf is None:
            raise RuntimeError("yfinance not installed and raw data missing.")
        logger.info("Downloading price data via yfinance...")
        end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df_price = yf.download(symbol, start=start, end=end_dt, interval=interval, auto_adjust=True, progress=False)
        if df_price is None or df_price.empty:
            raise ValueError("Downloaded data is empty")
        df_price = df_price.reset_index()
        df_price.columns = [col.lower().replace(" ", "_") for col in df_price.columns]
        df_price["date"] = pd.to_datetime(df_price["date"])
        os.makedirs(path.parent, exist_ok=True)
        df_price.to_csv(path, index=False)
        logger.info(f"Price data saved -> {path}")
        return df_price

df_price = load_price_data(raw_path, symbol, start_date, end_date, interval)

# ----------------------------- #
# 4) Load processed data with sentiment
# ----------------------------- #
if not processed_path.exists():
    logger.error(f"Processed data with sentiment not found: {processed_path}")
    raise FileNotFoundError(f"Missing sentiment data: {processed_path}")

df_sent = pd.read_csv(processed_path)
# Normalize date column names
if "date" in df_sent.columns:
    df_sent["date"] = pd.to_datetime(df_sent["date"])
elif "Date" in df_sent.columns:
    df_sent["date"] = pd.to_datetime(df_sent["Date"])
else:
    raise ValueError("Sentiment file must contain a 'date' or 'Date' column")

logger.info(f"Loaded processed sentiment data -> {processed_path} ({len(df_sent):,} rows)")

# ----------------------------- #
# 5) Merge price and sentiment (inner join on date)
# ----------------------------- #
# Ensure both datasets use the same date timezone/frequency
df_price["date"] = pd.to_datetime(df_price["date"]).dt.tz_localize(None)
df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.tz_localize(None)

# Merge and forward-fill/backfill basic columns if needed
df = pd.merge(df_price, df_sent, on="date", how="left", suffixes=("", "_sent"))
if "sentiment" not in df.columns:
    # attempt common alternatives
    possible = [c for c in df.columns if "sent" in c.lower() or "sentiment" in c.lower()]
    if possible:
        df["sentiment"] = df[possible[0]]
        logger.info(f"Using sentiment column: {possible[0]}")
    else:
        logger.warning("No sentiment column found after merge - filling with 0.0 (neutral)")
        df["sentiment"] = 0.0

# Fill small gaps in sentiment with neutral or forward-fill (configurable)
fill_method = cfg.get("sentiment_fill", "ffill")
if fill_method == "ffill":
    df["sentiment"] = df["sentiment"].fillna(method="ffill").fillna(0.0)
elif fill_method == "zero":
    df["sentiment"] = df["sentiment"].fillna(0.0)
else:
    df["sentiment"] = df["sentiment"].fillna(0.0)

logger.info(f"Combined dataframe prepared ({len(df):,} rows). Columns: {list(df.columns)}")

# ----------------------------- #
# 6) Build ensemble signals
# ----------------------------- #
logger.info("Building ensemble signals...")
try:
    # build_ensemble should accept price+sentiment df and config; returns df with 'position' column in [-1..1] or discrete actions
    final_df = build_ensemble(df.copy(), cfg)
    if not isinstance(final_df, pd.DataFrame):
        raise TypeError("build_ensemble must return a pandas.DataFrame")
except Exception as e:
    logger.exception(f"Ensemble build failed: {e}")
    raise

# Check position column exists
if "position" not in final_df.columns:
    raise KeyError("Result from build_ensemble must contain a 'position' column (numeric position per row)")

# Ensure position is numeric and bounded
final_df["position"] = pd.to_numeric(final_df["position"], errors="coerce").fillna(0.0)
final_df["position"] = final_df["position"].clip(-1.0, 1.0)

# ----------------------------- #
# 7) Backtest with realistic costs
# ----------------------------- #
comm_bps = float(cfg.get("commission_bps", 1.5)) / 10_000  # convert bps -> fraction
slippage_bps = float(cfg.get("slippage_bps", 2.0)) / 10_000

# Align close price column name
close_col = None
for c in ("close", "adj_close", "adj_close_price", "close_price"):
    if c in final_df.columns:
        close_col = c
        break
if close_col is None:
    raise KeyError("No 'close' or 'adj_close' column found in final_df")

final_df = final_df.sort_values("date").reset_index(drop=True)
final_df["raw_return"] = final_df[close_col].pct_change().fillna(0.0) * final_df["position"].shift(1).fillna(0.0)
position_change = final_df["position"].diff().abs().fillna(0.0)
trade_cost = position_change * (comm_bps + slippage_bps)
final_df["strategy_ret"] = final_df["raw_return"] - trade_cost

# Equity curves
strategy_equity = (1 + final_df["strategy_ret"].fillna(0)).cumprod() * initial_cap
bh_equity = (final_df[close_col] / final_df[close_col].iloc[0]) * initial_cap

# ----------------------------- #
# 8) Performance metrics and logging
# ----------------------------- #
sr = sharpe_ratio(final_df["strategy_ret"].dropna())
cagr = annualized_return(strategy_equity)
mdd = max_drawdown(strategy_equity)
outperformance = (strategy_equity.iloc[-1] / bh_equity.iloc[-1] - 1) * 100

logger.info("=" * 60)
logger.info("ENSEMBLE BACKTEST RESULTS")
logger.info("=" * 60)
logger.info(f"Final Equity       : ${strategy_equity.iloc[-1]:,.2f}")
logger.info(f"Buy & Hold Equity  : ${bh_equity.iloc[-1]:,.2f}")
logger.info(f"Total Return       : {(strategy_equity.iloc[-1] / initial_cap - 1):+.2%}")
logger.info(f"CAGR               : {cagr:+.2%}")
logger.info(f"Sharpe Ratio       : {sr:.3f}")
logger.info(f"Max Drawdown       : {mdd:.2%}")
logger.info(f"Outperformance     : {outperformance:+.2f}% vs Buy & Hold")
logger.info("=" * 60)

# ----------------------------- #
# 9) Plot results (best-effort)
# ----------------------------- #
try:
    plot_results(
        symbol=symbol,
        net_worth_with_mean=strategy_equity,
        buy_and_hold=bh_equity,
        initial_balance=initial_cap,
        data_interval=interval,
        walk_forward=False
    )
    plt.show()
except Exception as e:
    logger.warning(f"Plotting failed: {e}")

# ----------------------------- #
# 10) Expose workspace variables for Spyder
# ----------------------------- #
results_df = final_df.copy()
equity_curve = strategy_equity
benchmark_curve = bh_equity
daily_returns = final_df["strategy_ret"]
positions = final_df["position"]
config_used = cfg
symbol_used = symbol
interval_used = interval

logger.info("Ensemble execution completed. Variables exported to workspace (results_df, equity_curve, ...).")