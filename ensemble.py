# -*- coding: utf-8 -*-
"""
ensemble.py
Author: Francisco J. González (fran@ing.uc3m.es)
Repository: https://github.com/franjgs
Last update: 2025-11-17

Main ensemble execution script – fully robust version.
- Works with any column naming from yfinance (case-insensitive).
- Handles mixed "Date"/"date" columns automatically.
- Fully interval-aware sentiment loading (1m, 5m, 1h, 1d).
- Ready for Spyder (%run) – no if __name__ == "__main__".
- All comments and docstrings in English.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Project utilities ===
from src.gen_utils import load_config
from src.logging_config import setup_logging
from src.intraday_utils import adjust_config_for_interval
from src.metrics import sharpe_ratio, max_drawdown, annualized_return
from src.plot_utils import plot_results
from src.ensemble.ensemble_model import EnsembleModel

# Optional yfinance (safe import)
try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------------- #
# 1) Load config & logging
# ----------------------------- #
cfg = load_config("configs/config_ensemble.yaml")
logger = setup_logging(cfg.get("verbose", 1))
logger.info("Configuration loaded successfully")

# ----------------------------- #
# 2) Core parameters
# ----------------------------- #
symbol       = cfg.get("stock_symbol", "NVDA")
interval     = cfg.get("data_interval", "1h")
start_date   = cfg.get("start_date", "2023-12-01")
end_date     = cfg.get("end_date", "2025-11-15")
initial_cap  = float(cfg.get("initial_balance", 10_000))

logger.info(f"Ensemble execution started for {symbol} [{interval}] from {start_date} to {end_date}")

cfg = adjust_config_for_interval(cfg, interval)
raw_path = Path("data/raw") / f"{symbol}_{interval}_raw.csv"


# ----------------------------- #
# 3) Load / download price data – BULLETPROOF
# ----------------------------- #
def load_price_data(path: Path, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Load price data from cache or yfinance – handles any date column name/case."""
    if path.exists():
        logger.info(f"Loading cached price data -> {path}")
        df = pd.read_csv(path)
    else:
        logger.info("Downloading price data via yfinance...")
        if yf is None:
            raise RuntimeError("yfinance not installed")

        end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        result = yf.download(symbol, start=start, end=end_dt, interval=interval,
                             auto_adjust=True, progress=False, threads=False)

        # Fallback to daily if intraday not available
        if not isinstance(result, pd.DataFrame) or result.empty:
            logger.warning("Intraday data not available. Falling back to daily.")
            result = yf.download(symbol, start=start, end=end_dt, interval="1d",
                                 auto_adjust=True, progress=False, threads=False)

        df = result.reset_index()

        # Save raw download
        os.makedirs(path.parent, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Price data saved -> {path}")

    # === ROBUST DATE COLUMN NORMALIZATION (never fails again) ===
    date_candidates = ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]
    date_col = next((col for col in date_candidates if col in df.columns), None)
    if date_col is None:
        raise KeyError(f"No date column found in {path}. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True)  # preserve timezone

    # Normalize price columns (case-insensitive)
    col_map = {}
    for target, patterns in {
        "open": ["open"], "high": ["high"], "low": ["low"],
        "close": ["close", "adj close"], "volume": ["volume"]
    }.items():
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                col_map[col] = target
                break
    df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after normalization: {missing}")

    logger.info(f"Price data loaded -> {len(df)} rows | Date column: '{date_col}' → standardized to 'Date'")
    return df[["Date", "open", "high", "low", "close", "volume"]]


df_price = load_price_data(raw_path, symbol, start_date, end_date, interval)


# ----------------------------- #
# 4) Load sentiment data – INTERVAL-AWARE + ROBUST DATE HANDLING
# ----------------------------- #
sentiment_cfg = cfg.get("sentiment", {})
source_tag = sentiment_cfg.get("mode", "combined")
if source_tag != "combined":
    sources = sentiment_cfg.get("sources", ["finnhub"])
    source_tag = "_".join(sources) if len(sources) > 1 else sources[0]

# Primary path: with interval
processed_filename = f"{symbol}_{interval}_sentiment_{source_tag}.csv"
processed_path = Path("data/processed") / processed_filename

# Fallback chain
if not processed_path.exists():
    fallback1 = Path("data/processed") / f"{symbol}_sentiment_{source_tag}.csv"
    if fallback1.exists():
        logger.warning(f"Interval-specific sentiment not found. Using daily fallback: {fallback1}")
        processed_path = fallback1
    else:
        candidates = list(Path("data/processed").glob(f"{symbol}*_sentiment_*.csv"))
        if candidates:
            processed_path = max(candidates, key=lambda p: p.stat().st_mtime)
            logger.warning(f"Auto-detected sentiment file: {processed_path}")
        else:
            raise FileNotFoundError(f"No sentiment file found for {symbol}")

logger.info(f"Loading sentiment data -> {processed_path}")
df_sent = pd.read_csv(processed_path)

# Normalize sentiment date column (same logic as price)
date_col_sent = next((c for c in ["Date", "date", "Datetime", "datetime"] if c in df_sent.columns), df_sent.columns[0])
df_sent = df_sent.rename(columns={date_col_sent: "Date"})
df_sent["Date"] = pd.to_datetime(df_sent["Date"], utc=True)

logger.info(f"Sentiment loaded -> {len(df_sent)} rows | source: {source_tag}")


# ----------------------------- #
# 5) Merge price + sentiment – FINAL & BULLETPROOF
# ----------------------------- #
# Ensure both have timezone-aware Date for safe merge
df_price["Date"] = pd.to_datetime(df_price["Date"])
df_sent["Date"]  = pd.to_datetime(df_sent["Date"])

# Left merge: keep all price bars
df = pd.merge(df_price, df_sent[["Date", "sentiment"]], on="Date", how="left")

# Forward-fill sentiment (daily → intraday), then fill remaining with 0
df["sentiment"] = df["sentiment"].ffill().fillna(0.0).astype("float32")

# Final cleanup
df = df.dropna(subset=["close"]).sort_values("Date").set_index("Date")
logger.info(f"Final merged dataframe ready -> {len(df)} rows | sentiment non-zero: {(df['sentiment'] != 0).sum()}")


# ----------------------------- #
# 6) Build ensemble signals
# ----------------------------- #
logger.info("Building ensemble signals...")
ensemble = EnsembleModel(cfg)
final_df = ensemble.fit_predict(df.copy())


# ------------------ DEBUGGING BLOCK (keep it – super useful) ------------------
debug_dir = Path("results")
debug_dir.mkdir(exist_ok=True)

inspect_cols = [
    "signal_momentum", "signal_sentiment", "signal_xgboost", "signal_lstm",
    "signal_ensemble", "clean_ensemble", "exposure", "exposure_rl", "position"
]
for c in inspect_cols:
    if c not in final_df.columns:
        final_df[c] = np.nan

# Summary diagnostics
summary = []
for c in inspect_cols:
    s = final_df[c].dropna()
    total = len(s)
    nonzero = (s != 0).sum()
    summary.append({
        "col": c,
        "last": float(s.iloc[-1]) if not s.empty else None,
        "pct_nonzero": (nonzero / total * 100) if total > 0 else 0.0,
        "max_abs": float(s.abs().max()) if not s.empty else None,
        "mean": float(s.mean()) if not s.empty else None,
    })
summary_df = pd.DataFrame(summary).set_index("col")
print("\n=== SIGNALS SUMMARY (diagnostic) ===")
print(summary_df.round(4))

print("\n=== LAST 40 ROWS (signals) ===")
print(final_df[inspect_cols].tail(40))

dbg_path = debug_dir / "debug_signals_inspect.csv"
final_df[inspect_cols].to_csv(dbg_path)
print(f"\nDebug CSV saved -> {dbg_path}")
# ------------------ END DEBUGGING BLOCK ------------------


# ----------------------------- #
# 7) Backtest (REVISADO - uso seguro de position & costs)
# ----------------------------- #
if "position" not in final_df.columns:
    raise KeyError("EnsembleModel must return a 'position' column")

# Ensure numeric and clipped (should already be done inside ensemble, but doble cheque)
final_df["position"] = pd.to_numeric(final_df["position"], errors="coerce").fillna(0).clip(-1, 1)

# transaction cost params
comm_bps = cfg.get("commission_bps", 1.5) / 10_000
slippage_bps = cfg.get("slippage_bps", 2.0) / 10_000
per_trade_cost = (comm_bps + slippage_bps)

# raw asset returns (aligned)
final_df["raw_return"] = final_df["close"].pct_change().fillna(0.0)

# Use a single, explicit 'pos' series: position *applied* to returns is the position from previous bar
pos = final_df["position"].shift(1).fillna(0.0)
final_df["applied_position"] = pos  # useful for debugging/inspect

# Strategy return: asset return * position (position is fraction of exposure)
final_df["strategy_ret"] = final_df["raw_return"] * final_df["applied_position"]

# Transaction costs: cost is paid when changing position.
# Compute trade size using the executed position at this bar (pos) compared with previous executed pos.
trade_size = final_df["applied_position"].diff().abs().fillna(final_df["applied_position"].abs())
# Note: first bar cost = abs(initial position) * cost (we assume entering position costs money)
final_df["trade_cost"] = trade_size * per_trade_cost

# Subtract costs from returns (conservative: subtract cost in return units)
final_df["strategy_ret_net"] = final_df["strategy_ret"] - final_df["trade_cost"]

# Final equity curve uses the NET returns
strategy_equity = (1.0 + final_df["strategy_ret_net"]).cumprod() * initial_cap

# Buy & hold (benchmark) - keep index aligned with original Date index
bh_equity = (final_df["close"] / final_df["close"].iloc[0]) * initial_cap

# Make sure indices are datetimes (use original index which should be DatetimeIndex)
if not isinstance(strategy_equity.index, pd.DatetimeIndex):
    try:
        strategy_equity.index = pd.to_datetime(final_df.index)
        bh_equity.index = pd.to_datetime(final_df.index)
    except Exception:
        # fallback: don't change index
        pass


# ----------------------------- #
# 8) Performance metrics
# ----------------------------- #
sr = sharpe_ratio(final_df["strategy_ret"])
cagr = annualized_return(strategy_equity)
mdd = max_drawdown(strategy_equity)
outperf = (strategy_equity.iloc[-1] / bh_equity.iloc[-1] - 1) * 100

logger.info("="*60)
logger.info("ENSEMBLE BACKTEST RESULTS")
logger.info("="*60)
logger.info(f"Final Equity     : ${strategy_equity.iloc[-1]:,.0f}")
logger.info(f"Buy & Hold       : ${bh_equity.iloc[-1]:,.0f}")
logger.info(f"Total Return     : {(strategy_equity.iloc[-1]/initial_cap-1):+.2%}")
logger.info(f"CAGR             : {cagr:+.2%}")
logger.info(f"Sharpe Ratio     : {sr:.3f}")
logger.info(f"Max Drawdown     : {mdd:.2%}")
logger.info(f"Outperformance   : {outperf:+.2f}% vs B&H")
logger.info("="*60)


# ----------------------------- #
# 9) Plot
# ----------------------------- #
try:
    plot_results(
        df=final_df.reset_index(),
        symbol=symbol,
        net_worth_with_mean=strategy_equity,
        buy_and_hold=bh_equity,
        initial_balance=initial_cap,
        data_interval=interval,
        walk_forward=False
    )
    plt.show()
except Exception as e:
    logger.warning(f"Plot failed: {e}")


# ----------------------------- #
# 10) Export to Spyder workspace
# ----------------------------- #
results_df       = final_df.copy()
equity_curve     = strategy_equity
benchmark_curve  = bh_equity
daily_returns    = final_df["strategy_ret"]
positions        = final_df["position"]
config_used      = cfg

logger.info("Ensemble execution completed – all variables exported to workspace.")