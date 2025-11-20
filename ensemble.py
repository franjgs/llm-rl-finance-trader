# ensemble.py
"""
ensemble.py

Main ensemble execution script – fully robust version.
- Works with any column naming from yfinance (case-insensitive).
- Handles mixed "Date"/"date" columns automatically.
- Fully interval-aware sentiment loading (1m, 5m, 1h, 1d).
- Ready for Spyder (%run) – no if __name__ == "__main__".
- All comments and docstrings in English.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Project utilities ===
from src.gen_utils import load_config, load_price_data
from src.logging_config import setup_logging
from src.intraday_utils import adjust_config_for_interval
from src.metrics import sharpe_ratio, max_drawdown, annualized_return
from src.plot_utils import plot_results
from src.ensemble.ensemble_model import EnsembleModel

# ----------------------------- #
# 1) Load config & logging
# ----------------------------- #
cfg = load_config("configs/config_ensemble.yaml")
logger = setup_logging(cfg.get("verbose", 1))
logger.info("Configuration loaded successfully")

# ----------------------------- #
# 2) Core parameters
# ----------------------------- #
symbol = cfg.get("stock_symbol", "NVDA")
interval = cfg.get("data_interval", "1h")
start_date = cfg.get("start_date", "2023-12-01")
end_date = cfg.get("end_date", "2025-11-15")
initial_cap = float(cfg.get("initial_balance", 1_000_000))

logger.info("Ensemble execution started for %s [%s] from %s to %s", symbol, interval, start_date, end_date)

cfg = adjust_config_for_interval(cfg, interval)
raw_path = Path("data/raw") / f"{symbol}_{interval}_raw.csv"

# ----------------------------- #
# 3) Load / download price data – BULLETPROOF
# ----------------------------- #
df_price = load_price_data(raw_path, symbol, start_date, end_date, interval)

# ----------------------------- #
# 4) Load sentiment data – INTERVAL-AWARE + ROBUST DATE HANDLING
# ----------------------------- #
sentiment_cfg = cfg.get("sentiment", {})
source_tag = sentiment_cfg.get("mode", "combined")
if source_tag != "combined":
    sources = sentiment_cfg.get("sources", ["finnhub"])
    source_tag = "_".join(sources) if len(sources) > 1 else sources[0]

processed_filename = f"{symbol}_{interval}_sentiment_{source_tag}.csv"
processed_path = Path("data/processed") / processed_filename

if not processed_path.exists():
    fallback1 = Path("data/processed") / f"{symbol}_sentiment_{source_tag}.csv"
    if fallback1.exists():
        logger.warning("Interval-specific sentiment not found. Using daily fallback: %s", fallback1)
        processed_path = fallback1
    else:
        candidates = list(Path("data/processed").glob(f"{symbol}*_sentiment_*.csv"))
        if candidates:
            processed_path = max(candidates, key=lambda p: p.stat().st_mtime)
            logger.warning("Auto-detected sentiment file: %s", processed_path)
        else:
            raise FileNotFoundError(f"No sentiment file found for {symbol}")

logger.info("Loading sentiment data → %s", processed_path.name)
df_sent = pd.read_csv(processed_path)

date_col_sent = next((c for c in ["Date", "date", "Datetime", "datetime"] if c in df_sent.columns), df_sent.columns[0])
df_sent = df_sent.rename(columns={date_col_sent: "Date"})
df_sent["Date"] = pd.to_datetime(df_sent["Date"], utc=True)
logger.info("Sentiment loaded → %d rows | source: %s", len(df_sent), source_tag)

# ----------------------------- #
# 5) Merge price + sentiment – FINAL & BULLETPROOF
# ----------------------------- #
df_price["Date"] = pd.to_datetime(df_price["Date"])
df_sent["Date"] = pd.to_datetime(df_sent["Date"])

df = pd.merge(df_price, df_sent[["Date", "sentiment"]], on="Date", how="left")
df["sentiment"] = df["sentiment"].ffill().fillna(0.0).astype("float32")
df = df.dropna(subset=["close"]).sort_values("Date").set_index("Date")
logger.info("Final merged dataframe ready → %d rows | non-zero sentiment: %d", len(df), (df["sentiment"] != 0).sum())

# ----------------------------- #
# 6) Build ensemble signals
# ----------------------------- #
logger.info("Building ensemble signals...")
ensemble = EnsembleModel(cfg)
final_df = ensemble.fit_predict(df.copy())

# ------------------ DEBUGGING BLOCK ------------------
debug_dir = Path("results")
debug_dir.mkdir(exist_ok=True)

inspect_cols = [
    "signal_momentum", "signal_sentiment", "signal_xgboost", "signal_lstm",
    "signal_ensemble", "clean_ensemble", "exposure", "exposure_rl", "position"
]
for c in inspect_cols:
    if c not in final_df.columns:
        final_df[c] = np.nan

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
logger.info("\n=== SIGNALS SUMMARY (diagnostic) ===\n%s", summary_df.round(4).to_string())
logger.info("\n=== LAST 40 ROWS (signals) ===\n%s", final_df[inspect_cols].tail(40).to_string())

dbg_path = debug_dir / "debug_signals_inspect.csv"
final_df[inspect_cols].to_csv(dbg_path)
logger.info("Debug CSV saved → %s", dbg_path)

# ----------------------------- #
# 7) Backtest (safe position & costs)
# ----------------------------- #
if "position" not in final_df.columns:
    raise KeyError("EnsembleModel must return a 'position' column")

final_df["position"] = pd.to_numeric(final_df["position"], errors="coerce").fillna(0).clip(-1, 1)

comm_bps = cfg.get("commission_bps", 1.5) / 10_000
slippage_bps = cfg.get("slippage_bps", 2.0) / 10_000
per_trade_cost = comm_bps + slippage_bps

final_df["raw_return"] = final_df["close"].pct_change().fillna(0.0)
pos = final_df["position"].shift(1).fillna(0.0)
final_df["applied_position"] = pos

final_df["strategy_ret"] = final_df["raw_return"] * pos
trade_size = pos.diff().abs().fillna(pos.abs())
final_df["trade_cost"] = trade_size * per_trade_cost
final_df["strategy_ret_net"] = final_df["strategy_ret"] - final_df["trade_cost"]

strategy_equity = (1.0 + final_df["strategy_ret_net"]).cumprod() * initial_cap
bh_equity = (final_df["close"] / final_df["close"].iloc[0]) * initial_cap

# ----------------------------- #
# 8) Performance metrics
# ----------------------------- #
sr = sharpe_ratio(final_df["strategy_ret_net"])

# Safe annualized return (handles int or datetime index)
if isinstance(strategy_equity.index, pd.DatetimeIndex):
    days = (strategy_equity.index[-1] - strategy_equity.index[0]).days
else:
    days = len(strategy_equity) - 1
years = max(days / 365.25, 1e-6)
cagr = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) ** (1 / years) - 1

mdd = max_drawdown(strategy_equity)
outperf = (strategy_equity.iloc[-1] / bh_equity.iloc[-1] - 1) * 100

logger.info("=" * 90)
logger.info("ENSEMBLE BACKTEST RESULTS")
logger.info("=" * 90)
logger.info(f"Ticker         : {symbol}")
logger.info(f"Interval       : {interval}")
logger.info(f"Period         : {start_date} → {end_date}")
logger.info(f"Initial Capital: ${initial_cap:,.0f}")
logger.info("-" * 90)
logger.info(f"Final Equity      : ${strategy_equity.iloc[-1]:,.0f}")
logger.info(f"Buy & Hold Equity : ${bh_equity.iloc[-1]:,.0f}")
logger.info(f"Total Return      : {(strategy_equity.iloc[-1]/initial_cap - 1)*100:+.2f}%")
logger.info(f"CAGR              : {cagr*100:+.2f}%")
logger.info(f"Sharpe Ratio      : {sr:.3f}")
logger.info(f"Max Drawdown      : {mdd*100:.2f}%")
logger.info(f"Outperformance    : {outperf:+.2f}% vs B&H")
logger.info("=" * 90)

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
    logger.warning("Plot failed: %s", e)

# ----------------------------- #
# 10) Export to Spyder workspace
# ----------------------------- #
results_df = final_df.copy()
equity_curve = strategy_equity
benchmark_curve = bh_equity
daily_returns = final_df["strategy_ret_net"]
positions = final_df["position"]
config_used = cfg

logger.info("Ensemble execution completed – all variables exported to workspace.")