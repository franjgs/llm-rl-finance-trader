"""
Main ensemble execution script – fully robust version.
- Works with any column naming from yfinance.
- Handles missing columns, string dtypes, sentiment merge issues.
- Ready for Spyder (%run) – no if __name__ == "__main__".
- All comments and docstrings in English.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
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
symbol      = cfg.get("stock_symbol", "NVDA")
interval    = cfg.get("data_interval", "1h")
start_date  = cfg.get("start_date", "2023-12-01")
end_date    = cfg.get("end_date", "2025-11-15")
initial_cap = float(cfg.get("initial_balance", 10_000))

logger.info(f"Ensemble execution started for {symbol} [{interval}] from {start_date} to {end_date}")

cfg = adjust_config_for_interval(cfg, interval)

raw_path       = Path("data/raw")      / f"{symbol}_{interval}_raw.csv"
processed_path = Path("data/processed") / f"{symbol}_sentiment_combined.csv"


# ----------------------------- #
# 3) Load / download price data (bullet-proof)
# ----------------------------- #
def load_price_data(path: Path, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Load price data from cache or yfinance – handles every known edge case."""
    if path.exists():
        logger.info(f"Loading cached price data -> {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        return df

    logger.info("Downloading price data via yfinance...")
    if yf is None:
        raise RuntimeError("yfinance not installed")

    end_dt = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    result = yf.download(symbol, start=start, end=end_dt, interval=interval,
                         auto_adjust=True, progress=False, threads=False)

    if isinstance(result, tuple):  # intraday out of range → fallback
        logger.warning(f"yfinance error {result}. Falling back to daily.")
        result = yf.download(symbol, start=start, end=end_dt, interval="1d",
                              auto_adjust=True, progress=False, threads=False)

    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError("No price data could be downloaded")

    df = result.copy().reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Find date column
    date_col = next((c for c in df.columns if "date" in c or "time" in c), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Find close column
    close_col = next((c for c in df.columns if "close" in c), None)
    if close_col:
        df = df.rename(columns={close_col: "close"})

    os.makedirs(path.parent, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Price data saved -> {path}")
    return df


df_price = load_price_data(raw_path, symbol, start_date, end_date, interval)


# ----------------------------- #
# 4) Load sentiment data
# ----------------------------- #
if not processed_path.exists():
    raise FileNotFoundError(f"Sentiment file not found: {processed_path}")

df_sent = pd.read_csv(processed_path)
date_col_sent = "date" if "date" in df_sent.columns else df_sent.columns[0]
df_sent = df_sent.rename(columns={date_col_sent: "date"})
df_sent["date"] = pd.to_datetime(df_sent["date"])
logger.info(f"Loaded sentiment data -> {processed_path} ({len(df_sent)} rows)")


# ----------------------------- #
# 5) Merge price + sentiment – extremely robust
# ----------------------------- #
# Ensure price DataFrame has the required columns (fallback to available ones)
price_cols = []
for col in ["open", "high", "low", "close", "volume"]:
    candidates = [c for c in df_price.columns if col in c.lower()]
    if candidates:
        df_price = df_price.rename(columns={candidates[0]: col})
        price_cols.append(col)

# Always keep close (mandatory)
if "close" not in price_cols:
    raise KeyError("Could not find a close price column after normalization")

# Final clean price frame
df_price_clean = df_price[["date"] + price_cols].copy()
df_price_clean["date"] = pd.to_datetime(df_price_clean["date"]).dt.tz_localize(None)
df_sent["date"]        = pd.to_datetime(df_sent["date"]).dt.tz_localize(None)

# Merge
df = pd.merge(df_price_clean, df_sent[["date", "sentiment"]], on="date", how="left")

# Force numeric types
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["sentiment"] = df["sentiment"].ffill().fillna(0.0)

# Remove any row without price
df = df.dropna(subset=["close"]).reset_index(drop=True)
df = df.set_index("date").sort_index()

logger.info(f"Combined dataframe ready → {len(df)} rows | close dtype = {df['close'].dtype}")


# ----------------------------- #
# 6) Build ensemble signals
# ----------------------------- #
logger.info("Building ensemble signals...")
ensemble = EnsembleModel(cfg)
final_df = ensemble.fit_predict(df.copy())

if "position" not in final_df.columns:
    raise KeyError("EnsembleModel must return a 'position' column")
final_df["position"] = pd.to_numeric(final_df["position"], errors="coerce").fillna(0).clip(-1, 1)


# ----------------------------- #
# 7) Backtest with realistic costs
# ----------------------------- #
comm_bps     = cfg.get("commission_bps", 1.5) / 10_000
slippage_bps = cfg.get("slippage_bps", 2.0)    / 10_000

final_df["raw_return"]    = final_df["close"].pct_change().fillna(0)
final_df["strategy_ret"]  = final_df["raw_return"] * final_df["position"].shift(1).fillna(0)

# Transaction costs
position_change = final_df["position"].diff().abs().fillna(0)
final_df["strategy_ret"] -= position_change * (comm_bps + slippage_bps)

# === CRITICAL FIX: preserve datetime index for metrics ===
strategy_equity = (1 + final_df["strategy_ret"]).cumprod() * initial_cap
# Usamos pd.to_datetime() para garantizar que el índice sea DatetimeIndex, 
# evitando que se interprete como entero.
strategy_equity.index = pd.to_datetime(final_df.index) 
bh_equity         = (final_df["close"] / final_df["close"].iloc[0]) * initial_cap
bh_equity.index         = pd.to_datetime(final_df.index)

# ----------------------------- #
# 8) Performance metrics
# ----------------------------- #
sr      = sharpe_ratio(final_df["strategy_ret"])
cagr    = annualized_return(strategy_equity)      # ahora tiene DatetimeIndex → funciona
mdd     = max_drawdown(strategy_equity)
outperf = (strategy_equity.iloc[-1] / bh_equity.iloc[-1] - 1) * 100

logger.info("="*60)
logger.info("ENSEMBLE BACKTEST RESULTS – NVDA 1h")
logger.info("="*60)
logger.info(f"Final Equity       : ${strategy_equity.iloc[-1]:,.0f}")
logger.info(f"Buy & Hold         : ${bh_equity.iloc[-1]:,.0f}")
logger.info(f"Total Return       : {(strategy_equity.iloc[-1]/initial_cap-1):+.2%}")
logger.info(f"CAGR               : {cagr:+.2%}")
logger.info(f"Sharpe Ratio       : {sr:.3f}")
logger.info(f"Max Drawdown       : {mdd:.2%}")
logger.info(f"Outperformance     : {outperf:+.2f}% vs B&H")
logger.info("="*60)


# ----------------------------- #
# 9) Plot
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
    logger.warning(f"Plot failed: {e}")


# ----------------------------- #
# 10) Export to Spyder workspace
# ----------------------------- #
results_df      = final_df.copy()
equity_curve    = strategy_equity
benchmark_curve = bh_equity
daily_returns   = final_df["strategy_ret"]
positions       = final_df["position"]
config_used     = cfg

logger.info("Ensemble execution completed – all variables exported to workspace.")