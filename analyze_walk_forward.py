# analyze_walk_forward.py
"""
analyze_walk_forward.py

Complete post-walk-forward analysis using config_walk_forward.yaml.

Features
--------
- Loads config from ``configs/config_walk_forward.yaml``
- Loads **only new format** RL results:
    * ``AAPL_walk_forward_1day_with.csv``
    * ``AAPL_walk_forward_1day_without.csv``
- Computes Buy & Hold benchmark from raw price data (starts on first training day)
- Uses shared metrics from ``src/metrics.py``
- Generates:
    * ``performance_summary.csv`` – full performance table
    * ``equity_curve.png`` – via ``plot_utils.plot_results()``
    * ``drawdown.png`` – drawdown over time
    * ``actions_heatmap.png`` – daily trading actions
- Fully **Spyder-friendly**: no main guard, variables visible in Variable Explorer
- All docstrings and comments in **English**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from src.gen_utils import load_config
from src.metrics import (
    sharpe_ratio,
    max_drawdown,
    annualized_return,
    outperformance_vs_benchmark
)
from src.plot_utils import plot_results

# =============================================================================
# 0. LOGGING & CONFIG
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Load configuration
config = load_config()

# --- Extract core configuration ---
symbol: str = config["stock_symbol"]
initial_balance: float = config.get("initial_balance", 10_000.0)
commission: float = config.get("commission", 0.001)  # Not used in analysis, kept for consistency
raw_dir: Path = Path(config["raw_dir"])
results_dir: Path = Path("results/walk_forward")
sentiment_mode: str = config.get("sentiment_mode", "without")
start_date_cfg: str | None = config.get("start_date")
end_date_cfg: str | None = config.get("end_date")

# Ensure results directory exists
results_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. LOAD RL RESULTS (NEW FORMAT ONLY)
# =============================================================================


def load_rl_results() -> tuple[
    pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None
]:
    """
    Load RL results from **new format** aggregated walk-forward CSVs.

    Files:
        - ``{symbol}_walk_forward_1day_with.csv``
        - ``{symbol}_walk_forward_1day_without.csv``

    Returns
    -------
    nw_with : pd.Series | None
        Net worth series for RL + Sentiment.
    nw_without : pd.Series | None
        Net worth series for RL (No Sentiment).
    actions_with : pd.Series | None
        Last action per day for RL + Sentiment.
    actions_without : pd.Series | None
        Last action per day for RL (No Sentiment).
    """
    with_path = results_dir / f"{symbol}_walk_forward_1day_with.csv"
    without_path = results_dir / f"{symbol}_walk_forward_1day_without.csv"

    nw_with, nw_without = None, None
    actions_with, actions_without = None, None

    if with_path.exists():
        df = pd.read_csv(with_path)
        df["date"] = pd.to_datetime(df["date"])
        nw_with = pd.Series(df["net_worth"].values, index=df["date"], name="RL + Sentiment")
        actions_with = df.set_index("date")["action"]
        logger.info(f"Loaded {len(df)} days: RL + Sentiment (new format)")

    if without_path.exists():
        df = pd.read_csv(without_path)
        df["date"] = pd.to_datetime(df["date"])
        nw_without = pd.Series(df["net_worth"].values, index=df["date"], name="RL (No Sentiment)")
        actions_without = df.set_index("date")["action"]
        logger.info(f"Loaded {len(df)} days: RL (No Sentiment) (new format)")

    if nw_with is None and nw_without is None:
        raise FileNotFoundError(
            f"No walk-forward results found in {results_dir}\n"
            f"Expected files:\n"
            f"  - {with_path.name}\n"
            f"  - {without_path.name}"
        )

    return nw_with, nw_without, actions_with, actions_without


# =============================================================================
# 2. LOAD PRICE DATA
# =============================================================================


def load_prices() -> pd.Series:
    """
    Load closing prices from raw CSV file.

    Returns
    -------
    pd.Series
        Daily close prices with ``Date`` index.

    Raises
    ------
    FileNotFoundError
        If the raw price file is missing.
    """
    csv_path = raw_dir / f"{symbol}_raw.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Price data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    prices = df["close"]

    # Apply optional date filtering from config
    if start_date_cfg:
        prices = prices.loc[start_date_cfg:]
    if end_date_cfg:
        prices = prices.loc[:end_date_cfg]

    return prices


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
logger.info("Starting walk-forward analysis...")

# Load RL results (new format only)
nw_with, nw_without, actions_with, actions_without = load_rl_results()

# Load price data
prices = load_prices()

# Determine walk-forward date range from RL results
all_dates = pd.Index([])
if nw_with is not None:
    all_dates = all_dates.union(nw_with.index)
if nw_without is not None:
    all_dates = all_dates.union(nw_without.index)

start_date = all_dates.min().date()
end_date = all_dates.max().date()
logger.info(f"Walk-forward period: {start_date} to {end_date}")

# Filter prices to match walk-forward period
prices = prices.loc[start_date:end_date]

# === Buy & Hold Benchmark (starts on first training day) ===
first_train_date = prices.index[0]  # First day of walk-forward
start_price = prices.iloc[0]
shares = initial_balance / start_price
bh = shares * prices
bh.name = "Buy & Hold"

# Align all series to a common index
index = all_dates.union(bh.index).sort_values()
bh = bh.reindex(index).ffill()

if nw_with is not None:
    nw_with = nw_with.reindex(index).ffill()
if nw_without is not None:
    nw_without = nw_without.reindex(index).ffill()

# =============================================================================
# 4. PERFORMANCE METRICS
# =============================================================================
metrics = []


def add_strategy(name: str, nw: pd.Series) -> None:
    """
    Compute and append performance metrics for a strategy.

    Parameters
    ----------
    name : str
        Strategy name (e.g., "RL + Sentiment").
    nw : pd.Series
        Net worth series (indexed by date).
    """
    if nw is None or len(nw) == 0:
        return

    final = nw.iloc[-1]
    total_ret = (final / initial_balance - 1) * 100
    ann_ret = annualized_return(nw) * 100
    daily_ret = nw.pct_change().dropna()
    sharpe = sharpe_ratio(daily_ret)
    mdd = max_drawdown(nw) * 100
    outperf = outperformance_vs_benchmark(final, bh.iloc[-1])

    metrics.append({
        "Strategy": name,
        "Final $": f"${final:,.0f}",
        "Total Return": f"{total_ret:+.2f}%",
        "Annualized": f"{ann_ret:+.2f}%",
        "Sharpe": round(sharpe, 3),
        "Max DD": f"{mdd:+.1f}%",
        "vs B&H": f"{outperf:+.2f}%",
    })


# Add strategies
add_strategy("Buy & Hold", bh)
if nw_with is not None:
    add_strategy("RL + Sentiment", nw_with)
if nw_without is not None:
    add_strategy("RL (No Sentiment)", nw_without)

# Save performance summary
metrics_df = pd.DataFrame(metrics)
summary_path = results_dir / f"{symbol}_performance_summary.csv"
metrics_df.to_csv(summary_path, index=False)
logger.info(f"Performance summary saved: {summary_path}")

# Print to console
print("\n" + metrics_df.to_string(index=False))

# =============================================================================
# 5. PLOTS USING plot_utils
# =============================================================================

# Equity Curve + Outperformance
plot_results(
    walk_forward=True,
    symbol=symbol,
    seed=0,
    use_lstm=False,
    net_worth_with_mean=nw_with,
    net_worth_without_mean=nw_without,
    buy_and_hold=bh,
    initial_balance=initial_balance
)

# Drawdown Plot
plt.figure(figsize=(14, 5))
dd_bh = (bh / bh.cummax() - 1) * 100
plt.plot(dd_bh.index, dd_bh, label="Buy & Hold", color="gray", linestyle="--")

if nw_with is not None:
    dd_with = (nw_with / nw_with.cummax() - 1) * 100
    plt.plot(dd_with.index, dd_with, label="RL + Sentiment", color="green")

if nw_without is not None:
    dd_without = (nw_without / nw_without.cummax() - 1) * 100
    plt.plot(dd_without.index, dd_without, label="RL (No Sentiment)", color="blue")

plt.title("Drawdown (%)")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

dd_path = results_dir / f"{symbol}_drawdown.png"
plt.savefig(dd_path, dpi=300, bbox_inches='tight')
plt.show()
logger.info(f"Drawdown plot saved: {dd_path}")

# Actions Heatmap
if actions_with is not None or actions_without is not None:
    action_df = pd.DataFrame(index=index)
    if actions_with is not None:
        action_df["RL + Sentiment"] = actions_with.reindex(index)
    if actions_without is not None:
        action_df["RL (No Sentiment)"] = actions_without.reindex(index)
    action_df = action_df.fillna(method="ffill")

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        action_df.T,
        cmap="viridis",
        cbar_kws={'label': 'Action (0=hold, 1=buy, 2=sell)'},
        yticklabels=True
    )
    plt.title("Daily Trading Actions")
    plt.xlabel("Date")
    plt.tight_layout()

    act_path = results_dir / f"{symbol}_actions_heatmap.png"
    plt.savefig(act_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Actions heatmap saved: {act_path}")

logger.info("Analysis complete. All outputs in results/walk_forward/")