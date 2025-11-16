"""
Plotting utilities for trading results.
Uses shared metrics from src/metrics.py.
Includes Buy & Hold benchmark and outperformance calculation.
"""
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from src.metrics import sharpe_ratio, outperformance_vs_benchmark
from typing import Optional

logger = logging.getLogger(__name__)

def plot_results(
    df: Optional[pd.DataFrame] = None,
    net_worth_with: Optional[pd.Series] = None,
    actions_with: Optional[pd.Series] = None,
    net_worth_without: Optional[pd.Series] = None,
    actions_without: Optional[pd.Series] = None,
    symbol: str = "",
    seed: int = 0,
    use_lstm: bool = False,
    walk_forward: bool = False,
    net_worth_with_mean: Optional[pd.Series] = None,
    net_worth_without_mean: Optional[pd.Series] = None,
    buy_and_hold: Optional[pd.Series] = None,
    price_data: Optional[pd.Series] = None,
    initial_balance: float = 10000.0,
    data_interval: str = "1h" # <-- Added the required argument
) -> None:
    """
    Generate comparison plots for trading strategies, including the Buy & Hold benchmark.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing price and position data.
    net_worth_with_mean (pd.Series): The equity curve for the strategy (used in static mode for backtest).
    buy_and_hold (pd.Series): The equity curve for the benchmark.
    symbol (str): Stock symbol.
    data_interval (str): Time interval (e.g., '1h', '1d').
    walk_forward (bool): If True, plots a walk-forward specific graph.
    """
    plt.clf()

    # === COMPUTE BUY & HOLD IF NOT PROVIDED ===
    if buy_and_hold is None and price_data is not None and len(price_data) > 0:
        start_price = price_data.iloc[0]
        shares = initial_balance / start_price
        buy_and_hold = shares * price_data
        buy_and_hold.name = "Buy & Hold"

    # === WALK-FORWARD MODE ===
    if walk_forward:
        if buy_and_hold is None:
            logger.warning("Buy & Hold not provided and price_data missing. Skipping plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(13, 7))

        # Plot Buy & Hold
        ax.plot(buy_and_hold.index, buy_and_hold, label="Buy & Hold", color="gray", linewidth=2, linestyle="--")

        # Plot RL strategies
        if net_worth_with_mean is not None:
            ax.plot(net_worth_with_mean.index, net_worth_with_mean,
                     label="RL + Sentiment", color="green", linewidth=2)
            final_rl = net_worth_with_mean.iloc[-1]
            final_bh = buy_and_hold.iloc[-1]
            outperf = outperformance_vs_benchmark(final_rl, final_bh)
            sharpe = sharpe_ratio(net_worth_with_mean.pct_change().dropna())
            ax.text(0.02, 0.88, f"Outperf: +{outperf:+.1f}%\nSharpe: {sharpe:.2f}",
                     transform=ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

        if net_worth_without_mean is not None:
            ax.plot(net_worth_without_mean.index, net_worth_without_mean,
                     label="RL (No Sentiment)", color="blue", linewidth=2)
            final_rl = net_worth_without_mean.iloc[-1]
            final_bh = buy_and_hold.iloc[-1]
            outperf = outperformance_vs_benchmark(final_rl, final_bh)
            sharpe = sharpe_ratio(net_worth_without_mean.pct_change().dropna())
            ax.text(0.02, 0.70, f"Outperf: +{outperf:+.1f}%\nSharpe: {sharpe:.2f}",
                     transform=ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

        ax.set_title(f"Walk-Forward 1-Day Ahead | {symbol} | Seed {seed} | {'LSTM' if use_lstm else 'MLP'}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()

    # === STATIC SPLIT / BACKTEST MODE ===
    else:
        if df is None:
            logger.warning("No data provided for static mode.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [1, 1]})

        # Top: Price + actions
        ax1.plot(df.index, df['close'], label='Close Price', color='blue')
        # Use interval in the title
        ax1.set_title(f'[{data_interval}] {symbol} Close Price and Trading Actions')
        ax1.set_ylabel('Price ($)')
        if actions_with is not None:
            # Note: Ensemble script doesn't pass 'actions_with', this is usually for RL logs
            buy_points = df[actions_with == 1]
            sell_points = df[actions_with == 2]
            ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', label='Buy')
            ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', label='Sell')
        ax1.legend()

        # Bottom: Net worth + Buy & Hold
        if buy_and_hold is not None:
            ax2.plot(buy_and_hold.index, buy_and_hold, label="Buy & Hold (Benchmark)", color="gray", linestyle="--", linewidth=2)
        
        # The ensemble script passes strategy_equity to net_worth_with_mean, so we use it here.
        if net_worth_with_mean is not None:
            ax2.plot(net_worth_with_mean.index, net_worth_with_mean, label='Ensemble Strategy', color='purple')
        
        if net_worth_without is not None:
            ax2.plot(df.index, net_worth_without, label='Without Sentiment', color='orange', linestyle='--')
            
        ax2.set_title(f'Portfolio Net Worth (Seed: {seed})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Net Worth ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # === SAVE ===
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/walk_forward", exist_ok=True)
    algo_tag = "lstm" if use_lstm else "mlp"

    if walk_forward:
        plot_path = f"results/walk_forward/{symbol}_walk_forward_1day.png"
    else:
        plot_path = f"results/{symbol}_trading_results_comparison_seed_{seed}_{algo_tag}.png"

    # Save and show the plot
    plt.gcf().savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)
    plt.close(fig) 
    logger.info(f"Plot saved: {plot_path}")