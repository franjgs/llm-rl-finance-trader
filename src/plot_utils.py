# src/plot_utils.py
"""
Plotting utilities for trading results.
"""
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def plot_results(
    df: pd.DataFrame,
    net_worth_with: pd.Series,
    actions_with: pd.Series,
    net_worth_without: pd.Series,
    actions_without: pd.Series,
    sharpe_with: float,
    sharpe_without: float,
    symbol: str,
    seed: int,
    use_lstm: bool = False
) -> None:
    """
    Generate and save comparison plots.
    """
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top: Price + actions (with sentiment)
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.set_title(f'{symbol} Close Price and Trading Actions (With Sentiment)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')

    buy_points = df[actions_with == 1]
    sell_points = df[actions_with == 2]
    ax1.scatter(buy_points.index, buy_points['close'], color='green', marker='^', label='Buy (With Sentiment)')
    ax1.scatter(sell_points.index, sell_points['close'], color='red', marker='v', label='Sell (With Sentiment)')
    ax1.legend()

    # Bottom: Net worth
    ax2.plot(df.index, net_worth_with, label=f'Net Worth (With Sentiment, Sharpe: {sharpe_with:.2f})', color='purple')
    ax2.plot(df.index, net_worth_without, label=f'Net Worth (Without Sentiment, Sharpe: {sharpe_without:.2f})', color='orange', linestyle='--')
    ax2.set_title(f'Portfolio Net Worth Comparison (Seed: {seed})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Worth ($)')
    min_nw = min(net_worth_with.min(), net_worth_without.min())
    max_nw = max(net_worth_with.max(), net_worth_without.max())
    if np.std(net_worth_with) < 1e-6 and np.std(net_worth_without) < 1e-6:
        ax2.set_ylim(min_nw - 100, max_nw + 100)
    else:
        ax2.set_ylim(min_nw * 0.95, max_nw * 1.05)
    ax2.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    algo_tag = "lstm" if use_lstm else "mlp"
    plot_path = f"results/{symbol}_trading_results_comparison_seed_{seed}_{algo_tag}.png"
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close(fig)
    logger.info(f"Saved plot to {plot_path}")