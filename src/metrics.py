# src/metrics.py
"""
Common financial metrics used across training and analysis.
Avoids code duplication and ensures consistency.
"""
import pandas as pd
import numpy as np
from typing import Optional

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Daily returns (pd.Series).
        risk_free_rate: Annual risk-free rate (default 0.0).
    
    Returns:
        Annualized Sharpe ratio. Returns 0.0 if invalid.
    """
    if len(returns) < 2:
        return 0.0
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(net_worth: pd.Series) -> float:
    """
    Compute maximum drawdown (peak-to-trough decline).
    
    Args:
        net_worth: Portfolio value over time.
    
    Returns:
        Max drawdown as negative decimal (e.g., -0.25 = 25% drawdown).
    """
    if len(net_worth) < 2:
        return 0.0
    peak = net_worth.cummax()
    drawdown = (net_worth - peak) / peak
    return drawdown.min()

def annualized_return(net_worth: pd.Series) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    
    Args:
        net_worth: Portfolio value over time.
    
    Returns:
        Annualized return as decimal.
    """
    if len(net_worth) < 2:
        return 0.0
    days = (net_worth.index[-1] - net_worth.index[0]).days
    years = days / 365.25
    if years <= 0:
        return 0.0
    return (net_worth.iloc[-1] / net_worth.iloc[0]) ** (1 / years) - 1

def outperformance_vs_benchmark(rl_final: float, bh_final: float) -> float:
    """
    Compute outperformance percentage vs benchmark.
    
    Args:
        rl_final: Final net worth of RL strategy.
        bh_final: Final net worth of Buy & Hold.
    
    Returns:
        Outperformance in percentage points.
    """
    if bh_final == 0:
        return 0.0
    return (rl_final / bh_final - 1) * 100

# === PERFORMANCE METRICS PER DAY ===
def log_daily_performance(
    pred_date: pd.Timestamp,
    final_nw: float,
    test_slice: pd.DataFrame,
    initial_balance: float,
    buy_and_hold_value: float,
) -> None:
    """
    Logs performance metrics for a single prediction day.
    All docstrings and comments are in English.
    """
    # Daily return
    daily_return = (final_nw - initial_balance) / initial_balance
    # Buy & Hold return
    bh_return = (buy_and_hold_value - initial_balance) / initial_balance
    # Outperformance
    outperformance = daily_return - bh_return
    # Sharpe ratio (simplified: return / std of daily returns)
    # Since we have only 1 day, use a proxy: return / volatility (high/low range)
    price_range = test_slice["high"].max() - test_slice["low"].min()
    close_price = test_slice["close"].iloc[-1]
    volatility = price_range / close_price if close_price > 0 else 0.01
    sharpe = daily_return / volatility if volatility > 0 else 0.0

    print(
        f"[DAY {pred_date}] "
        f"Net Worth: {final_nw:,.2f} € | "
        f"Return: {daily_return:+.2%} | "
        f"Sharpe: {sharpe:.2f} | "
        f"vs B&H: {outperformance:+.2%}"
    )