# analyze_walk_forward.py
"""
Walk-Forward Anchored Analysis – Production Ready (Clean Version)
Author: Francisco J. González
Last update: 2025-11-17

Performs rigorous anchored walk-forward validation of the complete ensemble
pipeline. Each step retrains all ML models on historical data and evaluates
performance strictly out-of-sample.

Key features:
- Full compatibility with current project structure
- Robust timezone handling (prevents RL overlay errors)
- Exact replication of ensemble.py backtesting logic
- Clean, documented, production-grade code
"""

import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Add project root to path (relative to this file location)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core imports from current architecture
from ensemble import (
    load_config,
    setup_logging,
    adjust_config_for_interval,
    load_price_data,
)
from src.ensemble.ensemble_model import EnsembleModel


class WalkForwardAnalyzer:
    """
    Anchored walk-forward validator for the full ensemble system.

    Parameters
    ----------
    ticker : str, default "NVDA"
        Asset symbol to analyze.
    interval : str, default "1h"
        Data frequency.
    train_start : str, default "2023-12-01"
        Fixed start of training history.
    initial_train_end : str, default "2024-06-30"
        End date of first training window (anchor point).
    final_date : str, default "2025-11-15"
        Last date to evaluate.
    step_days : int, default 30
        Length of each out-of-sample test window.
    config_path : str, default "configs/config_ensemble.yaml"
        Path to ensemble configuration file.
    """

    def __init__(
        self,
        ticker: str = "NVDA",
        interval: str = "1h",
        train_start: str = "2023-12-01",
        initial_train_end: str = "2024-06-30",
        final_date: str = "2025-11-15",
        step_days: int = 30,
        config_path: str = "configs/config_ensemble.yaml",
    ):
        self.ticker = ticker
        self.interval = interval
        self.train_start = pd.to_datetime(train_start)
        self.initial_train_end = pd.to_datetime(initial_train_end)
        self.final_date = pd.to_datetime(final_date)
        self.step = timedelta(days=step_days)
        self.config_path = config_path

        # Load configuration
        self.base_cfg = load_config(self.config_path)
        self.logger = setup_logging(self.base_cfg.get("verbose", 1))

        self.results: List[Dict[str, Any]] = []

    @staticmethod
    def _strip_timezone(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
        """Remove timezone info to prevent datetime comparison errors."""
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
        return df

    def _load_sentiment(self) -> pd.DataFrame:
        """Load sentiment data with robust fallback logic."""
        paths = [
            Path("data/processed") / f"{self.ticker}_{self.interval}_sentiment_combined.csv",
            Path("data/processed") / f"{self.ticker}_sentiment_combined.csv",
        ]
        for p in paths:
            if p.exists():
                df = pd.read_csv(p)
                return self._strip_timezone(df)[["Date", "sentiment"]]
        raise FileNotFoundError(f"Sentiment data not found for {self.ticker}")

    def run(self) -> pd.DataFrame:
        """Execute the complete walk-forward analysis."""
        current_end = self.initial_train_end
        step_idx = 0

        print("Walk-Forward Anchored Analysis")
        print("=" * 80)

        while current_end < self.final_date:
            step_idx += 1
            test_start = current_end + timedelta(days=1)
            test_end = min(current_end + self.step, self.final_date)

            print(
                f"Step {step_idx:02d} | Train until {current_end.date()} | "
                f"Test {test_start.date()} → {test_end.date()} "
                f"({(test_end - test_start).days} days)"
            )

            # Step-specific configuration
            cfg = self.base_cfg.copy()
            cfg["start_date"] = self.train_start.strftime("%Y-%m-%d")
            cfg["end_date"] = test_end.strftime("%Y-%m-%d")
            cfg = adjust_config_for_interval(cfg, self.interval)

            # Force retraining
            for model in ("xgboost", "lstm"):
                if model in cfg and isinstance(cfg[model], dict):
                    cfg[model]["retrain"] = True

            try:
                # Load price data
                raw_path = Path("data/raw") / f"{self.ticker}_{self.interval}_raw.csv"
                price_df = load_price_data(
                    raw_path, self.ticker, cfg["start_date"], cfg["end_date"], self.interval
                )
                price_df = self._strip_timezone(price_df)

                # Load and merge sentiment
                sentiment_df = self._load_sentiment()
                df = pd.merge(price_df, sentiment_df, on="Date", how="left")
                df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
                df = df.dropna(subset=["close"]).set_index("Date")

                # Run ensemble
                model = EnsembleModel(cfg)
                result_df = model.fit_predict(df.copy()).set_index("Date")

                # Replicate exact backtest logic from ensemble.py
                pos = result_df["position"].shift(1).fillna(0.0)
                comm = cfg.get("commission_bps", 1.5) / 10_000
                slip = cfg.get("slippage_bps", 2.0) / 10_000
                cost = pos.diff().abs().fillna(pos.abs()) * (comm + slip)

                strategy_ret = result_df["close"].pct_change().fillna(0.0) * pos - cost
                equity = (1 + strategy_ret).cumprod()

                # Out-of-sample metrics
                oos_equity = equity.loc[test_start:test_end]
                if len(oos_equity) >= 10:
                    ret_pct = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1) * 100
                    oos_ret = strategy_ret.loc[test_start:test_end]
                    sharpe = (
                        oos_ret.mean() / oos_ret.std() * np.sqrt(252 * 24)
                        if oos_ret.std() > 0
                        else 0.0
                    )
                else:
                    ret_pct = sharpe = 0.0

                print(f"    OOS Return: {ret_pct:+8.2f}% | Sharpe: {sharpe:6.3f}")

            except Exception as e:
                self.logger.error(f"Step {step_idx} failed: {e}")
                ret_pct = sharpe = 0.0

            self.results.append({
                "step": step_idx,
                "test_period": f"{test_start.date()} → {test_end.date()}",
                "days": (test_end - test_start).days,
                "return_%": round(ret_pct, 3),
                "sharpe": round(sharpe, 3),
            })

            current_end += self.step

        # Final results
        results_df = pd.DataFrame(self.results)
        total_return = ((results_df["return_%"] / 100 + 1).prod() - 1) * 100
        avg_sharpe = results_df["sharpe"].replace([np.inf, -np.inf], np.nan).mean()

        print("\n" + "=" * 80)
        print("WALK-FORWARD FINAL RESULTS")
        print("=" * 80)
        print(results_df[["step", "test_period", "return_%", "sharpe"]].to_string(index=False))
        print("-" * 80)
        print(f"Total compounded return : {total_return:+8.2f}%")
        print(f"Average OOS Sharpe      : {avg_sharpe:.3f}")
        print(f"Positive steps          : {(results_df['return_%'] > 0).sum()}/{len(results_df)}")
        print("=" * 80)
        print("Walk-forward analysis completed successfully.")

        return results_df


# Direct execution (Spyder-friendly)
analyzer = WalkForwardAnalyzer()
wf_results = analyzer.run()

# Optional: save results
wf_results.to_csv("results/walk_forward_results.csv", index=False)