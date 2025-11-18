# analyze_walk_forward.py
"""
Walk-Forward Anchored Analysis — leakage-free version.

Key points:
- For each step we:
  1) build train_df = historical up to train_end (current_end)
  2) instantiate EnsembleModel but DO NOT allow its internal retrain
  3) manually train underlying ML predictors on train_df
  4) call ensemble.fit_predict on full data up to test_end (predict uses trained models)
  5) compute OOS metrics only on test period [test_start:test_end]
"""
from pathlib import Path
from datetime import timedelta
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ensemble import load_config, setup_logging, adjust_config_for_interval, load_price_data
from src.ensemble.ensemble_model import EnsembleModel

class WalkForwardAnalyzer:
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

        self.base_cfg = load_config(self.config_path)
        self.logger = setup_logging(self.base_cfg.get("verbose", 1))
        self.results = []

    def _strip_timezone(self, df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
        return df

    def _load_sentiment(self):
        paths = [
            Path("data/processed") / f"{self.ticker}_{self.interval}_sentiment_combined.csv",
            Path("data/processed") / f"{self.ticker}_sentiment_combined.csv",
        ]
        for p in paths:
            if p.exists():
                df = pd.read_csv(p)
                return self._strip_timezone(df)[["Date", "sentiment"]]
        raise FileNotFoundError(f"Sentiment data not found for {self.ticker}")

    def run(self):
        current_end = self.initial_train_end
        step_idx = 0
        print("Walk-Forward Anchored Analysis")
        print("="*80)

        while current_end < self.final_date:
            step_idx += 1
            test_start = current_end + timedelta(days=1)
            test_end = min(current_end + self.step, self.final_date)

            print(f"Step {step_idx:02d} | Train until {current_end.date()} | Test {test_start.date()} → {test_end.date()}")

            # copy config and set boundaries
            cfg = self.base_cfg.copy()
            cfg["start_date"] = self.train_start.strftime("%Y-%m-%d")
            cfg["end_date"] = test_end.strftime("%Y-%m-%d")
            cfg = adjust_config_for_interval(cfg, self.interval)

            # load data (full up to test_end)
            raw_path = Path("data/raw") / f"{self.ticker}_{self.interval}_raw.csv"
            price_df = load_price_data(raw_path, self.ticker, cfg["start_date"], cfg["end_date"], self.interval)
            price_df = self._strip_timezone(price_df)

            sent_df = self._load_sentiment()
            df = pd.merge(price_df, sent_df, on="Date", how="left")
            df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
            df = df.dropna(subset=["close"]).set_index("Date")

            # split train / test slices
            train_df = df.loc[:current_end].copy()
            test_df = df.loc[test_start:test_end].copy()

            try:
                # instantiate ensemble with internal retrain disabled so fit_predict won't retrain
                ensemble = EnsembleModel(cfg)
                if ensemble.xgb_predictor:
                    ensemble.xgb_predictor.retrain = False
                if ensemble.lstm_predictor:
                    ensemble.lstm_predictor.retrain = False

                # manual training on historical slice (strictly up to current_end)
                if ensemble.xgb_predictor:
                    try:
                        ensemble.xgb_predictor.train(train_df)
                    except Exception as e:
                        self.logger.error(f"XGB train failed: {e}")
                if ensemble.lstm_predictor:
                    try:
                        ensemble.lstm_predictor.train(train_df)
                    except Exception as e:
                        self.logger.error(f"LSTM train failed: {e}")

                # Now run predict across full df (train+test). Models are trained and retrain flags are False.
                result_df = ensemble.fit_predict(df.copy()).set_index("Date")

                # compute OOS using test slice only
                pos = result_df["position"].shift(1).fillna(0.0)
                comm = cfg.get("commission_bps", 1.5)/10_000
                slip = cfg.get("slippage_bps", 2.0)/10_000
                cost = pos.diff().abs().fillna(pos.abs()) * (comm + slip)

                strategy_ret = result_df["close"].pct_change().fillna(0.0) * pos - cost
                equity = (1 + strategy_ret).cumprod()

                oos_equity = equity.loc[test_start:test_end]
                if len(oos_equity) >= 10:
                    ret_pct = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1) * 100
                    oos_ret = strategy_ret.loc[test_start:test_end]
                    sharpe = (oos_ret.mean() / oos_ret.std() * np.sqrt(252 * 24)) if oos_ret.std() > 0 else 0.0
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

        results_df = pd.DataFrame(self.results)
        total_return = ((results_df["return_%"] / 100 + 1).prod() - 1) * 100
        avg_sharpe = results_df["sharpe"].replace([np.inf, -np.inf], np.nan).mean()

        print("\n" + "="*80)
        print("WALK-FORWARD FINAL RESULTS")
        print("="*80)
        print(results_df[["step", "test_period", "return_%", "sharpe"]].to_string(index=False))
        print("-"*80)
        print(f"Total compounded return : {total_return:+8.2f}%")
        print(f"Average OOS Sharpe      : {avg_sharpe:.3f}")
        print(f"Positive steps          : {(results_df['return_%'] > 0).sum()}/{len(results_df)}")
        print("="*80)

        return results_df


# run when executed in Spyder / script

analyzer = WalkForwardAnalyzer()
wf_results = analyzer.run()
wf_results.to_csv("results/walk_forward_results.csv", index=False)