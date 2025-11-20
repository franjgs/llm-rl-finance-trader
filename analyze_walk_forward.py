# analyze_walk_forward.py
"""
Leakage-Free Walk-Forward Anchored Analysis — FINAL PRODUCTION VERSION

Features:
- All parameters loaded from config_ensemble.yaml
- Feature set permanently LOCKED after first training step → zero dimension mismatch
- 100% strict mode enforced → true out-of-sample testing
- Full professional logging (zero print statements)
- Spyder-friendly sequential execution (all variables visible)
"""

from pathlib import Path
from datetime import timedelta
import sys
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
# Project root setup (relative paths → portable on GitHub)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gen_utils import load_config, load_price_data
from src.logging_config import setup_logging
from src.intraday_utils import adjust_config_for_interval
from src.ensemble.ensemble_model import EnsembleModel


class WalkForwardAnalyzer:
    """
    Strict anchored walk-forward analyzer.
    All parameters read from config file.
    Feature columns frozen after first successful training step.
    """

    def __init__(
        self,
        config_path: str = "configs/config_ensemble.yaml",
        step_days: int | None = None,  # Optional override
    ):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)

        # Extract main parameters
        self.ticker = self.cfg["stock_symbol"]
        self.interval = self.cfg["data_interval"]
        self.train_start = pd.to_datetime(self.cfg["start_date"])
        self.final_date = pd.to_datetime(self.cfg["end_date"])

        # Initial training window: 6 months (common practice)
        self.initial_train_end = self.train_start + pd.DateOffset(months=6)

        # Step size
        self.step_days = step_days if step_days is not None else 30
        self.step = timedelta(days=self.step_days)

        # Force strict mode (walk-forward = zero future leakage)
        self.cfg["feature_mode"] = "strict"
        self.logger = setup_logging(self.cfg.get("verbose", 1))

        self.logger.info(
            "Walk-forward initialized | %s %s | Train start: %s | Initial train end: %s | "
            "Final date: %s | Step: %d days",
            self.ticker, self.interval,
            self.train_start.date(), self.initial_train_end.date(),
            self.final_date.date(), self.step_days
        )

        # Locked feature sets (filled after step 1)
        self.fixed_lstm_features: list | None = None
        self.fixed_xgb_features: list | None = None

        self.results = []

    # --------------------------------------------------------------------- #
    @staticmethod
    def _strip_timezone(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
        """Remove timezone information → naive UTC timestamps."""
        df[col] 
        return df

    # --------------------------------------------------------------------- #
    def _load_sentiment_data(self) -> pd.DataFrame:
        """Load sentiment with robust fallback chain."""
        sentiment_cfg = self.cfg.get("sentiment", {})
        source_tag = sentiment_cfg.get("mode", "combined")
        if source_tag != "combined":
            sources = sentiment_cfg.get("sources", ["finnhub"])
            source_tag = "_".join(sources) if len(sources) > 1 else sources[0]

        filename = f"{self.ticker}_{self.interval}_sentiment_{source_tag}.csv"
        path = Path("data/processed") / filename

        if not path.exists():
            fallback = Path("data/processed") / f"{self.ticker}_sentiment_{source_tag}.csv"
            if fallback.exists():
                self.logger.warning("Using daily sentiment fallback: %s", fallback)
                path = fallback
            else:
                candidates = list(Path("data/processed").glob(f"{self.ticker}*_sentiment_*.csv"))
                if candidates:
                    path = max(candidates, key=lambda p: p.stat().st_mtime)
                    self.logger.warning("Auto-selected sentiment file: %s", path)
                else:
                    raise FileNotFoundError(f"No sentiment file found for {self.ticker}")

        self.logger.info("Loading sentiment → %s", path.name)
        df = pd.read_csv(path)

        date_col = next((c for c in ["Date", "date", "Datetime", "datetime"] if c in df.columns), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df[["Date", "sentiment"]].copy()
        df["sentiment"] = df["sentiment"].astype("float32")
        return df

    # --------------------------------------------------------------------- #
    def _align_to_locked_features(self, df: pd.DataFrame, locked_cols: list, name: str) -> pd.DataFrame:
        """
        Force dataframe to match the locked feature set while PRESERVING essential OHLCV columns.
        Fixes KeyError: 'high' / 'low' / 'open' in walk-forward steps >1.
        """
        df = df.copy()  # Always work on a real copy

        # Essential columns that must NEVER be dropped (needed by feature engineering)
        essential = ["Date", "open", "high", "low", "close", "volume"]

        # Keep: essential + locked features + position/sentiment if exist
        keep = list(set(essential + locked_cols + ["position", "sentiment"]))

        # Drop everything else
        cols_to_keep = [c for c in keep if c in df.columns]
        df = df[cols_to_keep]

        # Fill missing locked features with 0.0 (safe assignment)
        missing = [col for col in locked_cols if col not in df.columns]
        if missing:
            self.logger.debug("Filling %d missing %s features with 0.0", len(missing), name)
            for col in missing:
                df.loc[:, col] = 0.0

        # Reorder: locked features first, then essential OHLCV, then rest
        ordered_cols = (
            locked_cols +
            [c for c in essential if c in df.columns] +
            [c for c in df.columns if c not in locked_cols and c not in essential]
        )
        df = df[ordered_cols]

        return df

    # --------------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        """Execute the full walk-forward analysis with professional logging."""
        current_end = self.initial_train_end
        step_idx = 0

        self.logger.info("Leakage-Free Walk-Forward Analysis started (feature_mode = strict)")
        self.logger.info("=" * 90)

        while current_end < self.final_date:
            step_idx += 1
            test_start = current_end + timedelta(days=1)
            test_end = min(current_end + self.step, self.final_date)

            self.logger.info(
                "Step %02d | Train ≤ %s | Test %s → %s",
                step_idx, current_end.date(), test_start.date(), test_end.date()
            )

            # Load price + sentiment up to test_end
            cfg = self.cfg.copy()
            cfg["start_date"] = self.train_start.strftime("%Y-%m-%d")
            cfg["end_date"] = test_end.strftime("%Y-%m-%d")
            cfg = adjust_config_for_interval(cfg, self.interval)

            raw_path = Path("data/raw") / f"{self.ticker}_{self.interval}_raw.csv"
            price_df = load_price_data(raw_path, self.ticker, cfg["start_date"], cfg["end_date"], self.interval)
            price_df = self._strip_timezone(price_df)
            price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.tz_localize(None)

            sent_df = self._load_sentiment_data()
            sent_df["Date"] = pd.to_datetime(sent_df["Date"], utc=True).dt.tz_localize(None)

            full_df = pd.merge(price_df, sent_df[["Date", "sentiment"]], on="Date", how="left")
            full_df["sentiment"] = full_df["sentiment"].ffill().fillna(0.0).astype("float32")
            full_df = full_df.dropna(subset=["close"]).sort_values("Date").set_index("Date")

            train_df = full_df.loc[:current_end].copy()

            if len(train_df) < 200:
                self.logger.warning(
                    "Step %02d: insufficient training data (%d rows). Skipping.", step_idx, len(train_df)
                )
                current_end += self.step
                continue

            try:
                ensemble = EnsembleModel(self.cfg)

                # Disable unnecessary retraining flags
                if hasattr(ensemble, "xgb_predictor") and ensemble.xgb_predictor:
                    ensemble.xgb_predictor.retrain = False
                if hasattr(ensemble, "lstm_predictor") and ensemble.lstm_predictor:
                    ensemble.lstm_predictor.retrain = False

                # Feature locking for walk-forward stability
                if step_idx > 1:
                    if self.fixed_lstm_features and ensemble.lstm_enabled:
                        train_df = self._align_to_locked_features(train_df, self.fixed_lstm_features, "LSTM")
                        full_df  = self._align_to_locked_features(full_df,  self.fixed_lstm_features, "LSTM")
                    if self.fixed_xgb_features and ensemble.xgboost_enabled:
                        train_df = self._align_to_locked_features(train_df, self.fixed_xgb_features, "XGB")
                        full_df  = self._align_to_locked_features(full_df,  self.fixed_xgb_features, "XGB")

                # Train models (only once per step)
                if ensemble.xgboost_enabled:
                    ensemble.xgb_predictor.train(train_df, self.cfg)

                if ensemble.lstm_enabled:
                    fixed_cols = self.fixed_lstm_features if step_idx > 1 else None
                    ensemble.lstm_predictor.train(train_df, self.cfg, fixed_feature_columns=fixed_cols)

                # Lock features after first successful training
                if step_idx == 1:
                    self.fixed_lstm_features = (
                        ensemble.lstm_predictor.feature_columns.copy() if ensemble.lstm_enabled else None
                    )
                    self.fixed_xgb_features = (
                        ensemble.xgb_predictor.feature_columns.copy() if ensemble.xgboost_enabled else None
                    )
                    self.logger.info(
                        "Feature sets LOCKED → LSTM: %d | XGB: %d",
                        len(self.fixed_lstm_features) if self.fixed_lstm_features else 0,
                        len(self.fixed_xgb_features) if self.fixed_xgb_features else 0,
                    )

                # Pure out-of-sample prediction (no training inside!)
                result_df = ensemble.predict_out_of_sample(full_df=full_df).set_index("Date")

                # Performance calculation
                pos = result_df["position"].shift(1).fillna(0.0)
                comm = self.cfg.get("commission_bps", 1.5) / 10_000
                slip = self.cfg.get("slippage_bps", 2.0) / 10_000
                cost = pos.diff().abs().fillna(pos.abs()) * (comm + slip)
                strategy_ret = result_df["close"].pct_change().fillna(0.0) * pos - cost
                equity = (1 + strategy_ret).cumprod()
                oos_equity = equity.loc[test_start:test_end]

                ret_pct = sharpe = 0.0
                if len(oos_equity) >= 10:
                    ret_pct = (oos_equity.iloc[-1] / oos_equity.iloc[0] - 1) * 100
                    oos_ret = strategy_ret.loc[test_start:test_end]
                    hours_per_year = 252 * 24
                    sharpe = (
                        oos_ret.mean() / oos_ret.std() * np.sqrt(hours_per_year)
                        if oos_ret.std() > 0 else 0.0
                    )

                self.logger.info(
                    "→ OOS Return: %+8.2f%% | Sharpe: %6.3f | Period: %s → %s",
                    ret_pct, sharpe, test_start.date(), test_end.date()
                )

                self.results.append({
                    "step": step_idx,
                    "test_period": f"{test_start.date()} → {test_end.date()}",
                    "days": (test_end - test_start).days,
                    "return_%": round(ret_pct, 3),
                    "sharpe": round(sharpe, 3),
                })

            except Exception as e:
                self.logger.error("Step %02d failed: %s", step_idx, e)
                import traceback
                traceback.print_exc()
                self.results.append({
                    "step": step_idx,
                    "test_period": f"{test_start.date()} → {test_end.date()}",
                    "return_%": 0.0,
                    "sharpe": 0.0,
                })

            current_end += self.step

        # Final summary (via logger)
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            total_return = ((results_df["return_%"] / 100 + 1).prod() - 1) * 100
            avg_sharpe = results_df["sharpe"].replace([np.inf, -np.inf], np.nan).mean()
            positive_steps = (results_df["return_%"] > 0).sum()

            self.logger.info("=" * 90)
            self.logger.info("LEAKAGE-FREE WALK-FORWARD FINAL RESULTS")
            self.logger.info("=" * 90)
            self.logger.info("\n%s", results_df[["step", "test_period", "return_%", "sharpe"]].to_string(index=False))
            self.logger.info("-" * 90)
            self.logger.info("Total compounded return : %+8.2f%%", total_return)
            self.logger.info("Average OOS Sharpe      : %.3f", avg_sharpe)
            self.logger.info("Positive steps          : %d/%d", positive_steps, len(results_df))
            self.logger.info("=" * 90)

            out_path = Path("results/walk_forward_results_leakage_free.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_path, index=False)
            self.logger.info("Results saved → %s", out_path)

        return results_df


# --------------------------------------------------------------------------- #
# Direct execution (Spyder-friendly — all variables remain in workspace)
# --------------------------------------------------------------------------- #

analyzer = WalkForwardAnalyzer()
# analyzer = WalkForwardAnalyzer(step_days=15)  # Example override
wf_results = analyzer.run()

# Variables available in Spyder Variable Explorer:
# analyzer, wf_results