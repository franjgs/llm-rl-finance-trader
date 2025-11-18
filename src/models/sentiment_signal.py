# src/models/sentiment_signal.py
"""
Sentiment-based trading signal generator.

This module converts a continuous sentiment score (from -1.0 to +1.0) into a trading signal.
It supports two output modes:
  - "class": discrete long (+1), neutral (0), or short (-1) signal based on thresholds.
  - "raw": continuous signal (preserves the original sentiment magnitude).

Features:
  - Optional normalization of sentiment scores
  - Configurable positive/negative thresholds
  - Optional rolling-window smoothing
  - Fully compatible with EnsembleModel (implements .predict() wrapper)
"""

import pandas as pd
import numpy as np


class SentimentSignal:
    """
    Generates a trading signal based on news sentiment.

    Parameters
    ----------
    config : dict
        Configuration dictionary from config_ensemble.yaml (sentiment_signal section).
        Expected keys:
            - pos_threshold (float): upper threshold for long signal
            - neg_threshold (float): lower threshold for short signal
            - smoothing_window (int): rolling mean window (default: 3)
            - output_type (str): "class" or "raw" (default: "class")
            - scale_min / scale_max (float): target range for normalization
            - normalize (bool): whether to rescale sentiment to [scale_min, scale_max]
    """

    def __init__(self, config: dict):
        self.pos_threshold = config["pos_threshold"]
        self.neg_threshold = config["neg_threshold"]
        self.smoothing_window = config.get("smoothing_window", 3)
        self.output_type = config.get("output_type", "class").lower()
        self.scale_min = config.get("scale_min", -1.0)
        self.scale_max = config.get("scale_max", 1.0)
        self.normalize = config.get("normalize", False)

    # --------------------------------------------------------------------- #
    # Helper: normalize sentiment to custom range
    # --------------------------------------------------------------------- #
    def _normalize(self, series: pd.Series) -> pd.Series:
        """
        Normalize sentiment scores to the range [scale_min, scale_max].

        Parameters
        ----------
        series : pd.Series
            Raw sentiment values.

        Returns
        -------
        pd.Series
            Normalized sentiment in the configured range.
        """
        min_val = series.min()
        max_val = series.max()
        denominator = max_val - min_val + 1e-9
        normalized = (series - min_val) / denominator
        return normalized * (self.scale_max - self.scale_min) + self.scale_min

    # --------------------------------------------------------------------- #
    # Main signal generation
    # --------------------------------------------------------------------- #
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add column 'signal_sentiment' to the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'sentiment' column with float values in [-1.0, 1.0].

        Returns
        -------
        pd.DataFrame
            Original DataFrame with new column 'signal_sentiment'.
        """
        if "sentiment" not in df.columns:
            df["signal_sentiment"] = np.nan
            return df

        # Ensure sentiment is float and fill NaNs
        sent = df["sentiment"].astype(float).fillna(0.0)

        # Optional normalization to custom range
        if self.normalize:
            sent = self._normalize(sent)

        # -----------------------------------------------------------------
        # Core logic: class vs raw mode
        # -----------------------------------------------------------------
        if self.output_type == "raw":
            # Use the continuous sentiment score directly (no thresholding)
            signal = sent.copy()
        else:
            # Class mode: hard thresholds → +1 / 0 / -1
            signal = np.where(
                sent > self.pos_threshold, 1.0,
                np.where(sent < self.neg_threshold, -1.0, 0.0)
            )

        # Optional smoothing with rolling mean
        if self.smoothing_window > 1:
            signal = pd.Series(signal, index=df.index).rolling(
                window=self.smoothing_window,
                min_periods=1,
                center=False
            ).mean()

        # Assign final signal
        if hasattr(self, 'signal_scaling'):
            signal = signal * float(self.signal_scaling)
        signal = signal.clip(-1.0, 1.0)
        df["signal_sentiment"] = signal
        return df

    # --------------------------------------------------------------------- #
    # Compatibility wrapper for EnsembleModel
    # --------------------------------------------------------------------- #
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper method required by EnsembleModel interface.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with price and sentiment data.

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'signal_sentiment' column.
        """
        return self.apply(df)