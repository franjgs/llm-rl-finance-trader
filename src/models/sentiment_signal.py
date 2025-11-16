# src/models/sentiment_signal.py
"""
Sentiment-based trading signal.
Applies thresholds on sentiment score to generate long/neutral/short signals.
"""

import pandas as pd
import numpy as np


class SentimentSignal:
    def __init__(self, config):
        self.pos_threshold = config["pos_threshold"]
        self.neg_threshold = config["neg_threshold"]
        self.smoothing_window = config.get("smoothing_window", 3)
        self.output_type = config.get("output_type", "class")

        self.scale_min = config.get("scale_min", -1.0)
        self.scale_max = config.get("scale_max", 1.0)
        self.normalize = config.get("normalize", False)

    # ---------------------------------------------------------
    # Prepare sentiment column
    # ---------------------------------------------------------

    def _normalize(self, x):
        """Normalize sentiment to a fixed range."""
        return (x - x.min()) / (x.max() - x.min() + 1e-9) * \
               (self.scale_max - self.scale_min) + self.scale_min

    # ---------------------------------------------------------
    # Main signal computation
    # ---------------------------------------------------------

    def apply(self, df):
        if "sentiment" not in df.columns:
            df["signal_sentiment"] = np.nan
            return df

        sent = df["sentiment"].astype(float).fillna(0)

        # optional normalization
        if self.normalize:
            sent = self._normalize(sent)

        # basic long/short/neutral rule
        signal = np.where(sent > self.pos_threshold, 1,
                 np.where(sent < self.neg_threshold, -0.5, 0))

        # smoothing
        smoothed = pd.Series(signal, index=df.index).rolling(self.smoothing_window).mean()

        # class or raw output
        if self.output_type == "class":
            final_signal = np.where(smoothed > 0.1, 1,
                           np.where(smoothed < -0.1, -1, 0))
        else:
            final_signal = smoothed

        df["signal_sentiment"] = final_signal
        return df