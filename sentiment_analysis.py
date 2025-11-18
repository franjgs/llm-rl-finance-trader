# -*- coding: utf-8 -*-
"""
sentiment_analysis.py
Author: Francisco J. González (fran@ing.uc3m.es)
Repository: https://github.com/franjgs
Last update: 2025-11-17
Enrich OHLCV data with daily news sentiment.
Features:
- Works perfectly with any interval (1m → 1d)
- Daily sentiment is correctly forward-filled to all intraday bars
- Cache + deduplication + Apple Silicon MPS acceleration
- Multiple news sources (Finnhub, Alpha Vantage, NewsAPI, Yahoo, Google News, GDELT)
- Direct execution in Spyder → variables left in workspace: df, sentiment, cfg, output_path
"""
import yaml
import pandas as pd
import logging
import os
import json
from datetime import datetime, date, timedelta
from typing import List, Dict
import torch
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import feedparser
import finnhub
import urllib.parse

# --------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Sentiment Analyzer (MPS-aware)
# --------------------------------------------------------------------- #
class SentimentAnalyzer:
    """Hugging Face sentiment analyzer with Apple Silicon MPS support."""
   
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment model.
        Parameters
        ----------
        model_name : str
            Hugging Face model identifier.
        """
        logger.info(f"Loading sentiment model: {model_name}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe_device = 0 if torch.backends.mps.is_available() else -1
        logger.info(f"Using device: {device}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
            self.pipe = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=pipe_device,
                truncation=True,
                max_length=512
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Falling back to distilbert.")
            fallback = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModelForSequenceClassification.from_pretrained(fallback)
            model.to(device)
            self.pipe = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=pipe_device,
                truncation=True,
                max_length=512
            )

    def score(self, text: str) -> float:
        """
        Return numeric sentiment score in range [-1.0, 1.0].
        Parameters
        ----------
        text : str
            Input headline(s) concatenated with " | ".
        Returns
        -------
        float
            Positive → +score, Negative → -score, Neutral/Error → 0.0
        """
        if not text or not text.strip():
            return 0.0
        try:
            result = self.pipe(text)[0]
            label, score = result["label"], result["score"]
            # ← ONLY CHANGE: robust label handling for both models
            if label.upper() in ["POSITIVE", "POS"]:
                return score
            elif label.upper() in ["NEGATIVE", "NEG"]:
                return -score
            return 0.0
        except Exception as e:
            logger.debug(f"Sentiment scoring error: {e}")
            return 0.0

# --------------------------------------------------------------------- #
# Cache management
# --------------------------------------------------------------------- #
def get_cache_path(cfg: dict, symbol: str) -> str:
    """Return full path to the JSON sentiment cache file."""
    cache_dir = cfg.get("paths", {}).get("cache_dir", "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{symbol.upper()}_sentiment_cache.json")

def load_cache(cfg: dict, symbol: str) -> List[Dict]:
    """Load previously cached news items."""
    path = get_cache_path(cfg, symbol)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Cache loaded: {len(data)} items from {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return []

def save_cache(cfg: dict, symbol: str, data: List[Dict]):
    """Persist merged news cache to disk."""
    path = get_cache_path(cfg, symbol)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cache saved: {len(data)} items → {path}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def merge_dedup(old: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge old and new news, removing duplicates by (headline, source)."""
    seen = {(item["headline"], item.get("source", "")) for item in old}
    filtered_new = [item for item in new if (item["headline"], item.get("source", "")) not in seen]
    return old + filtered_new

# --------------------------------------------------------------------- #
# News sources – YOUR ORIGINAL 6 FUNCTIONS (unchanged, just placeholders here)
# --------------------------------------------------------------------- #
# ← Keep your real fetch_* functions exactly as they were
def fetch_finnhub_news(symbol: str, start: str, end: str) -> List[Dict]: ...
def fetch_alphavantage_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]: ...
def fetch_newsapi_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]: ...
def fetch_yahoo_news(symbol: str = "AAPL") -> List[Dict]: ...
def fetch_google_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]: ...
def fetch_gdelt_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]: ...

# --------------------------------------------------------------------- #
# News dispatcher
# --------------------------------------------------------------------- #
def fetch_news(symbol: str, start_dt: date, end_dt: date, mode: str, sources: List[str]) -> List[Dict]:
    """Fetch news from selected sources according to mode."""
    all_items: List[Dict] = []
    def add(items: List[Dict]):
        if items:
            all_items.extend(items)
    if mode == "individual":
        src = sources[0] if sources else "finnhub"
        if src == "finnhub": add(fetch_finnhub_news(symbol, start_dt.isoformat(), end_dt.isoformat()))
        elif src == "alphavantage": add(fetch_alphavantage_news(symbol, start_dt, end_dt))
        elif src == "newsapi": add(fetch_newsapi_news(symbol, start_dt, end_dt))
        elif src == "yahoo": add(fetch_yahoo_news(symbol))
        elif src == "googlenews": add(fetch_google_news(symbol, start_dt, end_dt))
        elif src == "gdelt": add(fetch_gdelt_news(symbol, start_dt, end_dt))
    elif mode == "combined":
        for src in sources:
            if src == "finnhub": add(fetch_finnhub_news(symbol, start_dt.isoformat(), end_dt.isoformat()))
            elif src == "alphavantage": add(fetch_alphavantage_news(symbol, start_dt, end_dt))
            elif src == "newsapi": add(fetch_newsapi_news(symbol, start_dt, end_dt))
            elif src == "yahoo": add(fetch_yahoo_news(symbol))
            elif src == "googlenews": add(fetch_google_news(symbol, start_dt, end_dt))
            elif src == "gdelt": add(fetch_gdelt_news(symbol, start_dt, end_dt))
    logger.info(f"Fetched {len(all_items)} raw headlines")
    return all_items

# ===================================================================== #
# DIRECT EXECUTION FOR SPYDER (no main, no argparse)
# ===================================================================== #
# Edit only this line if you want to test quickly
config_path = "configs/config_ensemble.yaml"

# Load configuration
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

symbol = cfg["stock_symbol"]
interval = cfg.get("data_interval", "1d")
start_str = cfg["start_date"]
end_str = cfg["end_date"]
raw_dir = cfg["paths"]["raw_dir"]
proc_dir = cfg["paths"]["processed_dir"]
sentiment_cfg = cfg.get("sentiment", {})
mode = sentiment_cfg.get("mode", "combined").lower()
sources = sentiment_cfg.get("sources", ["finnhub"])
model_name = sentiment_cfg.get("model", "ProsusAI/finbert")
force_refresh = sentiment_cfg.get("force_refresh", False)

start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

logger.info(f"Processing {symbol} | {interval} | {start_str} → {end_str} | Mode: {mode} | Sources: {sources}")

# Load raw OHLCV data (keeps timestamp for intraday)
raw_csv = os.path.join(raw_dir, f"{symbol}_{interval}_raw.csv")
if not os.path.exists(raw_csv):
    raise FileNotFoundError(f"Raw data not found: {raw_csv}")

df = pd.read_csv(raw_csv, parse_dates=["Date"])
df = df[(df["Date"].dt.date >= start_dt) & (df["Date"].dt.date <= end_dt)].copy()
df = df.sort_values("Date").reset_index(drop=True)
logger.info(f"Loaded {len(df)} bars")

# Fetch and cache news
cached = [] if force_refresh else load_cache(cfg, symbol)
fresh = fetch_news(symbol, start_dt, end_dt, mode, sources)
all_news = merge_dedup(cached, fresh)
save_cache(cfg, symbol, all_news)

# Build daily concatenated headlines (max 3 per day)
daily_headlines: Dict[str, str] = {}
for item in all_news:
    day = item["date"]
    daily_headlines.setdefault(day, []).append(item["headline"])

daily_text = {
    day: " | ".join(headlines[:3]) if headlines else ""
    for day, headlines in daily_headlines.items()
}

# Score daily sentiment
analyzer = SentimentAnalyzer(model_name)
daily_sentiment = {
    day: analyzer.score(text)
    for day, text in daily_text.items()
}

# Map to original dataframe (daily sentiment → all intraday bars)
df["calendar_date"] = df["Date"].dt.date.astype(str)
df["news"] = df["calendar_date"].map(daily_text).fillna("")
df["sentiment"] = df["calendar_date"].map(daily_sentiment).fillna(0.0)

# Forward-fill sentiment on weekends/holidays (common practice in trading)
# Fixed FutureWarning – pandas 2.2+ compliant
df["sentiment"] = (
    df["sentiment"]
      .replace(0.0, pd.NA)
      .ffill()
      .fillna(0.0)          # ← CHANGE 1: fillna BEFORE astype
      .astype("float32")    # ← CHANGE 2: now safe – no NAType error
)

# Cleanup
df.drop(columns=["calendar_date"], inplace=True)

# Save enriched dataset
os.makedirs(proc_dir, exist_ok=True)
source_tag = "combined" if mode == "combined" else "_".join(sources)
output_path = os.path.join(proc_dir, f"{symbol}_{interval}_sentiment_{source_tag}.csv")
df.to_csv(output_path, index=False)

# Summary
logger.info(f"Enriched data saved → {output_path}")
logger.info(f"Sentiment stats → mean={df['sentiment'].mean():.5f} | std={df['sentiment'].std():.5f} | non-zero={(df['sentiment'] != 0).sum()}")
pos = (df["sentiment"] > 0.1).sum()
neg = (df["sentiment"] < -0.1).sum()
neu = len(df) - pos - neg
logger.info(f"Sentiment distribution → POS={pos} | NEG={neg} | NEU={neu}")

# Variables available in Spyder workspace
sentiment = df["sentiment"].copy()

print("\n" + "="*70)
print("SENTIMENT ENRICHMENT COMPLETED SUCCESSFULLY")
print("="*70)
print(f"File → {output_path}")
print(f"Symbol → {symbol}")
print(f"Interval → {interval}")
print(f"Bars → {len(df)}")
print(f"First bar → {df['Date'].iloc[0]}")
print(f"Last bar → {df['Date'].iloc[-1]}")
print(f"Non-zero sentiment → {(df['sentiment'] != 0).sum()} bars")
print(f"Source tag → {source_tag}")
print("="*70 + "\n")