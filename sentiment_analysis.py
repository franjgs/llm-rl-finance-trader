#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:04:42 2025

@author: fran
"""

# src/sentiment_analysis.py
"""
Enrich historical stock data with daily news sentiment using multiple sources.

Key features:
- Global date range from config.yaml (sources clip internally)
- Apple Silicon MPS acceleration
- Cache with deduplication by (headline, source)
- 6 news sources with time limits respected
- Outputs numeric sentiment (-1 to +1) for RL training
- Input: data/raw/<symbol>_raw.csv
- Output: data/processed/<symbol>_sentiment_<source>.csv
- Run in Spyder → df, sentiment visible
"""

import argparse
import yaml
import pandas as pd
import logging
import os
import json
import torch
import requests
from datetime import datetime, timedelta, date
from typing import List, Dict
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
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Sentiment Analyzer (MPS-aware)
# --------------------------------------------------------------------- #
class SentimentAnalyzer:
    """Hugging Face sentiment analyzer with MPS support."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer with the specified model.
        
        Parameters
        ----------
        model_name : str
            Hugging Face model name. Defaults to FinBERT.
        """
        logger.info(f"Loading sentiment model: {model_name}")
        mps_available = torch.backends.mps.is_available()
        device_name = "mps" if mps_available else "cpu"
        pipeline_device = 0 if mps_available else -1
        logger.info(f"Using device: {device_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device_name)
            self.model.eval()
            self.pipe = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Falling back to distilbert.")
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device_name)
            self.model.eval()
            self.pipe = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device
            )

    def score(self, text: str) -> float:
        """
        Return numeric sentiment: +score (POS), -score (NEG), 0.0 (NEU/ERROR).
        
        Parameters
        ----------
        text : str
            Input headline or concatenated news.
        
        Returns
        -------
        float
            Sentiment score in range [-1.0, 1.0].
        """
        if not text.strip():
            return 0.0
        try:
            res = self.pipe(text[:512])[0]
            label, score = res["label"], res["score"]
            if "POS" in label.upper():
                return score
            elif "NEG" in label.upper():
                return -score
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Sentiment scoring failed: {e}")
            return 0.0


# --------------------------------------------------------------------- #
# Cache management
# --------------------------------------------------------------------- #
def get_cache_file(cfg: dict, symbol: str) -> str:
    """Return the path to the cache file for the given symbol."""
    cache_dir = cfg.get("cache_dir", "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{symbol.upper()}_sentiment_cache.json")


def load_cache(cfg: dict, symbol: str) -> List[Dict]:
    """Load cached news items from disk."""
    path = get_cache_file(cfg, symbol)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                logger.info(f"Cache loaded: {len(data)} items → {path}")
                return data
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    return []


def save_cache(cfg: dict, symbol: str, data: List[Dict]):
    """Save news items to cache."""
    path = get_cache_file(cfg, symbol)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cache saved: {len(data)} items → {path}")
    except Exception as e:
        logger.error(f"Cache save failed: {e}")


def merge_and_deduplicate(old: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge old and new news, deduplicating by (headline, source)."""
    seen = {(d["headline"], d.get("source", "")): d for d in old}
    seen.update({(d["headline"], d.get("source", "")): d for d in new})
    return list(seen.values())


# --------------------------------------------------------------------- #
# 1. Finnhub – Recent (~30 days)
# --------------------------------------------------------------------- #
def fetch_finnhub_news(symbol: str, start: str, end: str) -> List[Dict]:
    """Fetch news from Finnhub (requires API key)."""
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("FINNHUB_API_KEY missing")
        return []
    
    client = finnhub.Client(api_key=api_key)
    today = datetime.now().strftime("%Y-%m-%d")
    if end > today:
        end = today
    if start > end:
        return []
    
    try:
        raw = client.company_news(symbol, _from=start, to=end)
        items = []
        for entry in raw:
            if isinstance(entry, dict) and "headline" in entry:
                try:
                    date_str = datetime.fromtimestamp(entry["datetime"]).date().isoformat()
                    items.append({"date": date_str, "headline": entry["headline"], "source": "finnhub"})
                except:
                    continue
        logger.info(f"Finnhub: {len(items)} headlines")
        return items
    except Exception as e:
        logger.error(f"Finnhub error: {e}")
        return []


# --------------------------------------------------------------------- #
# 2. Alpha Vantage – Historical (2+ years)
# --------------------------------------------------------------------- #
def fetch_alphavantage_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]:
    """Fetch news from Alpha Vantage (requires API key)."""
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        logger.warning("ALPHAVANTAGE_API_KEY missing")
        return []
    
    url = "https://www.alphavantage.co/query"
    delta = timedelta(days=30)
    current = start_dt
    all_items = []
    
    while current < end_dt:
        next_dt = min(current + delta, end_dt)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": current.strftime("%Y%m%dT0000"),
            "time_to": next_dt.strftime("%Y%m%dT2359"),
            "limit": 1000,
            "apikey": api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if "feed" not in data:
                logger.info(f"Alpha Vantage: {data.get('Note', 'No data')}")
                break
            for item in data["feed"]:
                try:
                    date_str = datetime.strptime(item["time_published"][:8], "%Y%m%d").date().isoformat()
                    all_items.append({"date": date_str, "headline": item["title"], "source": "alphavantage"})
                except:
                    continue
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            break
        current = next_dt + timedelta(days=1)
    
    logger.info(f"Alpha Vantage: {len(all_items)} headlines")
    return all_items


# --------------------------------------------------------------------- #
# 3. NewsAPI – Last 30 days (free tier)
# --------------------------------------------------------------------- #
def fetch_newsapi_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]:
    """Fetch news from NewsAPI.org (free tier, last 30 days)."""
    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_KEY missing")
        return []
    
    adjusted_start = max(start_dt, end_dt - timedelta(days=30))
    if adjusted_start > start_dt:
        logger.info("NewsAPI: Clipped to last 30 days (free tier)")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{symbol} stock OR {symbol} earnings",
        "from": adjusted_start.isoformat(),
        "to": end_dt.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }
    items = []
    page = 1
    while len(items) < 1000:
        params["page"] = page
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if data.get("status") != "ok":
                break
            for art in data.get("articles", []):
                try:
                    date_str = datetime.fromisoformat(art["publishedAt"][:10]).date().isoformat()
                    items.append({"date": date_str, "headline": art["title"], "source": "newsapi"})
                except:
                    continue
            if len(data.get("articles", [])) < 100:
                break
            page += 1
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            break
    
    logger.info(f"NewsAPI: {len(items)} headlines")
    return items


# --------------------------------------------------------------------- #
# 4. Yahoo Finance RSS – Today only
# --------------------------------------------------------------------- #
def fetch_yahoo_news() -> List[Dict]:
    """Fetch today's headlines from Yahoo Finance RSS."""
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    try:
        feed = feedparser.parse(url)
        today = date.today().isoformat()
        items = []
        for entry in feed.entries:
            if not hasattr(entry, "published_parsed"):
                continue
            try:
                pub_date = datetime(*entry.published_parsed[:6]).date().isoformat()
                if pub_date == today:
                    items.append({"date": today, "headline": entry.title, "source": "yahoo"})
            except:
                continue
        logger.info(f"Yahoo RSS: {len(items)} live headlines")
        return items
    except Exception as e:
        logger.error(f"Yahoo RSS error: {e}")
        return []


# --------------------------------------------------------------------- #
# 5. Google News RSS – Recent
# --------------------------------------------------------------------- #
def fetch_google_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]:
    """Fetch recent news from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={symbol}+stock+when:1y"
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries:
            try:
                pub_date = datetime(*e.published_parsed[:6]).date()
                if start_dt <= pub_date <= end_dt:
                    items.append({"date": pub_date.isoformat(), "headline": e.title, "source": "googlenews"})
            except:
                continue
        logger.info(f"Google News RSS: {len(items)} headlines")
        return items
    except Exception as e:
        logger.error(f"Google News RSS error: {e}")
        return []


# --------------------------------------------------------------------- #
# 6. GDELT – Last 15 days only
# --------------------------------------------------------------------- #
def fetch_gdelt_news(symbol: str, start_dt: date, end_dt: date) -> List[Dict]:
    """
    Fetch global news from GDELT (last ~15 days only).
    """
    COMPANY_MAP = {
        "NVDA": "NVIDIA", "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
        "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta",
    }
    company = COMPANY_MAP.get(symbol.upper(), symbol)
    
    # Clip to last 15 days
    today = date.today()
    max_start = today - timedelta(days=15)
    start_dt = max(start_dt, max_start)
    if start_dt > end_dt:
        logger.info("GDELT: No data (outside last 15 days)")
        return []
    
    raw_query = f'{company} stock'
    encoded_query = urllib.parse.quote(raw_query)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": encoded_query,
        "mode": "ArtList",
        "maxrecords": 250,
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d000000"),
        "enddatetime": end_dt.strftime("%Y%m%d235959"),
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = []
        for d in data.get("articles", []):
            try:
                date_str = datetime.strptime(d["seendate"][:8], "%Y%m%d").date().isoformat()
                items.append({"date": date_str, "headline": d["title"], "source": "gdelt"})
            except (KeyError, ValueError, TypeError):
                continue
        logger.info(f"GDELT: {len(items)} headlines for {symbol}")
        return items
    except requests.exceptions.RequestException as e:
        logger.error(f"GDELT request error: {e}")
        return []
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"GDELT JSON error: {e}")
        return []
    except Exception as e:
        logger.error(f"GDELT error: {e}")
        return []


# --------------------------------------------------------------------- #
# News dispatcher
# --------------------------------------------------------------------- #
def fetch_news(symbol: str, start_dt: date, end_dt: date, mode: str, sources: List[str]) -> List[Dict]:
    """Dispatch news fetching based on mode and sources."""
    news_items = []
    
    def add(items):
        if items:
            news_items.extend(items)
    
    if mode == "individual":
        src = sources[0] if sources else "finnhub"
        if src == "finnhub":
            add(fetch_finnhub_news(symbol, start_dt.isoformat(), end_dt.isoformat()))
        elif src == "alphavantage":
            add(fetch_alphavantage_news(symbol, start_dt, end_dt))
        elif src == "newsapi":
            add(fetch_newsapi_news(symbol, start_dt, end_dt))
        elif src == "yahoo":
            add(fetch_yahoo_news())
        elif src == "googlenews":
            add(fetch_google_news(symbol, start_dt, end_dt))
        elif src == "gdelt":
            add(fetch_gdelt_news(symbol, start_dt, end_dt))
        else:
            logger.warning(f"Unknown source: {src}")
    
    elif mode == "combined":
        for src in sources:
            if src == "finnhub":
                add(fetch_finnhub_news(symbol, start_dt.isoformat(), end_dt.isoformat()))
            elif src == "alphavantage":
                add(fetch_alphavantage_news(symbol, start_dt, end_dt))
            elif src == "newsapi":
                add(fetch_newsapi_news(symbol, start_dt, end_dt))
            elif src == "yahoo":
                add(fetch_yahoo_news())
            elif src == "googlenews":
                add(fetch_google_news(symbol, start_dt, end_dt))
            elif src == "gdelt":
                add(fetch_gdelt_news(symbol, start_dt, end_dt))
            else:
                logger.warning(f"Unknown source in combined: {src}")
    
    else:
        raise ValueError("sentiment_mode must be 'individual' or 'combined'")
    
    logger.info(f"Total raw headlines: {len(news_items)}")
    return news_items


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Enrich stock data with news sentiment")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()


# --------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------- #
cfg = yaml.safe_load(open(args.config))
symbol = cfg["stock_symbol"]
mode = cfg.get("sentiment_mode", "individual").lower()
sources = cfg.get("sentiment_sources", ["finnhub"])
model_name = cfg.get("sentiment_model", "ProsusAI/finbert")
force_refresh = cfg.get("force_refresh", False)
start_dt = datetime.strptime(cfg["start_date"], "%Y-%m-%d").date()
end_dt = datetime.strptime(cfg["end_date"], "%Y-%m-%d").date()

logger.info(f"Global range: {start_dt} to {end_dt} | Mode: {mode} | Sources: {sources}")

# Load raw data
raw_csv = f"{cfg['raw_dir']}/{symbol}_raw.csv"
if not os.path.exists(raw_csv):
    raise FileNotFoundError(f"Raw data not found: {raw_csv}")

df = pd.read_csv(raw_csv)
df["Date"] = pd.to_datetime(df["Date"]).dt.date
df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)
logger.info(f"Loaded {len(df)} trading days")

# Cache
cached = [] if force_refresh else load_cache(cfg, symbol)
fresh = fetch_news(symbol, start_dt, end_dt, mode, sources)
combined = merge_and_deduplicate(cached, fresh)
save_cache(cfg, symbol, combined)

# Map news to dates (max 3 headlines per day)
news_by_date = {}
for item in combined:
    news_by_date.setdefault(item["date"], []).append(item["headline"])

texts = []
for date_val in df["Date"]:
    headlines = news_by_date.get(date_val.isoformat(), [])
    text = " | ".join(headlines[:3]) if headlines else ""
    texts.append(text)

# Sentiment scoring
analyzer = SentimentAnalyzer(model_name)
df["news"] = texts
df["sentiment"] = [analyzer.score(text) for text in texts]

logger.info(f"Sentiment mean: {df['sentiment'].mean():.4f} | std: {df['sentiment'].std():.4f}")

# Save
source_tag = "combined" if mode == "combined" else sources[0]
out_csv = f"{cfg['processed_dir']}/{symbol}_sentiment_{source_tag}.csv"
os.makedirs(cfg["processed_dir"], exist_ok=True)
df.to_csv(out_csv, index=False)
logger.info(f"Enriched data saved → {out_csv}")

# Summary
pos = (df["sentiment"] > 0.1).sum()
neg = (df["sentiment"] < -0.1).sum()
neu = len(df) - pos - neg
logger.info(f"Sentiment summary: POS={pos}, NEG={neg}, NEU={neu}")

# Make variables visible in Spyder
sentiment = df["sentiment"].copy()