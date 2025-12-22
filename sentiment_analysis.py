# src/sentiment_analysis.py
# Run in Spyder → df, sentiment visible
# 3 sources: finnhub (recent), alphavantage (historical), yahoo (live)

import argparse
import yaml
import pandas as pd
import logging
import os
import torch
import requests
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import finnhub
from datetime import datetime, timedelta
import numpy as np

# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {cfg}")
    return cfg

# --------------------------------------------------------------------- #
# 2. Finnhub – Recent news (last 30 days)
# --------------------------------------------------------------------- #
def setup_finnhub_client():
    load_dotenv()
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise ValueError("FINNHUB_API_KEY missing in .env")
    return finnhub.Client(api_key=key)

def fetch_finnhub_news(client, symbol, start, end):
    """Fetch recent company news (free tier: ~30 days)."""
    today = datetime.now().strftime("%Y-%m-%d")
    if end > today:
        end = today
    try:
        raw = client.company_news(symbol, _from=start, to=end)
        items = []
        for entry in raw:
            if isinstance(entry, dict) and "headline" in entry:
                try:
                    date = datetime.strptime(entry["datetime"][:10], "%Y-%m-%d").date()
                    items.append({"date": date, "headline": entry["headline"]})
                except:
                    continue
        logger.info(f"Finnhub: {len(items)} recent headlines")
        return items
    except Exception as e:
        logger.error(f"Finnhub error: {e}")
        return []

# --------------------------------------------------------------------- #
# 3. Alpha Vantage – Historical news (2023-2024)
# --------------------------------------------------------------------- #
def fetch_alphavantage_news(symbol, start_dt, end_dt):
    """Fetch historical news from Alpha Vantage."""
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        logger.warning("ALPHAVANTAGE_API_KEY missing")
        return []

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": start_dt.strftime("%Y%m%dT0000"),
        "time_to": end_dt.strftime("%Y%m%dT2359"),
        "limit": 200,
        "apikey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "feed" not in data:
            logger.warning(f"Alpha Vantage: {data.get('Note', 'No data')}")
            return []
        items = []
        for item in data["feed"]:
            try:
                date = datetime.strptime(item["time_published"][:8], "%Y%m%d").date()
                items.append({"date": date, "headline": item["title"]})
            except:
                continue
        logger.info(f"Alpha Vantage: {len(items)} historical headlines")
        return items
    except Exception as e:
        logger.error(f"Alpha Vantage error: {e}")
        return []

# --------------------------------------------------------------------- #
# 4. Yahoo Finance – Live news (today only)
# --------------------------------------------------------------------- #
def fetch_yahoo_news():
    """Fetch live market news from Yahoo RSS (today only)."""
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    try:
        feed = feedparser.parse(url)
        today = datetime.now().date()
        items = []
        for entry in feed.entries:
            if not hasattr(entry, "published_parsed"):
                continue
            try:
                pub_date = datetime(*entry.published_parsed[:6]).date()
                if pub_date == today:
                    items.append({"date": today, "headline": entry.title})
            except:
                continue
        logger.info(f"Yahoo RSS: {len(items)} live headlines")
        return items
    except Exception as e:
        logger.error(f"Yahoo RSS error: {e}")
        return []

# --------------------------------------------------------------------- #
# 5. FinBERT – Sentiment
# --------------------------------------------------------------------- #
def compute_finbert_sentiment(texts):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"FinBERT on {device}")

    def score(t):
        if not t: return 0.0
        try:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            pos, neg, neu = probs[1].item(), probs[0].item(), probs[2].item()
            return pos if pos > max(neg, neu) else -neg if neg > max(pos, neu) else neu * 0.5
        except:
            return 0.0
    return [score(t) for t in texts]

# --------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

symbol = cfg["stock_symbol"]
start_str = cfg["start_date"]
end_str = cfg["end_date"]


raw_dir = cfg["raw_dir"]
raw_csv = f"{raw_dir}/{cfg['stock_symbol']}.csv"
source = cfg.get("sentiment_source", "finnhub").lower()

processed_dir = cfg["processed_dir"]
out_csv = f"{processed_dir}/{cfg['stock_symbol']}_sentiment_{source}.csv"

start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

df = pd.read_csv(raw_csv)
df["Date"] = pd.to_datetime(df["Date"]).dt.date
df = df.sort_values("Date").reset_index(drop=True)
logger.info(f"Loaded {len(df)} trading days")

# --- Fetch news ---
if source == "finnhub":
    client = setup_finnhub_client()
    news_items = fetch_finnhub_news(client, symbol, start_str, end_str)
elif source == "alphavantage":
    news_items = fetch_alphavantage_news(symbol, start_dt, end_dt)
elif source == "yahoo":
    news_items = fetch_yahoo_news()
else:
    raise ValueError("sentiment_source must be 'finnhub', 'alphavantage', or 'yahoo'")

# --- Map to dates ---
news_by_date = {}
for item in news_items:
    news_by_date.setdefault(item["date"], []).append(item["headline"])

texts = []
for date in df["Date"]:
    h = news_by_date.get(date, [])
    text = " | ".join(h[:3]) if h else ""
    texts.append(text)

df["news"] = texts
df["sentiment"] = compute_finbert_sentiment(texts)
logger.info(f"Sentiment mean: {df['sentiment'].mean():.4f}")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
logger.info(f"Saved → {out_csv}")