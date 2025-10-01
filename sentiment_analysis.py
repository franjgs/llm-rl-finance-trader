import argparse
import pandas as pd
import logging
import os
import torch
from transformers import pipeline
from dotenv import load_dotenv
import finnhub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_finnhub_client():
    """Initialize Finnhub client with API key from environment variables.

    Returns:
        finnhub.Client: Configured Finnhub client.

    Raises:
        ValueError: If FINNHUB_API_KEY is not set in .env.
    """
    load_dotenv()
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        logger.error("FINNHUB_API_KEY not found in .env file")
        raise ValueError("FINNHUB_API_KEY not found in .env file")
    return finnhub.Client(api_key=api_key)

def fetch_news(finnhub_client, symbol, start_date, end_date):
    """Fetch financial news for a given stock symbol from Finnhub.

    Args:
        finnhub_client (finnhub.Client): Configured Finnhub client.
        symbol (str): Stock symbol (e.g., 'AAPL').
        start_date (str): Start date for news data (YYYY-MM-DD).
        end_date (str): End date for news data (YYYY-MM-DD).

    Returns:
        list: List of news headlines.
    """
    try:
        news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        logger.info(f"Fetched {len(news)} news articles for {symbol}")
        return [n['headline'] for n in news]
    except Exception as e:
        logger.error(f"Error fetching news from Finnhub for {symbol}: {e}")
        return []

def compute_sentiment(news_texts):
    """Compute sentiment scores using FinBERT for a list of news texts.

    Args:
        news_texts (list): List of news headlines.

    Returns:
        list: List of sentiment scores (positive: >0, negative: <0, neutral: ~0).
    """
    device = 0 if torch.backends.mps.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    logger.info("Initialized FinBERT sentiment pipeline")

    def get_sentiment_score(text):
        """Compute sentiment score for a single text.

        Args:
            text (str): Text to analyze (e.g., news headline).

        Returns:
            float: Sentiment score.
        """
        if not text:
            return 0.0
        try:
            result = sentiment_pipeline(text)[0]
            if result['label'] == 'positive':
                return result['score']
            elif result['label'] == 'negative':
                return -result['score']
            else:
                return result['score'] * 0.5
        except Exception as e:
            logger.warning(f"Error processing sentiment for text '{text}': {e}")
            return 0.0

    return [get_sentiment_score(text) for text in news_texts]

def process_sentiment(input_path, output_path, symbol='AAPL', start_date='2023-11-16', end_date='2024-11-10'):
    """Process stock data to generate static sentiment scores using Finnhub and FinBERT.

    Args:
        input_path (str): Path to input CSV file with stock data.
        output_path (str): Path to save output CSV with sentiment data.
        symbol (str): Stock symbol (e.g., 'AAPL').
        start_date (str): Start date for news data (YYYY-MM-DD).
        end_date (str): End date for news data (YYYY-MM-DD).
    """
    # Load stock data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data from {input_path} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        raise

    # Initialize Finnhub client
    finnhub_client = setup_finnhub_client()

    # Fetch news and compute sentiment
    news_texts = fetch_news(finnhub_client, symbol, start_date, end_date)
    df['news'] = news_texts + [''] * (len(df) - len(news_texts))  # Pad with empty strings
    df['sentiment'] = compute_sentiment(df['news'])
    logger.info(f"Computed sentiment for {len(df)} rows")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sentiment data to {output_path}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate static sentiment scores using Finnhub and FinBERT")
parser.add_argument("--input", default="data/raw/AAPL.csv", help="Path to input stock data CSV")
parser.add_argument("--output", default="data/processed/AAPL_sentiment.csv", help="Path to output CSV")
parser.add_argument("--symbol", default="AAPL", help="Stock symbol for news fetching")
parser.add_argument("--start", default="2023-11-16", help="Start date for news (YYYY-MM-DD)")
parser.add_argument("--end", default="2024-11-10", help="End date for news (YYYY-MM-DD)")
args = parser.parse_args()

# Process sentiment data
process_sentiment(args.input, args.output, args.symbol, args.start, args.end)