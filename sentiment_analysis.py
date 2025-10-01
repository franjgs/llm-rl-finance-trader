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

def get_real_news_sentiment(df, finnhub_client, start_date="2023-11-16", end_date="2024-11-10"):
    """Fetch news from Finnhub and apply FinBERT sentiment analysis.

    Args:
        df (pd.DataFrame): Input DataFrame with stock data.
        finnhub_client (finnhub.Client): Configured Finnhub client.
        start_date (str): Start date for news data (YYYY-MM-DD).
        end_date (str): End date for news data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with added 'news' and 'sentiment' columns.
    """
    # Fetch news for AAPL
    try:
        news = finnhub_client.company_news('AAPL', _from=start_date, to=end_date)
        logger.info(f"Fetched {len(news)} news articles for AAPL")
    except Exception as e:
        logger.error(f"Error fetching news from Finnhub: {e}")
        news = []
    
    # Assign news headlines, limiting to DataFrame length
    news_texts = [n['headline'] for n in news[:len(df)]]
    df['news'] = news_texts + [''] * (len(df) - len(news_texts))  # Pad with empty strings if needed

    # Initialize FinBERT pipeline
    device = 0 if torch.backends.mps.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    logger.info("Initialized FinBERT sentiment pipeline")

    def get_sentiment_score(text):
        """Compute sentiment score for a given text using FinBERT.

        Args:
            text (str): Text to analyze (e.g., news headline).

        Returns:
            float: Sentiment score (positive: >0, negative: <0, neutral: ~0).
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
                return result['score'] * 0.5  # Neutral scores scaled down
        except Exception as e:
            logger.warning(f"Error processing sentiment for text '{text}': {e}")
            return 0.0

    # Apply sentiment analysis
    df['sentiment'] = df['news'].apply(get_sentiment_score)
    logger.info(f"Computed sentiment for {len(df)} rows")
    return df

def process_sentiment(input_path, output_path):
    """Process stock data to add sentiment analysis based on Finnhub news.

    Args:
        input_path (str): Path to input CSV file with stock data.
        output_path (str): Path to save output CSV with sentiment data.
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

    # Add sentiment data
    df = get_real_news_sentiment(df, finnhub_client)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sentiment data to {output_path}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process stock data with Finnhub news sentiment analysis")
parser.add_argument("--input", default="data/raw/AAPL.csv", help="Path to input stock data CSV")
parser.add_argument("--output", default="data/processed/AAPL_sentiment.csv", help="Path to output CSV")
args = parser.parse_args()

# Process sentiment data
process_sentiment(args.input, args.output)