import argparse
import pandas as pd
import logging
import os
import torch
from transformers import pipeline
from dotenv import load_dotenv
import finnhub
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.

    Raises:
        yaml.YAMLError: If the YAML file is invalid.
        Exception: For other file loading errors.
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            logger.info(f"YAML content:\n{content}")
            return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

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
        # Adjust end_date to avoid future dates
        today = datetime.now().strftime('%Y-%m-%d')
        if end_date > today:
            logger.warning(f"end_date {end_date} is in the future; adjusting to {today}")
            end_date = today
        news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        logger.info(f"Fetched {len(news)} news articles for {symbol} from {start_date} to {end_date}")
        return [n['headline'] for n in news if 'headline' in n]
    except Exception as e:
        logger.error(f"Error fetching news from Finnhub for {symbol}: {e}")
        return []

def compute_sentiment(news_texts):
    """Compute sentiment scores using FinBERT for a list of news texts.

    Args:
        news_texts (list): List of news headlines or empty strings.

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
        if not text or pd.isna(text):
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

def process_sentiment(input_path, output_path, symbol, start_date, end_date):
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
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Loaded data from {input_path} with {len(df)} rows")
        logger.info(f"Stock data date range: {df['Date'].min()} to {df['Date'].max()}")
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

    # Verify date alignment
    expected_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    missing_dates = expected_dates.difference(df['Date'])
    if len(missing_dates) > 0:
        logger.warning(f"Missing {len(missing_dates)} dates in stock data: {missing_dates}")
    logger.info(f"Output DataFrame columns: {df.columns.tolist()}")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sentiment data to {output_path}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate static sentiment scores using Finnhub and FinBERT")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
input_path = f"data/raw/{config['stock_symbol']}.csv"
output_path = config['data_path']
symbol = config['stock_symbol']
start_date = config['start_date']
end_date = config['end_date']

# Process sentiment data
process_sentiment(input_path, output_path, symbol, start_date, end_date)