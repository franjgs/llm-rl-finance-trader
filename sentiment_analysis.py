import pandas as pd
import argparse
import os
from transformers import pipeline
import torch

# Usa MPS si está disponible
device = 0 if torch.backends.mps.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

def get_sentiment_score(text):
    try:
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text input: {text}")
            return 0
        result = sentiment_pipeline(text)[0]
        print(f"Sentiment result for '{text}': {result}")  # Debug
        score = result['score']
        label = result['label']
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:  # neutral
            return score * 0.5  # Escala neutral
    except Exception as e:
        print(f"Error processing sentiment for text '{text}': {e}")
        return 0

def process_sentiment(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {input_path} with columns: {df.columns.tolist()}")
        df['news'] = "Apple announced record-breaking quarterly earnings, boosting stock prices."
        df['sentiment'] = df['news'].apply(get_sentiment_score)
        print(f"Sentiment scores (first 5): {df['sentiment'].head().tolist()}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error in process_sentiment: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/AAPL.csv")
    parser.add_argument("--output", default="data/processed/AAPL_sentiment.csv")
    args = parser.parse_args()
    process_sentiment(args.input, args.output)