# LLM-RL-Finance-Trader

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Conda](https://img.shields.io/badge/conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A Reinforcement Learning (RL) and Large Language Model (LLM)-based trading system inspired by the paper ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2310.03080). This project uses stock price data (e.g., AAPL) and sentiment analysis (via Finnhub and FinBERT) to train a PPO model for trading decisions. Built with Python 3.10, Gymnasium, Stable-Baselines3, and PyTorch with MPS acceleration on Apple M3 Pro.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements a trading environment (`TradingEnv`) that uses historical stock data and sentiment scores to train a PPO agent for portfolio management. The pipeline fetches stock data (e.g., AAPL from 2023-11-16 to 2024-11-10), fetches news and applies sentiment analysis using Finnhub and FinBERT, and trains an RL model to optimize trading strategies. The goal is to replicate the paper’s results (e.g., $14K vs. $11K for multi-stock portfolios) with a focus on single-stock (AAPL) for now.

## Features
- **Data Fetching**: Downloads historical stock data using `pandas_datareader` (Stooq) with fallback to `yfinance`.
- **Sentiment Analysis**: Fetches financial news from Finnhub and processes sentiment with FinBERT.
- **RL Training**: Trains a PPO model using `stable-baselines3` in a custom `TradingEnv`.
- **Visualization**: Generates plots of stock prices, trading actions (buy/sell), and portfolio net worth (`results/aapl_trading_results.png`).
- **MPS Acceleration**: Optimized for Apple M3 Pro with PyTorch MPS support.
- **Spyder Support**: `train_model.py` runs inline for variable inspection in Spyder’s Variable Explorer.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/franjgs/llm-rl-finance-trader.git
   cd llm-rl-finance-trader
   ```

2. **Create Conda Environment**:
   ```bash
   conda create -n llm_rl_finance python=3.10
   conda activate llm_rl_finance
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt`:
   ```
   gymnasium==0.29.1
   stable-baselines3==2.3.2
   torch==2.4.1
   transformers==4.45.1
   pandas==2.2.3
   yfinance==0.2.48
   pandas-datareader==0.10.0
   python-dotenv==1.0.1
   pyyaml==6.0.2
   ipykernel==6.29.5
   ipywidgets==8.1.5
   matplotlib==3.9.2
   finnhub-python==2.4.20
   ```

4. **Configure API Keys**:
   - Create a `.env` file in the project root.
   - Add your Finnhub API key (obtain from https://finnhub.io/register):
     ```
     FINNHUB_API_KEY=your_api_key_here
     ```

## Usage
1. **Fetch Stock Data**:
   Downloads AAPL data from Stooq and saves to `data/raw/AAPL.csv`.
   ```bash
   python data_fetch.py --stock AAPL --start 2023-11-16 --end 2024-11-10
   ```

2. **Process Sentiment**:
   Fetches news from Finnhub, applies FinBERT sentiment analysis, and saves to `data/processed/AAPL_sentiment.csv`.
   ```bash
   python sentiment_analysis.py --input data/raw/AAPL.csv --output data/processed/AAPL_sentiment.csv
   ```
   In Spyder:
   - Open `sentiment_analysis.py`, set the working directory to the project root (`os.chdir('/path/to/llm-rl-finance-trader')`), and run.

3. **Train PPO Model**:
   Trains the RL model and saves to `models/trading_model`. Generates a plot in `results/aapl_trading_results.png`. Variables (`df`, `model`, `net_worth`, `actions`) are inspectable in Spyder.
   ```bash
   python train_model.py --config configs/config.yaml
   ```
   In Spyder:
   - Open `train_model.py`, set the working directory, and run line by line or the entire script to inspect variables in Variable Explorer.

4. **Visualize Results**:
   View the price and trading results plot:
   ```bash
   open results/aapl_trading_results.png
   ```

## Project Structure
```
llm-rl-finance-trader/
├── configs/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                # Raw stock data (e.g., AAPL.csv)
│   └── processed/          # Data with sentiment (e.g., AAPL_sentiment.csv)
├── models/                 # Trained models
├── results/                # Visualizations (e.g., aapl_trading_results.png)
├── src/
│   └── trading_env.py      # Custom Gym environment
├── notebooks/
│   └── quick_start.ipynb   # Interactive analysis
├── data_fetch.py           # Fetch stock data
├── sentiment_analysis.py   # Fetch news and apply sentiment analysis
├── train_model.py         # Train RL model
├── requirements.txt        # Dependencies
├── .env                    # API keys (not tracked)
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Troubleshooting
- **YFinance Errors**: If `data_fetch.py` fails with `YFTzMissingError`, it falls back to `pandas_datareader` (Stooq). Manually download from [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history) if needed.
- **FileNotFoundError**: Ensure `data/raw/AAPL.csv` and `data/processed/AAPL_sentiment.csv` exist. Set the working directory to the project root in Spyder (Tools > Current working directory).
- **TqdmWarning**: Install `ipywidgets`:
  ```bash
  pip install ipywidgets
  ```
- **ModuleNotFoundError: No module named 'gym'**: Use `import gymnasium as gym` in `trading_env.py` and ensure `gymnasium==0.29.1` is installed.
- **ValueError in TradingEnv**: Ensure `observation_space` matches `AAPL_sentiment.csv` columns (`open,high,low,close,volume,sentiment`).
- **Sentiment Scores Always 0**: Ensure Finnhub API key is set in `.env` and FinBERT processes news correctly. Test with:
  ```python
  from transformers import pipeline
  import finnhub
  from dotenv import load_dotenv
  import os
  load_dotenv()
  finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
  news = finnhub_client.company_news('AAPL', _from="2023-11-16", to="2024-11-10")
  sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0 if torch.backends.mps.is_available() else -1)
  print(sentiment_pipeline(news[0]['headline'])[0])
  ```
  If `neutral` or `0`, verify the news text and `get_sentiment_score` logic in `sentiment_analysis.py`.
- **MPS Acceleration**: Verify MPS with:
  ```python
  import torch
  print(torch.backends.mps.is_available())  # Should be True
  ```
  If `False`, reinstall PyTorch:
  ```bash
  conda install pytorch==2.4.1 -c pytorch
  ```

## Future Improvements
- **Multi-Stock Portfolio**: Extend `TradingEnv` for stocks like LEXCX to replicate the paper’s $14K vs. $11K results.
- **Backtesting**: Use `backtrader` for metrics like Sharpe ratio.
- **Advanced Visualizations**: Add plots for cumulative returns and action frequency.
- **Real-Time Sentiment**: Integrate X API for real-time news sentiment analysis.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure code follows PEP 8 and includes English docstrings/comments.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.