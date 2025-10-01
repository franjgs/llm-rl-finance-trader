# LLM-RL-Finance-Trader

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Conda](https://img.shields.io/badge/conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A Reinforcement Learning (RL) and Large Language Model (LLM)-based trading system inspired by the paper ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2310.03080). This project uses stock price data (e.g., AAPL) and sentiment analysis (via FinBERT) to train a PPO model for trading decisions. Built with Python 3.10, Gymnasium, Stable-Baselines3, and PyTorch with MPS acceleration on Apple M3 Pro.

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
This project implements a trading environment (`TradingEnv`) that uses historical stock data and sentiment scores to train a PPO agent for portfolio management. The pipeline fetches stock data (e.g., AAPL from 2023-11-16 to 2024-11-10), applies sentiment analysis using FinBERT, and trains an RL model to optimize trading strategies. The goal is to replicate the paper’s results (e.g., $14K vs. $11K for multi-stock portfolios) with a focus on single-stock (AAPL) for now.

## Features
- **Data Fetching**: Downloads historical stock data using `pandas_datareader` (Stooq) with fallback to `yfinance`.
- **Sentiment Analysis**: Processes news sentiment with FinBERT (currently dummy data; extensible to Finnhub/X).
- **RL Training**: Trains a PPO model using `stable-baselines3` in a custom `TradingEnv`.
- **Visualization**: Generates plots of stock prices, trading actions (buy/sell), and portfolio net worth (`results/aapl_trading_results.png`).
- **MPS Acceleration**: Optimized for Apple M3 Pro with PyTorch MPS support.
- **Spyder Support**: `train_model.py` runs inline for variable inspection in Spyder’s Variable Explorer.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/llm-rl-finance-trader.git
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
   yfinance
   pandas-datareader
   python-dotenv==1.0.1
   pyyaml==6.0.2
   ipykernel
   ipywidgets
   matplotlib
   ```

## Usage
1. **Fetch Stock Data**:
   Downloads AAPL data from Stooq and saves to `data/raw/AAPL.csv`.
   ```bash
   python data_fetch.py --stock AAPL --start 2023-11-16 --end 2024-11-10
   ```

2. **Process Sentiment**:
   Applies FinBERT to generate sentiment scores and saves to `data/processed/AAPL_sentiment.csv`.
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
│   └── config.yaml          # Configuración del pipeline
├── data/
│   ├── raw/                # Datos de precios (e.g., AAPL.csv)
│   └── processed/          # Datos con sentiment (e.g., AAPL_sentiment.csv)
├── models/                 # Modelos entrenados
├── results/                # Gráficos (e.g., aapl_trading_results.png)
├── src/
│   └── trading_env.py      # Entorno Gym para RL
├── data_fetch.py          # Descarga datos de precios
├── sentiment_analysis.py  # Genera sentiment con FinBERT
├── train_model.py        # Entrena modelo PPO y genera gráficos
├── requirements.txt       # Dependencias
└── README.md             # Documentación
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
- **Sentiment Scores Always 0**: Ensure FinBERT processes text correctly. Test with:
  ```python
  from transformers import pipeline
  sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0 if torch.backends.mps.is_available() else -1)
  print(sentiment_pipeline("Apple announced record-breaking quarterly earnings, boosting stock prices.")[0])
  ```
  If `neutral` or `0`, verify the text and `get_sentiment_score` logic in `sentiment_analysis.py`.
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
- **Real Sentiment**: Integrate Finnhub or X API for news data:
  ```python
  import finnhub
  finnhub_client = finnhub.Client(api_key="YOUR_API_KEY")
  news = finnhub_client.company_news('AAPL', from_date="2023-11-16", to_date="2024-11-10")
  df['news'] = [n['summary'] for n in news[:len(df)]]
  ```
- **Multi-Stock Portfolio**: Extend `TradingEnv` for stocks like LEXCX to replicate the paper’s $14K vs. $11K results.
- **Backtesting**: Use `backtrader` for metrics like Sharpe ratio.
- **Advanced Visualizations**: Add plots for cumulative returns and action frequency.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.