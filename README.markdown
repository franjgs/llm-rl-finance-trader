# LLM-RL-Finance-Trader

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Conda](https://img.shields.io/badge/conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A hybrid trading system combining Large Language Models (LLM) for static financial news sentiment analysis and Reinforcement Learning (RL) for dynamic trading strategy optimization. Inspired by the paper ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2411.11059). Built with Python 3.10, Gymnasium, Stable-Baselines3, and PyTorch with MPS acceleration on Apple M3 Pro.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements a hybrid trading system where a Large Language Model (FinBERT) processes financial news from Finnhub to generate static sentiment scores, which are then used by a Reinforcement Learning agent (PPO) in a custom `TradingEnv` to dynamically optimize trading decisions. The pipeline fetches stock data (e.g., AAPL from 2023-11-16 to 2024-11-10), generates sentiment scores, and trains the RL model to maximize portfolio value, aiming to replicate the paper’s results (e.g., $14K vs. $11K for multi-stock portfolios) with a focus on single-stock (AAPL) for now.

## Architecture
The system follows a hybrid LLM + RL architecture:
- **Static LLM Component**: FinBERT processes financial news from Finnhub to generate sentiment scores, providing a static input that captures market sentiment for each trading day.
- **Dynamic RL Component**: A PPO agent uses the sentiment scores alongside stock price data (open, high, low, close, volume) in a custom `TradingEnv` to make dynamic trading decisions (buy, sell, hold).

This separation ensures that the LLM handles static analysis of textual data, while the RL agent adapts dynamically to market conditions.

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

## Future Improvements
- **Enhanced LLM Preprocessing**: Use more advanced LLMs (e.g., BERT variants or GPT-based models) to extract richer features from news, such as sentiment intensity or event relevance, to improve static inputs for RL.
- **Dynamic RL Strategies**: Implement advanced RL algorithms (e.g., SAC or DDPG) to handle continuous action spaces and improve trading decision robustness.
- **Hybrid Integration**: Develop a feedback loop where RL decisions influence LLM input selection (e.g., prioritizing news based on market volatility).
- **Multi-Stock Portfolio**: Extend `TradingEnv` to handle multiple stocks (e.g., LEXCX) with LLM-derived sentiment scores for each, replicating the paper’s $14K vs. $11K results.
- **Backtesting with Hybrid Metrics**: Use `backtrader` to evaluate combined LLM+RL performance with metrics like Sharpe ratio, incorporating sentiment impact.

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