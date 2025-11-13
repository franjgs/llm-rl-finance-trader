# 📈 LLM-RL Finance Trader

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Conda](https://img.shields.io/badge/Conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)

Welcome to **LLM-RL Finance Trader**, a hybrid research project that integrates static AI models (specifically, **FinBERT** for feature extraction) with dynamic **Reinforcement Learning (RL)** to optimize a trading strategy.

Inspired by academic work on financial news-driven RL, this repository implements a **PPO**-based agent to manage assets (e.g., AAPL) by comparing performance **with and without** an enhanced **sentiment feature**.

This repository serves as the core implementation for a TFG (Trabajo de Fin de Grado), emphasizing reproducibility, portability, and optimization for macOS (Apple Silicon MPS support).

The project now supports **walk-forward training**, a realistic rolling-window approach for time-series data, to better simulate real-world trading and reduce overfitting.

---

## 🚀 Features

### 🧠 Reinforcement Learning (RL)
* Trains a **PPO** agent (Stable-Baselines3) using a custom `TradingEnv`.
* Compares two strategies:
    * **Baseline:** Uses only **Price and Volume** data.
    * **Enhanced:** Adds the daily **Sentiment Score** feature.
* Performance evaluated using **Sharpe Ratio**, **Net Worth**, and **Drawdown**.

### Walk-Forward Training
* **Rolling-window optimization**: Trains on all historical data up to the current day, predicts the next 1 day, then advances.
* **Warm start**: Loads the previous day's best model and incrementally trains (e.g., 1,000 timesteps per day after initial 20,000).
* **Configurable sentiment**: "with", "without", or "both" for comparison.
* **Speedup**: ~20× faster than full re-training each day.
* **Real-world simulation**: Mimics daily trading with continuous learning from new data.

---

### 📰 Sentiment Analysis Pipeline (`sentiment_analysis.py`)
This module aggregates market sentiment by combining data from multiple sources.

* **Source Aggregation**: Collects and merges financial news from **six sources** including **Finnhub**, **Alpha Vantage** (historical, batched), **GDELT** (global), **NewsAPI**, **Google News RSS**, and **Yahoo Finance**.
* **Model**: Uses the pre-trained **FinBERT** (`ProsusAI/finbert`) model for robust financial sentiment classification.
* **Hardware Acceleration**: Optimized for fast local inference using **MPS (Apple Silicon)**.
* **Output**: Converts textual sentiment into a single **numeric value (-1.0 → +1.0)** per trading day.
* **Caching**: Includes a **cache system** with **deduplication** by `(headline, source)` to minimize API calls and ensure data consistency.

---

### 🔄 Data Workflow
The project runs sequentially through three main scripts defined in `configs/config.yaml`.

1.  **`data_fetch.py`**: Downloads historical stock data (e.g., AAPL) using `yfinance` based on the dates defined in the config. **Output**: `data/raw/<symbol>_raw.csv`.
2.  **`sentiment_analysis.py`**: Fetches news, computes FinBERT sentiment, and joins the feature with the historical data. **Output**: `data/processed/<symbol>_sentiment_<source>.csv`.
3.  **`train_model.py`**: Loads the enriched data, trains two PPO agents (with/without sentiment), simulates their performance, and generates the final comparison plot.
4. **`train_walk_forward.py`**: Performs walk-forward training with warm-start and incremental learning. **Output**: 
    * Daily best models in `models/best_walk_forward/`
    * Learning curves in `results/walk_forward/`
    * Final equity curves in `results/walk_forward/*_1day_with.csv` and `results/walk_forward/*_1day_without.csv`
    * Final plot via `plot_results()` (walk-forward mode)

---

## 📋 Prerequisites

* **OS**: macOS (optimized for **Apple Silicon** with MPS support)
* **Python**: **3.11**
* **Conda**: Environment `llm_rl_finance`
* **External API Keys**: **Finnhub**, Alpha Vantage, and NewsAPI keys (stored in `.env`).

---

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/llm-rl-finance-trader.git](https://github.com/your-username/llm-rl-finance-trader.git)
    cd llm-rl-finance-trader
    ```

2.  **Set Up Conda Environment**:
    ```bash
    conda create -n llm_rl_finance python=3.11
    conda activate llm_rl_finance
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys**:
    - Create a `.env` file in the project root.
    - Add your required API keys (e.g., Finnhub) to this file:
    ```bash
    echo "FINNHUB_API_KEY=your_finnhub_key" > .env
    echo "ALPHAVANTAGE_API_KEY=your_av_key" >> .env
    echo "NEWSAPI_KEY=your_newsapi_key" >> .env
    ```

---

## 📂 Project Structure

```
llm-rl-finance-trader/
├── configs/                    # Configuration files
│   └─ config.yaml              # Project settings (stock symbol, dates, etc.)
│   └─ config_walk_forward.yaml # Project settings (stock symbol, dates, etc.)
├── data/                       # Data storage
│   ├── raw/                    # Raw stock data (e.g., AAPL.csv)
│   └── processed/              # Processed data with sentiment (e.g., AAPL_sentiment.csv)
│   └── cache/                  # Cached news with sentiment (e.g., cache_AAPL.json)
├── models/                     # Trained RL models
├── results/                    # Output plots and results
├── src/                        # Auxiliary modules
│   └── trading_env.py          # Custom Gym environment for trading
│   └── rl_utils.py		# Customize LSTM Policy
│   └── plot_utils.py		# Plot Utils
├── data_fetch.py               # Fetches stock data
├── sentiment_analysis.py       # Computes sentiment from financial news
├── train_model.py              # Trains and evaluates RL trading model
├── train_walk_forward.py       # Walk Forward RL trading model
├── .env                        # Environment variables (Finnhub API key)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🖥️ Usage

1. **Configure the Project**:
   Edit `configs/config.yaml`:
   ```yaml
   stock_symbol: AAPL
   # Capital
   initial_balance: 10000

   start_date: 2023-11-16
   end_date: 2024-11-08
   # Base directories
   raw_dir: data/raw
   processed_dir: data/processed
   cache_dir: data/cache
   timesteps: 10000

   # Sentiment configuration
   sentiment_mode: combined  # "individual" or "combined"
   ```

2. **Run the Pipeline**:
   ```bash
   conda activate Trader
   python data_fetch.py --config configs/config.yaml
   python sentiment_analysis.py --config configs/config.yaml
   python train_model.py --config configs/config.yaml
   ```

3. **Check Outputs**:
   - **Data**: `data/processed/AAPL_sentiment.csv` (stock data with sentiment)
   - **Models**: `models/trading_model_with_sentiment.zip`, `models/trading_model_without_sentiment.zip`
   - **Results**: `results/aapl_trading_results.csv` (trading performance)
   - **Plots**: `results/aapl_trading_results_comparison.png` (visualization of prices and net worth)

4. **Debugging**:
   - Check logs for errors or missing dates:
     ```bash
     cat logs/train_model.log | grep "Missing dates"
     ```
   - Verify data alignment:
     ```python
     import pandas as pd
     df = pd.read_csv('data/processed/AAPL_sentiment.csv')
     results_df = pd.read_csv('results/aapl_trading_results.csv')
     print(df['Date'].equals(results_df['Date']))
     ```

---

## 📊 Results

The project compares two RL trading strategies for AAPL (16/11/2023 - 10/11/2024):
- **With Sentiment**: Uses FinBERT sentiment scores from financial news.
- **Without Sentiment**: Relies solely on price and volume data.

Key metrics (from logs):
- **Sharpe Ratio (With Sentiment)**: 0.6627
- **Sharpe Ratio (Without Sentiment)**: 0.2501
- **Output Plot**: Visualizes stock prices, buy/sell actions, and portfolio net worth.

![Trading Results](results/aapl_trading_results_comparison.png)

---

## 🐛 Troubleshooting

* **API Rate Limits**: If errors occur during `sentiment_analysis.py`, check the logs. Set `force_refresh: false` in `config.yaml` to utilize the cache and avoid re-fetching headlines already stored in `cache_SYMBOL.json`.
* **Early Simulation Stop**: If `train_model.py` logs a warning about the simulation stopping early, it indicates the custom environment (`trading_env.py`) reached its `max_steps` limit prematurely due to a data-related issue. Check the alignment of the input CSV.
* **Dependency Issues (MPS)**: Ensure your `torch` installation supports MPS if you are on Apple Silicon.

---

## 🌟 Future Work

* **Advanced Metrics**: Integrate Maximum Drawdown and Annualized Returns into the final evaluation script.
* **LLM Policy Integration**: Implement the **LLM as Policy** module described in the inspiring paper.
* **Portfolio Optimization**: Extend the environment to handle trading in **multiple stocks** simultaneously.
* **Hyperparameter Search**: Implement a framework (e.g., Optuna) for systematic **hyperparameter tuning** of the PPO agent.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Inspired by ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2411.11059).
- Built for a TFG in Machine Learning and Finance.
- Revise [Finnhub](https://finnhub.io/) and [Hugging Face](https://huggingface.co/) for APIs and models.
