# LLM-RL Finance Trader + Statistical Ensemble

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Conda](https://img.shields.io/badge/Conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Branch](https://img.shields.io/badge/branch-ensemble-success.svg)

Hybrid research repository that combines two state-of-the-art approaches:

- **main / develop** → Reinforcement Learning (PPO) + FinBERT sentiment (original TFG)
- **ensemble** → Production-grade statistical + ML ensemble (momentum, volatility targeting, XGBoost, LSTM, sentiment signal, RL-style risk overlay)

Both pipelines share the same data, sentiment engine and results folders, so you can compare RL vs. classical quant strategies on the exact same dataset.

---

## 🚀 **Ensemble features**

### Core Components
- Time-Series Momentum (Moskowitz, Ooi & Pedersen 2012)
- Volatility Targeting & Risk-Parity scaling (Harvey & Liu 2015)
- XGBoost tabular classifier
- LSTM sequence predictor
- Daily FinBERT sentiment signal (your existing pipeline, forward-filled)
- RL-style dynamic risk overlay (drawdown-based exposure scaling)
- Automatic day → bar conversion (1h, 30m, 15m, 5m, daily)
- Realistic costs (commission + slippage)
- Full metrics: Sharpe, CAGR, Max Drawdown, Outperformance vs Buy & Hold

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
3.  **`ensemble.py`**: Loads the enriched data, trains 

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
├── configs/
│   ├── config.yaml                     # Legacy RL config (kept for compatibility)
│   ├── config_walk_forward.yaml        # Legacy walk-forward config
│   └── config_ensemble.yaml            # Main ensemble configuration (active)
├── data/                               # All data (ignored via .gitignore)
│   ├── raw/
│   ├── processed/
│   └── cache/
├── models/                             # Trained models (ignored via .gitignore)
│   ├── rl/
│   │   └── best_walk_forward/
│   └── ensemble/                       # Statistical + ML models
│       ├── xgb_ensemble.joblib
│       └── lstm_ensemble.pth
├── results/                            # Plots and reports
│   ├── rl/
│   └── ensemble/
│       └── NVDA_1h_ensemble_2025.png
├── src/
│   ├── __init__.py
│   ├── gen_utils.py                    # load_config(), shared utilities
│   ├── logging_config.py               # Centralized logging
│   ├── intraday_utils.py               # Intraday utilities
│   ├── metrics.py                      # Performance metrics
│   ├── plot_utils.py                   # Plotting utilities
│   ├── features.py                     # Shared feature engineering
│   ├── trading_env.py                  # Original Gym environment (RL)
│   │
│   ├── ensemble/                       # NEW: Unified ensemble module
│   │   ├── __init__.py
│   │   └── ensemble_model.py           # Single source of truth: replaces ensemble_core.py
│   │
│   └── models/                         # Individual model implementations
│       ├── __init__.py
│       ├── momentum.py
│       ├── volatility_targeting.py
│       ├── xgboost_model.py            # or xgboost_predictor.py
│       ├── lstm_model.py
│       ├── sentiment_signal.py
│       └── rl_risk_overlay.py

├── data_fetch.py                       # Enhanced data downloader
├── sentiment_analysis.py               # Sentiment pipeline (shared)
├── ensemble.py                         # Main execution script (Spyder-first, no main())
├── requirements.txt
├── .env                                # API keys (gitignored)
├── .gitignore
├── README.md
└── LICENSE                             # MIT
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
