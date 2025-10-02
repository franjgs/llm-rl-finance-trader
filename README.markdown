# 📈 LLM-RL Finance Trader

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Conda](https://img.shields.io/badge/Conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)

Welcome to **LLM-RL Finance Trader**, a cutting-edge project that combines **Reinforcement Learning (RL)** with **Large Language Models (LLM)** to enhance stock trading strategies using financial news sentiment analysis. Inspired by the paper ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2411.11059), this project implements a PPO-based RL agent to trade stocks (e.g., AAPL) with and without sentiment data, comparing their performance using metrics like Sharpe Ratio.

This repository is part of a TFG (Trabajo de Fin de Grado) and is designed to be reproducible, portable, and optimized for macOS (Apple M3 Pro with MPS support) using a Conda environment (`llm_rl_finance`).

---

## 🚀 Features

- **Data Fetching**: Downloads historical stock data (e.g., AAPL) using `yfinance`.
- **Sentiment Analysis**: Integrates financial news from Finnhub and computes sentiment scores using FinBERT.
- **RL Trading**: Trains a PPO model with Stable-Baselines3 to trade stocks, with optional sentiment integration.
- **Performance Evaluation**: Compares trading strategies (with/without sentiment) using net worth and Sharpe Ratio.
- **Visualization**: Generates plots of stock prices, trading actions, and portfolio net worth.
- **Reproducible Workflow**: Uses relative paths and Conda for portability, with detailed logging for debugging.

---

## 📋 Prerequisites

- **OS**: macOS (optimized for Apple M3 Pro with MPS support)
- **Python**: 3.10
- **Conda**: Environment `llm_rl_finance`
- **External API**: Finnhub API key (stored in `.env`)

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/llm-rl-finance-trader.git
   cd llm-rl-finance-trader
   ```

2. **Set Up Conda Environment**:
   ```bash
   conda create -n llm_rl_finance python=3.10
   conda activate llm_rl_finance
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   pandas
   numpy
   matplotlib
   torch
   stable-baselines3
   gymnasium
   yfinance
   finnhub-python
   transformers
   python-dotenv
   pyyaml
   ```

4. **Set Up Finnhub API Key**:
   - Create a `.env` file in the project root:
     ```bash
     echo "FINNHUB_API_KEY=your_api_key" > .env
     ```
   - Obtain your API key from [Finnhub](https://finnhub.io/).

---

## 📂 Project Structure

```
llm-rl-finance-trader/
├── configs/                    # Configuration files
│   └── config.yaml             # Project settings (stock symbol, dates, etc.)
├── data/                       # Data storage
│   ├── raw/                    # Raw stock data (e.g., AAPL.csv)
│   └── processed/              # Processed data with sentiment (e.g., AAPL_sentiment.csv)
├── models/                     # Trained RL models
├── results/                    # Output plots and results
├── src/                        # Auxiliary modules
│   └── trading_env.py           # Custom Gym environment for trading
├── data_fetch.py               # Fetches stock data
├── sentiment_analysis.py       # Computes sentiment from financial news
├── train_model.py              # Trains and evaluates RL trading model
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
   start_date: 2023-11-16
   end_date: 2024-11-08
   data_path: data/processed/AAPL_sentiment.csv
   timesteps: 10000
   ```

2. **Run the Pipeline**:
   ```bash
   conda activate llm_rl_finance
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
- **Sharpe Ratio (With Sentiment)**: 0.8666
- **Sharpe Ratio (Without Sentiment)**: -0.0899
- **Output Plot**: Visualizes stock prices, buy/sell actions, and portfolio net worth.

![Trading Results](results/aapl_trading_results_comparison.png)

---

## 🐛 Troubleshooting

- **Simulation Stops Early**:
  - Check logs for `Simulation (with sentiment) stopped early at step X, date=Y. Missing dates: [...]`.
  - Inspect `src/trading_env.py` to ensure `max_steps = len(df)`.

- **Data Misalignment**:
  - Verify that `data/raw/AAPL.csv` and `data/processed/AAPL_sentiment.csv` have 247 rows:
    ```python
    import pandas as pd
    df_raw = pd.read_csv('data/raw/AAPL.csv')
    df_sentiment = pd.read_csv('data/processed/AAPL_sentiment.csv')
    print(len(df_raw), len(df_sentiment))
    ```

- **Missing API Key**:
  - Ensure `.env` contains `FINNHUB_API_KEY`.

---

## 🌟 Future Work

- **Add Metrics**: Include maximum drawdown and annualized returns.
- **LLM Integration**: Implement Module II (LLM as policy) from the paper.
- **Multi-Stock Trading**: Extend to multiple stocks for portfolio optimization.
- **Hyperparameter Tuning**: Optimize PPO hyperparameters for better performance.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Inspired by ["Financial News-Driven LLM Reinforcement Learning for Portfolio Management"](https://arxiv.org/abs/2411.11059).
- Built with ❤️ for a TFG in Machine Learning and Finance.
- Thanks to [Finnhub](https://finnhub.io/) and [Hugging Face](https://huggingface.co/) for APIs and models.

---

*Happy trading! 🚀 Questions or contributions? Open an issue or PR on GitHub!*