# 📈 LLM-RL Finance Trader

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Conda](https://img.shields.io/badge/Conda-llm_rl_finance-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)

**LLM-RL Finance Trader** integrates financial sentiment analysis and reinforcement learning (RL) to optimize stock trading.  
It extends traditional price-based strategies with **news-derived sentiment features**, improving decision-making robustness.

---

## 🚀 Features

### 📰 Sentiment Analysis (`sentiment_analysis.py`)
- Collects and merges financial news from **six sources**:
  - 🧠 **Finnhub** – Recent company news (30 days)
  - 📊 **Alpha Vantage** – Historical sentiment (up to 2 years, batched)
  - 🌍 **GDELT** – Global news mentions
  - 📰 **NewsAPI** – General financial headlines (30-day free tier)
  - 💬 **Google News** – Public RSS search
  - 💼 **Yahoo Finance** – Daily live headlines
- Supports **individual** or **combined** fetching mode.
- Uses **FinBERT** (`ProsusAI/finbert`) for sentiment classification.
- **MPS (Apple Silicon)** acceleration for local inference.
- Converts textual sentiment into **numeric values (-1.0 → +1.0)**.
- **Cache system** to avoid refetching; deduplication by `(headline, source)`.
- Optional **`force_refresh`** flag to rebuild cache entirely.
- Saves enriched stock data to:
  ```
  data/processed/<symbol>_sentiment_<source>.csv
  ```

### 📊 RL Trading
- Trains a **PPO** agent (Stable-Baselines3) using:
  - Price and volume data
  - Optional sentiment feature
- Evaluates with **Sharpe Ratio**, **net worth**, and **drawdown**.

### 📉 Data Workflow
- `data_fetch.py`: Downloads historical stock data from Yahoo Finance.
- `sentiment_analysis.py`: Adds sentiment signals to each trading day.
- `train_model.py`: Trains RL agents and compares performance.

---

## 📋 Requirements

```bash
conda create -n llm_rl_finance python=3.11
conda activate llm_rl_finance

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers finnhub-python feedparser requests python-dotenv pyyaml pandas numpy scikit-learn matplotlib stable-baselines3 gymnasium
```

Optional (Apple Silicon GPU support):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps
```

---

## ⚙️ Configuration (`configs/config.yaml`)

Example configuration file:

```yaml
stock_symbol: "AAPL"
sentiment_mode: "combined"  # "individual" or "combined"
sentiment_sources: ["finnhub", "alphavantage", "newsapi", "yahoo", "googlenews", "gdelt"]
sentiment_model: "ProsusAI/finbert"
force_refresh: false
start_date: "2023-01-01"
end_date: "2025-01-01"
raw_dir: "data/raw"
processed_dir: "data/processed"
```

---

## 🧠 Output Example

The script enriches stock data with a daily sentiment score:

| Date       | Open  | Close | Volume | news | sentiment |
|-------------|-------|-------|--------|------|------------|
| 2024-10-01  | 180.2 | 182.4 | 73.4M  | Apple stock rises... | 0.78 |
| 2024-10-02  | 182.1 | 181.9 | 68.1M  | Analysts warn... | -0.43 |

---

## 📦 Output Files

```
data/processed/AAPL_sentiment_combined.csv
data/cache/cache_AAPL.json
```

---

## 🧩 License
MIT License © 2025 Fran J. Glez
