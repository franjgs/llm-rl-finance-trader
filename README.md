# LLM-RL Finance Trader

Proyecto híbrido que integra Large Language Models (LLM) para análisis de sentiment en noticias financieras con Reinforcement Learning (RL) para optimizar estrategias de inversión y portafolios. Inspirado en el paper "Financial News-Driven LLM Reinforcement Learning for Portfolio Management" (arXiv 2411.11059v1).

## Features
- Fetch datos de precios (yfinance) y noticias (Finnhub o similar).
- Sentiment analysis con LLM (FinBERT via Hugging Face).
- Entorno RL custom (Gymnasium) con actions continuas y rewards sentiment-aligned.
- Entrenamiento con PPO (Stable Baselines3).
- Soporte para single-stock (e.g., AAPL) y portafolios (e.g., LEXCX-like).

## Instalación
1. Clona el repo: `git clone https://github.com/tu-usuario/llm-rl-finance-trader.git`
2. Crea entorno virtual: `python -m venv venv` y activa: `source venv/bin/activate` (Linux/Mac) o `venv\Scripts\activate` (Windows).
3. Instala dependencias: `pip install -r requirements.txt`
4. Configura APIs: Agrega keys para yfinance/Finnhub en .env (usa python-dotenv).

## Uso
- Fetch datos: `python src/data_fetch.py --stock AAPL --start 2023-11-16 --end 2024-11-10`
- Procesa sentiment: `python src/sentiment_analysis.py --input data/raw/AAPL.csv --output data/processed/AAPL_sentiment.csv`
- Entrena: `python src/train_model.py --config configs/config.yaml`
- Explora en notebook: Abre `notebooks/quick_start.ipynb` en Jupyter.

## Contribuciones
Bienvenidas! Abre issues/PRs. Enfocado en proyectos ML-finanzas: agrega features como backtesting con Zipline o integración con X para sentiment real-time.

## Licencia
MIT
