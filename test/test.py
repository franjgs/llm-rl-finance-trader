# test.py
"""
Quick integration test for the core ensemble components.
All comments and docstrings in English.
Runs perfectly in Spyder – no extra dependencies needed.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Core models
from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target
from src.models.sentiment_signal import SentimentSignal
from src.models.rl_risk_overlay import RLRiskOverlay


# ----------------------------------------------------------------------
# 1. Download price data
# ----------------------------------------------------------------------
print("\nDownloading NVDA daily data...")
df_price = yf.download(
    tickers="NVDA",
    start="2022-01-01",
    end="2024-01-01",
    interval="1d",
    progress=False,
    auto_adjust=True,
    threads=False,
)

df_price = df_price[["Close"]].rename(columns={"Close": "close"})
df_price.index.name = "Date"
df_price = df_price.reset_index()
df_price["Date"] = pd.to_datetime(df_price["Date"])
df_price = df_price.set_index("Date")
print(f"   → {len(df_price)} rows loaded")


# ----------------------------------------------------------------------
# 2. Momentum signal
# ----------------------------------------------------------------------
print("\nGenerating momentum signal...")
mom_params = {
    "lookback_bars": 20,
    "vol_lookback_bars": 20,
    "use_log_returns": True,
    "target_vol": 0.02,
    "smoothing_alpha": 0.12,
    "min_abs_mom": 0.0,
    "max_exposure": 1.0,
    "return_raw": True,
}

df_mom = generate_momentum_signal(df_price.reset_index(), mom_params)
df_mom = df_mom.set_index("Date")
df_mom["clean_signal"] = df_mom["signal_momentum"]
print(f"   → Momentum signal ready (last value: {df_mom['clean_signal'].iloc[-1]:.4f})")


# ----------------------------------------------------------------------
# 3. Volatility targeting
# ----------------------------------------------------------------------
print("\nApplying volatility targeting...")
vt_params = {
    "target_vol": 0.20,
    "vol_lookback_bars": 20,
    "max_leverage": 3.0,
    "min_vol": 1e-6,
    "max_scale": 5.0,
    "return_raw": True,
}

df_vt = apply_vol_target(df_mom.copy(), vt_params)
print(f"   → Volatility targeting applied (last exposure: {df_vt['exposure'].iloc[-1]:.4f})")


# ----------------------------------------------------------------------
# 4. Neutral sentiment signal
# ----------------------------------------------------------------------
print("\nAdding neutral sentiment signal...")
df_vt["sentiment"] = 0.10  # neutral example

sentiment_model = SentimentSignal({
    "pos_threshold": 0.35,
    "neg_threshold": -0.25,
    "smoothing_window": 3,
    "output_type": "raw",
    "scale_min": -1.0,
    "scale_max": 1.0,
    "normalize": False,
})
df_vt = sentiment_model.apply(df_vt)


# ----------------------------------------------------------------------
# 5. Simulate equity curve (CRITICAL FIX – force 1D array)
# ----------------------------------------------------------------------
print("\nApplying RL risk overlay (simulated equity)...")

# Force a clean 1-dimensional Series with exactly the same index as df_vt
sim_returns = df_vt["close"].pct_change().fillna(0) * 0.001
equity_values = (1 + sim_returns).cumprod() * 1_000_000

# The key line – .values.ravel() guarantees a 1D numpy array
equity = pd.Series(
    equity_values.values.ravel(),   # ← this eliminates the (n,1) shape
    index=df_vt.index,
    name="equity"
)

rl_overlay = RLRiskOverlay({
    "drawdown_reduce_level": -0.08,
    "drawdown_flat_level": -0.15,
    "sharpe_boost_threshold": 1.2,
    "sharpe_lookback_days": 60,
    "max_boost": 1.3,
    "reduce_factor": 0.5,
    "store_multiplier": True,
})

df_final = rl_overlay.apply(df_vt.copy(), equity)


# ----------------------------------------------------------------------
# 6. Results
# ----------------------------------------------------------------------
print("\n" + "="*80)
print("INTEGRATION TEST – LAST 10 ROWS")
print("="*80)

cols = ["close", "clean_signal", "vol_annual", "vol_scale_factor",
        "exposure", "signal_sentiment", "risk_multiplier"]
if "exposure_rl" in df_final.columns:
    cols.append("exposure_rl")

print(df_final[cols].tail(10).round(4))
print("="*80)


# ----------------------------------------------------------------------
# 7. Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df_final.index, df_final["close"], label="NVDA Close")
plt.title("NVDA Price")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df_final.index, df_final["exposure"], label="Exposure (vol-targeted)", linewidth=1.5)
if "exposure_rl" in df_final.columns:
    plt.plot(df_final.index, df_final["exposure_rl"], label="Final Position (RL overlay)", color="red", linewidth=2)
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(1, color="gray", linestyle="--", alpha=0.7)
plt.axhline(-1, color="gray", linestyle="--", alpha=0.7)
plt.title("Final Trading Position")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAll components executed successfully – no errors.")