import yfinance as yf
import pandas as pd
from src.models.momentum import generate_momentum_signal
from src.models.volatility_targeting import apply_vol_target

# download 1h data (if available) or daily fallback
df = yf.download("NVDA", start="2022-01-01", end="2023-12-01", interval="1d", progress=False)
df = df.reset_index().rename(columns={"Date":"Date", "Adj Close":"close"})  # adjust if needed
df = df[["Date","close"]]
df = df.set_index(pd.to_datetime(df["Date"]))

# create a simple momentum signal (re-use your params)
from pprint import pprint
params_mom = {
    "lookback_bars": 20,
    "vol_lookback_bars": 20,
    "use_log_returns": True,
    "target_vol": 0.02,
    "smoothing_alpha": 0.12,
    "return_raw": True
}
df2 = generate_momentum_signal(df.reset_index(), params_mom).set_index("Date")
# ensure clean_signal exists (momentum function creates signal_momentum; map to clean_signal)
df2["clean_signal"] = df2["signal_momentum"]

vt_params = {
    "target_vol": 0.20,
    "vol_lookback_bars": 20,
    "max_leverage": 3.0,
    "min_vol": 1e-4,
    "max_scale": 4.0,
    "return_raw": True
}
out = apply_vol_target(df2, vt_params)
print(out[["clean_signal","vol_annual","vol_scale_factor","exposure"]].tail(10))