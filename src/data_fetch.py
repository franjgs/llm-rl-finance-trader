import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import argparse
import os
import time
from requests.exceptions import RequestException

def fetch_prices(stock, start, end, max_retries=3):
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        df = web.DataReader(stock, data_source='stooq', start=start, end=end)
        if df.empty:
            raise ValueError(f"No data from Stooq for {stock}")
        df = df.sort_index()
        df.columns = [col.lower() for col in df.columns]
        df.to_csv(f"data/raw/{stock}.csv")
        print(f"Éxito (Stooq): Datos guardados en data/raw/{stock}.csv ({len(df)} filas)")
        return df
    except Exception as e:
        print(f"Stooq falló para {stock}: {e}")
    
    for attempt in range(max_retries):
        try:
            df = yf.download(stock, start=start, end=end, ignore_tz=True, progress=False)
            if df.empty:
                raise ValueError(f"No yfinance data for {stock} on attempt {attempt+1}")
            df.columns = [col.lower() for col in df.columns]
            df.to_csv(f"data/raw/{stock}.csv")
            print(f"Éxito (yfinance): Datos guardados en data/raw/{stock}.csv ({len(df)} filas)")
            return df
        except (RequestException, ValueError) as e:
            print(f"yfinance intento {attempt+1} falló para {stock}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    
    raise ValueError(f"No se pudo descargar datos para {stock}. Descarga manualmente desde finance.yahoo.com")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="AAPL")
    parser.add_argument("--start", default="2023-11-16")
    parser.add_argument("--end", default="2024-11-10")
    args = parser.parse_args()
    fetch_prices(args.stock, args.start, args.end)