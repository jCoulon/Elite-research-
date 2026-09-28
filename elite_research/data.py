"""Market data loading with a local on-disk cache.

Prices come from Yahoo Finance via ``yfinance`` (split- and dividend-adjusted,
i.e. total-return series). Macro series come from FRED's public CSV endpoint.
Everything is cached under ``data/`` so a research run is reproducible offline
once the cache is populated.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
CROSS_ASSET = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
OTHER = ["AGG", "SHY", "QQQ", "IWM", "RSP"]
INDICES = ["^VIX", "^IRX"]
ALL_TICKERS = sorted(set(SECTORS + CROSS_ASSET + OTHER + INDICES))

FRED_SERIES = ["DGS10", "DGS2", "DGS3MO", "T10YIE", "BAMLH0A0HYM2", "UNRATE", "CPIAUCSL"]


def _download_yahoo(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(5):
        try:
            df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:  # network hiccup / rate limit
            pass
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Could not download {ticker}")


def load_ohlcv(ticker: str, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker.replace('^', '_')}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = _download_yahoo(ticker)
    df.to_csv(path)
    return df


def load_panel(tickers=ALL_TICKERS, field: str = "Close", refresh: bool = False) -> pd.DataFrame:
    """Wide DataFrame (dates x tickers) of one OHLCV field."""
    return pd.DataFrame({t: load_ohlcv(t, refresh)[field] for t in tickers}).sort_index()


def load_fred(series: str, refresh: bool = False) -> pd.Series:
    import urllib.request

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"FRED_{series}.csv"
    if not path.exists() or refresh:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    df = pd.read_csv(path)
    df.columns = ["date", series]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[series], errors="coerce").dropna()


def cash_returns(index: pd.DatetimeIndex, refresh: bool = False) -> pd.Series:
    """Daily T-bill return from the 13-week yield (^IRX, percent annualised)."""
    irx = load_ohlcv("^IRX", refresh)["Close"].reindex(index).ffill().fillna(0.0)
    return (irx / 100.0 / 252.0).rename("cash")
