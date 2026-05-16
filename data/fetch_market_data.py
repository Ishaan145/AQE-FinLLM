"""
data/fetch_market_data.py — Market data ingestion.
Sources: yfinance (OHLCV + VIX + FX), NSEPython (FII/DII flows).
Caches to CSV to avoid repeated API hits.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from config import MARKETS, TRAIN_START, TEST_END, DATA_DIR

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    ticker: str,
    start: str = TRAIN_START,
    end:   str = TEST_END,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance.
    Returns DataFrame indexed by Date with columns
    [Open, High, Low, Close, Volume].
    """
    fname = Path(DATA_DIR) / f"{ticker.replace('^','').replace('-','_')}_ohlcv.csv"
    if cache and fname.exists():
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        if not df.empty and set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
            return df
        # corrupted/legacy cache — fall through to re-download
        fname.unlink(missing_ok=True)

    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    # Flatten MultiIndex columns (yfinance v0.2.40+ behaviour)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df  = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df  = df.apply(pd.to_numeric, errors="coerce")
    df.index.name = "Date"

    if cache:
        df.to_csv(fname)
    return df


def fetch_vix(
    vix_ticker: str | None,
    start: str = TRAIN_START,
    end:   str = TEST_END,
) -> pd.Series:
    """
    Fetch volatility index. Returns daily close as Series.
    If vix_ticker is None (e.g. Bitcoin), returns NaN series.
    """
    if vix_ticker is None:
        return pd.Series(dtype=float, name="vix")
    raw = yf.download(vix_ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")
    close.name = "vix"
    return close


def fetch_fx(
    fx_ticker: str,
    start: str = TRAIN_START,
    end:   str = TEST_END,
) -> pd.Series:
    """Fetch FX close price."""
    raw = yf.download(fx_ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")
    close.name = "fx"
    return close


def fetch_fii_dii(
    start: str = TRAIN_START,
    end:   str = TEST_END,
    cache: bool = True,
) -> pd.Series:
    """
    Fetch NSE FII/DII net institutional flows (India only).
    Uses NSEPython; falls back to zero-series if unavailable.
    Returns daily net buy (INR Crores) as Series.
    """
    fname = Path(DATA_DIR) / "fii_dii_net.csv"
    if cache and fname.exists():
        return pd.read_csv(fname, index_col=0, parse_dates=True).squeeze()

    try:
        import nsepython
        # nsepython API name varies by version
        if hasattr(nsepython, "fii_dii_data"):
            df = nsepython.fii_dii_data()
        elif hasattr(nsepython, "nse_fiidii"):
            df = nsepython.nse_fiidii()
        else:
            raise ImportError("no fii_dii function in nsepython")
        df.index = pd.to_datetime(df.index)
        net = pd.to_numeric(df["Net"], errors="coerce").astype(float)
        net = net.loc[start:end]
        net.name = "capital_flow"
        if cache:
            net.to_csv(fname)
        return net
    except Exception as e:
        print(f"[fetch_fii_dii] NSEPython unavailable: {e}. "
              "Returning zero series.")
        idx = pd.date_range(start, end, freq="B")
        return pd.Series(0.0, index=idx, name="capital_flow")


def fetch_all_for_market(
    market_key: str,
    start: str = TRAIN_START,
    end:   str = TEST_END,
) -> dict[str, pd.Series | pd.DataFrame]:
    """
    Convenience wrapper: fetches all price/vol/fx data for one market.
    Returns dict with keys: ohlcv, vix, fx, capital_flow.
    """
    cfg = MARKETS[market_key]
    ohlcv  = fetch_ohlcv(cfg["ticker"], start, end)
    vix    = fetch_vix(cfg["vix"],    start, end)
    fx     = fetch_fx(cfg["fx"],      start, end)

    if market_key in ("NIFTY50", "BANKNIFTY"):
        flow = fetch_fii_dii(start, end)
    else:
        # Use fund flow proxy: approximate with USD DXY returns or zeros
        flow = pd.Series(0.0, index=ohlcv.index, name="capital_flow")

    # Align all to ohlcv business-day index
    idx    = ohlcv.index
    vix    = vix.reindex(idx).ffill()
    fx     = fx.reindex(idx).ffill()
    flow   = flow.reindex(idx).fillna(0.0)

    return {"ohlcv": ohlcv, "vix": vix, "fx": fx, "capital_flow": flow}


if __name__ == "__main__":
    for mkt in MARKETS:
        print(f"Fetching {mkt}...")
        data = fetch_all_for_market(mkt)
        print(f"  OHLCV rows: {len(data['ohlcv'])}")
