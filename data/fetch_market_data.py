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


def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
    """FIX: yfinance v0.2.40+ returns MultiIndex even for one ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _series_from(raw: pd.DataFrame, col: str, name: str) -> pd.Series:
    """FIX: robustly extract one column as a clean float Series."""
    raw = _flatten(raw)
    if raw is None or raw.empty or col not in raw.columns:
        return pd.Series(dtype=float, name=name)
    s = raw[col]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze("columns")
    s = pd.to_numeric(s, errors="coerce")
    s.name = name
    return s


def fetch_ohlcv(
    ticker: str,
    start: str = TRAIN_START,
    end:   str = TEST_END,
    cache: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    fname = Path(DATA_DIR) / f"{ticker.replace('^','').replace('-','_')}_ohlcv.csv"
    if cache and fname.exists():
        df = pd.read_csv(fname, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        ok_cols = set(["Open","High","Low","Close","Volume"]).issubset(df.columns)
        fresh = (not df.empty
                 and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5))
        if ok_cols and fresh:
            return df
        fname.unlink(missing_ok=True)

    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    raw = _flatten(raw)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"[fetch_ohlcv] yfinance returned NO data for '{ticker}'. "
            f"Check ticker / network / rate limit.")
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
    fallback_close: pd.Series | None = None,   # FIX: realized-vol fallback
) -> pd.Series:
    """
    Fetch volatility index as a Series.
    FIX: if the VIX ticker is dead (^JNIV, often ^INDIAVIX) or returns
    empty, fall back to 20-day annualised realized vol of the index
    close, so feature 4 is never silently all-NaN.
    """
    if vix_ticker is None:
        s = pd.Series(dtype=float, name="vix")
    else:
        raw = yf.download(vix_ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        s = _series_from(raw, "Close", "vix")

    if s.dropna().empty:
        if fallback_close is not None and not fallback_close.dropna().empty:
            print(f"[fetch_vix] '{vix_ticker}' empty -> using 20d "
                  f"realized-vol proxy.")
            ret = np.log(fallback_close / fallback_close.shift(1))
            rv  = ret.rolling(20).std() * np.sqrt(252) * 100.0
            rv.name = "vix"
            return rv
        print(f"[fetch_vix] '{vix_ticker}' empty and no fallback -> zeros.")
    return s


def fetch_fx(
    fx_ticker: str,
    start: str = TRAIN_START,
    end:   str = TEST_END,
) -> pd.Series:
    """Fetch FX close price as a Series."""
    raw = yf.download(fx_ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    s = _series_from(raw, "Close", "fx")
    if s.dropna().empty:
        print(f"[fetch_fx] '{fx_ticker}' returned empty.")
    return s


def fetch_fii_dii(
    start: str = TRAIN_START,
    end:   str = TEST_END,
    cache: bool = True,
) -> pd.Series:
    """NSE FII/DII net flows (India). Zero-series fallback if unavailable."""
    fname = Path(DATA_DIR) / "fii_dii_net.csv"
    if cache and fname.exists():
        s = pd.read_csv(fname, index_col=0, parse_dates=True).squeeze("columns")
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").rename("capital_flow")

    try:
        import nsepython
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
    cfg = MARKETS[market_key]
    ohlcv  = fetch_ohlcv(cfg["ticker"], start, end)
    # FIX: pass ohlcv close so dead VIX tickers fall back to realized vol
    vix    = fetch_vix(cfg["vix"], start, end,
                       fallback_close=ohlcv["Close"])
    fx     = fetch_fx(cfg["fx"], start, end)

    if market_key in ("NIFTY50", "BANKNIFTY"):
        flow = fetch_fii_dii(start, end)
    else:
        flow = pd.Series(0.0, index=ohlcv.index, name="capital_flow")

    idx    = ohlcv.index
    vix    = vix.reindex(idx).ffill().bfill()      # FIX: bfill head NaNs
    fx     = fx.reindex(idx).ffill().bfill()       # FIX
    flow   = flow.reindex(idx).fillna(0.0)

    # FIX: hard guard — if VIX still all-NaN, use realized vol so
    # feature 4 is never a constant (which broke ADF earlier)
    if vix.dropna().empty:
        ret = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))
        vix = (ret.rolling(20).std() * np.sqrt(252) * 100.0
               ).reindex(idx).ffill().bfill()
        vix.name = "vix"

    return {"ohlcv": ohlcv, "vix": vix, "fx": fx, "capital_flow": flow}


if __name__ == "__main__":
    for mkt in MARKETS:
        print(f"Fetching {mkt}...")
        data = fetch_all_for_market(mkt)
        o = data["ohlcv"]
        print(f"  OHLCV rows: {len(o)}  "
              f"({o.index.min().date()} -> {o.index.max().date()})")
        print(f"  VIX non-null: {data['vix'].notna().sum()}  "
              f"FX non-null: {data['fx'].notna().sum()}")