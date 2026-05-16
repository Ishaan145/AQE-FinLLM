"""
utils/features.py — Feature engineering for AQE-FinLLM.
Produces all 12 features per Table I of the paper.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ── Core price features (1-10) ──────────────────────────────────────

def log_returns(close: pd.Series) -> pd.Series:
    """Feature 1: log(P_t / P_{t-1})."""
    return np.log(close / close.shift(1)).rename("log_return")


def squared_returns(ret: pd.Series) -> pd.Series:
    """Feature 2: r_{t-1}^2 (variance proxy)."""
    return (ret.shift(1) ** 2).rename("sq_return")


def realized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    """Feature 3: sqrt(sum r_i^2) over rolling window."""
    return (ret ** 2).rolling(window).sum().apply(np.sqrt).rename("realized_vol")


def parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Feature 5: (1/4ln2) * (ln(H/L))^2 — intraday efficiency."""
    return (
        (1.0 / (4.0 * np.log(2)))
        * (np.log(high / low) ** 2)
    ).rename("parkinson_vol")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Feature 6: Relative Strength Index."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Feature 7: MACD = EMA_fast - EMA_slow."""
    return (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    ).rename("macd")


def fx_return(fx_close: pd.Series) -> pd.Series:
    """Feature 9: log FX return vs USD."""
    return np.log(fx_close / fx_close.shift(1)).rename("fx_return")


def lagged_returns(ret: pd.Series, lags: int = 5) -> pd.DataFrame:
    """Feature 10: lagged returns r_{t-1} ... r_{t-lags}."""
    return pd.concat(
        [ret.shift(i).rename(f"lag_{i}") for i in range(1, lags + 1)],
        axis=1,
    )


# ── Build full feature matrix ────────────────────────────────────────

def build_features(
    ohlcv: pd.DataFrame,    # columns: Open, High, Low, Close, Volume
    vix: pd.Series,         # Feature 4: market IV / VIX close
    capital_flow: pd.Series,# Feature 8: FII/DII or fund flow proxy
    fx: pd.Series,          # Feature 9: FX close price
    sentiment: pd.Series,   # Feature 11: per-market S_t^(m)
    global_sentiment: pd.Series,  # Feature 12: global bar_S_t
) -> pd.DataFrame:
    """
    Assemble all 12 features as a DataFrame.
    Returns aligned DataFrame with target column 'log_return'.
    """
    close = ohlcv["Close"]
    high  = ohlcv["High"]
    low   = ohlcv["Low"]

    ret = log_returns(close)

    feat = pd.concat([
        ret,                                   # 1
        squared_returns(ret),                  # 2
        realized_vol(ret),                     # 3
        vix.rename("vix"),                     # 4
        parkinson_vol(high, low),              # 5
        rsi(close),                            # 6
        macd(close),                           # 7
        capital_flow.rename("capital_flow"),   # 8
        fx_return(fx),                         # 9
        lagged_returns(ret),                   # 10 (5 cols)
        sentiment.rename("sentiment_local"),   # 11
        global_sentiment.rename("sentiment_global"),  # 12
    ], axis=1)

    # Sentiment is sparse (few news days); forward/back-fill it so it
    # does not annihilate every row in the join. Slow-moving signal —
    # consistent with the holiday forward-fill policy used elsewhere.
    for col in ("sentiment_local", "sentiment_global"):
        if col in feat.columns:
            feat[col] = feat[col].ffill().bfill().fillna(0.0)

    return feat.dropna()


# ── ADF stationarity enforcement ────────────────────────────────────

def enforce_stationarity(
    series: pd.Series,
    pvalue_thresh: float = 0.05,
) -> pd.Series:
    """
    Paper: if S_t^(m) fails ADF, substitute first differences.
    Returns stationary series. Constant/degenerate series are
    returned unchanged (ADF undefined on a constant).
    """
    clean = series.dropna()
    if clean.empty or clean.nunique() <= 1 or clean.std() == 0:
        return series
    try:
        pval = adfuller(clean, autolag="AIC")[1]
    except ValueError:
        return series
    if pval > pvalue_thresh:
        return series.diff().rename(str(series.name) + "_diff")
    return series


def check_all_stationary(
    df: pd.DataFrame,
    pvalue_thresh: float = 0.05,
) -> dict[str, float]:
    """
    Run ADF on all columns; return {col: p-value}.
    Constant or near-constant columns (e.g. zero-filled capital
    flow when NSEPython is unavailable) are skipped and reported
    as p-value = 1.0 (treated as 'not rejected', handled downstream).
    """
    results: dict[str, float] = {}
    for col in df.columns:
        series = df[col].dropna()
        # Skip degenerate columns: ADF requires non-constant input
        if series.empty or series.nunique() <= 1 or series.std() == 0:
            results[col] = 1.0
            continue
        try:
            results[col] = adfuller(series, autolag="AIC")[1]
        except ValueError:
            results[col] = 1.0
    return results
