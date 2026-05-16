"""
backtesting/backtest_suite.py — Full backtesting suite.
Implements all four tests from paper Section VI:
  1. Kupiec POF test          (unconditional coverage)
  2. Christoffersen test      (exception independence)
  3. Dynamic Quantile (DQ)    (information extraction)
  4. Diebold-Mariano test     (comparative accuracy)
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Results container for a single model/market backtesting run."""
    model_name:     str
    market:         str
    n_obs:          int
    n_violations:   int
    violation_rate: float

    # Kupiec POF
    lr_pof:         float
    p_kupiec:       float
    kupiec_pass:    bool

    # Christoffersen
    lr_ind:         float
    lr_cc:          float
    p_chr_ind:      float
    p_chr_cc:       float
    christoffersen_pass: bool

    # Dynamic Quantile
    dq_stat:        float
    p_dq:           float
    dq_pass:        bool

    def __str__(self) -> str:
        status = lambda p: "PASS" if p >= 0.05 else "FAIL"
        return (
            f"[{self.model_name} | {self.market}] "
            f"Viol={self.violation_rate:.3f} "
            f"Kupiec={status(self.p_kupiec)}(p={self.p_kupiec:.3f}) "
            f"Chr={status(self.p_chr_ind)}(p={self.p_chr_ind:.3f}) "
            f"DQ={status(self.p_dq)}(p={self.p_dq:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────
# 1. Kupiec Proportion of Failures (POF) Test
# ─────────────────────────────────────────────────────────────────────

def kupiec_pof(
    hits:  np.ndarray,   # (T,) binary: 1 = violation
    alpha: float = 0.05, # nominal violation rate
) -> tuple[float, float]:
    """
    Kupiec (1995) POF likelihood ratio test.
    H0: p = alpha (observed violation rate equals nominal).
    LR_POF ~ chi2(1); reject if > 3.84 (5% level).

    Returns (LR_POF, p-value).
    """
    N  = len(hits)
    x  = int(hits.sum())
    p  = alpha
    ph = x / N if x > 0 else 1e-10

    eps = 1e-10
    ph  = np.clip(ph, eps, 1 - eps)

    lr = -2.0 * (
        x * np.log(p / ph) + (N - x) * np.log((1 - p) / (1 - ph))
    )
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(pval)


# ─────────────────────────────────────────────────────────────────────
# 2. Christoffersen Independence Test
# ─────────────────────────────────────────────────────────────────────

def christoffersen(
    hits:  np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Christoffersen (1998) conditional coverage / independence test.
    Models hit sequence as first-order Markov chain.
    Returns (LR_IND, LR_CC, p_ind, p_cc).
    """
    # Transition counts
    T    = len(hits) - 1
    n_00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n_01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n_10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n_11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    eps = 1e-10
    pi01 = n_01 / max(n_00 + n_01, 1)
    pi11 = n_11 / max(n_10 + n_11, 1)
    pi   = (n_01 + n_11) / max(T, 1)
    pi01 = np.clip(pi01, eps, 1 - eps)
    pi11 = np.clip(pi11, eps, 1 - eps)
    pi   = np.clip(pi,   eps, 1 - eps)

    # LR_IND
    log_num = (n_00 + n_10) * np.log(1 - pi)   + (n_01 + n_11) * np.log(pi)
    log_den = n_00 * np.log(1 - pi01)           + n_01 * np.log(pi01) \
            + n_10 * np.log(1 - pi11)           + n_11 * np.log(pi11)
    lr_ind  = -2.0 * (log_num - log_den)
    p_ind   = 1.0 - stats.chi2.cdf(lr_ind, df=1)

    # LR_POF for CC
    lr_pof, _ = kupiec_pof(hits, alpha)
    lr_cc     = lr_pof + lr_ind
    p_cc      = 1.0 - stats.chi2.cdf(lr_cc, df=2)

    return float(lr_ind), float(lr_cc), float(p_ind), float(p_cc)


# ─────────────────────────────────────────────────────────────────────
# 3. Engle-Manganelli Dynamic Quantile (DQ) Test
# ─────────────────────────────────────────────────────────────────────

def dynamic_quantile(
    hits:       np.ndarray,   # (T,) binary violation indicators
    var_preds:  np.ndarray,   # (T,) VaR predictions (positive values)
    extra_X:    np.ndarray | None = None,  # (T, k) additional regressors
    alpha:      float = 0.05,
    n_lags:     int   = 4,
) -> tuple[float, float]:
    """
    Engle & Manganelli (2004) Dynamic Quantile test.
    Regresses H_t = I_t - (1-alpha) on:
      - n_lags lagged H_{t-i}
      - lagged VaR
      - optional extra_X (India VIX, S_t sentiment, etc.)

    DQ stat ~ chi2(q) where q = number of regressors.
    Returns (DQ_stat, p-value).
    """
    T = len(hits)
    H = hits - alpha   # centered hit process

    # Build regressor matrix Z_t
    Z_cols = []
    for lag in range(1, n_lags + 1):
        Z_cols.append(np.roll(H, lag))    # lagged centered hits

    Z_cols.append(var_preds)             # lagged VaR

    if extra_X is not None:
        if extra_X.ndim == 1:
            Z_cols.append(extra_X)
        else:
            for j in range(extra_X.shape[1]):
                Z_cols.append(extra_X[:, j])

    Z = np.column_stack(Z_cols)
    # Drop first n_lags rows (no valid lags)
    start = n_lags
    H_reg = H[start:]
    Z_reg = Z[start:]

    # Add intercept
    Z_reg = np.column_stack([np.ones(len(Z_reg)), Z_reg])
    q     = Z_reg.shape[1] - 1   # df = number of regressors (excl. intercept)

    # OLS estimate
    try:
        ZtZ_inv = np.linalg.pinv(Z_reg.T @ Z_reg)
        beta_hat = ZtZ_inv @ Z_reg.T @ H_reg
        H_hat    = Z_reg @ beta_hat

        # DQ statistic (quadratic form)
        dq_stat = (H_reg @ Z_reg @ ZtZ_inv @ Z_reg.T @ H_reg) / (
            alpha * (1.0 - alpha)
        )
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    p_val = 1.0 - stats.chi2.cdf(dq_stat, df=q)
    return float(dq_stat), float(p_val)


# ─────────────────────────────────────────────────────────────────────
# 4. Diebold-Mariano Test
# ─────────────────────────────────────────────────────────────────────

def diebold_mariano(
    y_true:   np.ndarray,    # (T,) actual returns
    var_a:    np.ndarray,    # (T,) VaR from model A (e.g. AQE-FinLLM)
    var_b:    np.ndarray,    # (T,) VaR from model B (baseline)
    alpha:    float = 0.05,
) -> tuple[float, float]:
    """
    Diebold & Mariano (1995) test of equal predictive accuracy.
    Loss function: quadratic violation-indicator loss.
    l_t = (I_t - alpha)^2 where I_t = 1{r_t < -VaR_t}.

    H0: E[d_t] = 0 (equal accuracy).
    DM ~ N(0,1) for large T.
    Returns (DM_stat, p-value two-sided).
    """
    I_a = (y_true < -var_a).astype(float)
    I_b = (y_true < -var_b).astype(float)

    L_a = (I_a - alpha) ** 2
    L_b = (I_b - alpha) ** 2
    d   = L_a - L_b           # loss differential

    T      = len(d)
    d_bar  = d.mean()
    # Newey-West HAC variance (lag=int(T^(1/3)))
    h      = max(1, int(T ** (1.0 / 3.0)))
    gamma0 = np.var(d, ddof=0)
    gammas = sum(
        2.0 * (1.0 - j / (h + 1)) * np.cov(d[j:], d[:-j])[0, 1]
        for j in range(1, h + 1)
        if len(d[j:]) > 1
    )
    var_d  = (gamma0 + gammas) / T
    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d)
    p_val   = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


# ─────────────────────────────────────────────────────────────────────
# Full backtest runner
# ─────────────────────────────────────────────────────────────────────

def run_backtest(
    model_name:  str,
    market:      str,
    y_returns:   np.ndarray,   # (T,) actual log returns
    var_preds:   np.ndarray,   # (T,) predicted VaR (positive)
    extra_X:     np.ndarray | None = None,
    alpha:       float = 0.05,
    n_lags:      int   = 4,
) -> BacktestResult:
    """
    Run all three tests for one model × market combination.
    Returns a BacktestResult dataclass.
    """
    hits = (y_returns < -var_preds).astype(float)
    N    = len(hits)
    x    = int(hits.sum())

    lr_pof, p_kup   = kupiec_pof(hits, alpha)
    lr_ind, lr_cc, p_ind, p_cc = christoffersen(hits, alpha)
    dq_stat, p_dq   = dynamic_quantile(hits, var_preds, extra_X, alpha, n_lags)

    return BacktestResult(
        model_name     = model_name,
        market         = market,
        n_obs          = N,
        n_violations   = x,
        violation_rate = x / N,
        lr_pof         = lr_pof,
        p_kupiec       = p_kup,
        kupiec_pass    = p_kup >= alpha,
        lr_ind         = lr_ind,
        lr_cc          = lr_cc,
        p_chr_ind      = p_ind,
        p_chr_cc       = p_cc,
        christoffersen_pass = p_ind >= alpha,
        dq_stat        = dq_stat,
        p_dq           = p_dq,
        dq_pass        = (p_dq >= alpha) if not np.isnan(p_dq) else False,
    )


def run_all_models_backtest(
    models_var: dict[str, np.ndarray],   # {model_name: VaR series}
    y_returns:  np.ndarray,
    market:     str,
    extra_X:    np.ndarray | None = None,
    alpha:      float = 0.05,
) -> dict[str, BacktestResult]:
    """Run backtesting for all models on one market."""
    return {
        name: run_backtest(name, market, y_returns, var, extra_X, alpha)
        for name, var in models_var.items()
    }


def regime_backtest(
    model_name: str,
    market:     str,
    y_returns:  np.ndarray,
    var_preds:  np.ndarray,
    dates:      np.ndarray,         # DatetimeIndex or array of dates
    regimes:    dict[str, tuple],   # {regime_name: (start_str, end_str)}
    alpha:      float = 0.05,
) -> dict[str, BacktestResult]:
    """
    Run backtesting per regime window.
    Returns {regime_name: BacktestResult}.
    """
    import pandas as pd
    dates = pd.to_datetime(dates)
    results = {}
    for regime, (start, end) in regimes.items():
        mask = (dates >= start) & (dates <= end)
        if mask.sum() < 10:
            continue
        results[regime] = run_backtest(
            f"{model_name}_{regime}", market,
            y_returns[mask], var_preds[mask], alpha=alpha,
        )
    return results
