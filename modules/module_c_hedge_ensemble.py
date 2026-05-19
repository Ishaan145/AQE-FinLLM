"""
modules/module_c_hedge_ensemble.py — Module C: CI-Driven Adaptive Quantile Ensemble.

Implements the online Hedge algorithm (Freund & Schapire 1997) with
per-market rolling Kupiec partial log-likelihoods as the loss signal.

Paper equations:
  Eq (9):  l_t^(k,m) = rolling Kupiec PLL
  Eq (10): w_t^(k,m) = softmax(eta * l_t^(k,m))
  Eq (11): regret bound O(sqrt(T ln K))
  After:   CVaR via quantile averaging
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config import HEDGE_ETA, HEDGE_WINDOW, TAU_GRID
from utils.pinball import cvar_from_quantiles, rearrange_quantiles



# ─────────────────────────────────────────────────────────────────────
# Rolling Kupiec partial log-likelihood (loss signal for Hedge)
# ─────────────────────────────────────────────────────────────────────

def kupiec_pll(
    hits:   np.ndarray,   # binary hit sequence: 1 if return < -VaR
    p_hat:  float = 0.05, # nominal violation rate
) -> float:
    """
    Kupiec (1995) partial log-likelihood.
    Paper Eq (9): l_t^(k,m).
    Higher = better calibration for this learner.

    ll = sum_t [ I_t * ln(p_hat) + (1-I_t) * ln(1 - p_hat) ]
    """
    n_hits = hits.sum()
    n_ok   = len(hits) - n_hits
    eps    = 1e-9
    return (
        n_hits * np.log(p_hat + eps)
        + n_ok  * np.log(1.0 - p_hat + eps)
    )


def rolling_pll(
    hit_sequence: np.ndarray,  # (T,) binary
    window:       int  = HEDGE_WINDOW,
    p_hat:        float = 0.05,
) -> np.ndarray:
    """
    Compute rolling Kupiec PLL over a sliding window.
    Returns array (T,); first `window` entries are NaN.
    """
    T      = len(hit_sequence)
    result = np.full(T, np.nan)
    for t in range(window, T):
        window_hits = hit_sequence[t - window:t]
        result[t]   = kupiec_pll(window_hits, p_hat)
    return result


# ─────────────────────────────────────────────────────────────────────
# Hedge weight update (Paper Eq 10)
# ─────────────────────────────────────────────────────────────────────

def hedge_weights(
    pll_matrix: np.ndarray,   # (T, K) — rolling PLL per learner
    eta:        float = HEDGE_ETA,
) -> np.ndarray:
    """
    w_t^(k) = exp(eta * l_t^(k)) / sum_{k'} exp(eta * l_t^(k'))
    Paper Eq (10).

    pll_matrix: (T, K) — rows are timesteps, cols are K learners.
    Returns weight matrix (T, K) with rows summing to 1.
    First `window` rows (where PLL is NaN) get uniform 1/K weights.
    """
    T, K   = pll_matrix.shape
    W      = np.full((T, K), 1.0 / K)   # default: uniform

    for t in range(T):
        row = pll_matrix[t]
        if np.all(np.isfinite(row)):
            exp_scores = np.exp(eta * (row - row.max()))  # numerical stability
            W[t]       = exp_scores / exp_scores.sum()

    return W


# ─────────────────────────────────────────────────────────────────────
# AQE-FinLLM ensemble prediction (Paper Eq 9-10)
# ─────────────────────────────────────────────────────────────────────

def ensemble_predict(
    base_preds: dict[str, dict[float, np.ndarray]],
    W:          np.ndarray,    # (T, K) adaptive weights
    tau_grid:   list[float] = TAU_GRID,
    learner_order: list[str] = None,
) -> dict[float, np.ndarray]:
    """
    Compute the AQE-FinLLM weighted ensemble prediction per quantile.
    Paper Eq (9):
        Q_hat^AQE_tau(x_t) = sum_k w_t^(k) * Q_hat^(k)_tau(x_t)

    Returns {tau: array(T,)} of ensemble quantile predictions.
    """
    if learner_order is None:
        learner_order = sorted(base_preds.keys())

    T = W.shape[0]
    ensemble = {}

    for tau in tau_grid:
        # Stack predictions: (T, K)
        preds_matrix = np.column_stack([
            base_preds[name][tau] for name in learner_order
        ])
        # Replace NaN (LSTM warmup) with row mean of non-NaN
        nan_mask = np.isnan(preds_matrix)
        for t in range(T):
            if nan_mask[t].any():
                finite_vals = preds_matrix[t, ~nan_mask[t]]
                if len(finite_vals) > 0:
                    preds_matrix[t, nan_mask[t]] = finite_vals.mean()

        # Weighted sum per timestep
        ensemble[tau] = (W * preds_matrix).sum(axis=1)  # (T,)

    return ensemble


# ─────────────────────────────────────────────────────────────────────
# Full Module C pipeline: fit → predict → CVaR
# ─────────────────────────────────────────────────────────────────────

class HedgeEnsemble:
    """
    Online Hedge adaptive quantile ensemble.
    Maintains rolling Kupiec PLLs per learner and updates weights daily.
    """

    def __init__(
        self,
        learner_names: list[str],
        tau_grid:      list[float] = TAU_GRID,
        eta:           float = HEDGE_ETA,
        window:        int   = HEDGE_WINDOW,
        alpha:         float = 0.05,
    ):
        self.learner_names = learner_names
        self.tau_grid      = tau_grid
        self.eta           = eta
        self.window        = window
        self.alpha         = alpha   # VaR level for hit computation

        K = len(learner_names)
        self._weights      = np.full(K, 1.0 / K)  # current weights
        self._hit_history  = {name: [] for name in learner_names}

    def compute_hits(
        self,
        returns:    np.ndarray,  # actual log returns
        var_preds:  dict[str, np.ndarray],  # {learner: VaR predictions}
    ) -> dict[str, np.ndarray]:
        """
        I_t^(k) = 1 if return_t < -VaR_t^(k).
        VaR = -Q_{alpha}^(k) (sign convention: VaR is positive).
        """
        hits = {}
        for name in self.learner_names:
            var = -var_preds[name]   # convert quantile to positive VaR
            hits[name] = (returns < -var).astype(float)
        return hits

    def update_weights(
        self,
        hit_history: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Recompute adaptive Hedge weights from rolling PLLs.
        Returns weight array (K,).
        """
        plls = []
        for name in self.learner_names:
            h     = np.array(hit_history[name])
            tail  = h[-self.window:] if len(h) >= self.window else h
            plls.append(kupiec_pll(tail))

        plls   = np.array(plls)
        exp_s  = np.exp(self.eta * (plls - plls.max()))
        self._weights = exp_s / exp_s.sum()
        return self._weights

    def run(
        self,
        base_preds:  dict[str, dict[float, np.ndarray]],
        y_returns:   np.ndarray,  # actual returns (T,)
    ) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]:
        
        """
        Run the full online Hedge loop over T timesteps.

        Returns:
            ensemble_q : {tau: (T,)} ensemble quantile predictions
            var_series : (T,) VaR estimate at alpha
            cvar_series: (T,) CVaR estimate
            weight_mat : (T, K) weight history
        """
        T  = len(y_returns)
        K  = len(self.learner_names)
        W  = np.full((T, K), 1.0 / K)

        # Initialise hit history arrays
        for name in self.learner_names:
            self._hit_history[name] = []

        for t in range(T):
            if t >= self.window:
                # Update weights from past hits
                W[t] = self.update_weights(self._hit_history)

            # Record hit for this timestep
            for i, name in enumerate(self.learner_names):
                var_t = -base_preds[name][self.alpha][t]   # positive VaR
                hit   = float(y_returns[t] < -var_t)
                self._hit_history[name].append(hit)

        # Compute ensemble quantiles with final weight matrix
        ensemble_q = ensemble_predict(
            base_preds, W, self.tau_grid, self.learner_names,
        )

        # Rearrange and compute CVaR
        rearranged  = rearrange_quantiles(ensemble_q)
        cvar_series = cvar_from_quantiles(rearranged, alpha=self.alpha)
        var_series  = -rearranged[self.alpha]   # sign: VaR is positive

        return ensemble_q, var_series, cvar_series, W


# ─────────────────────────────────────────────────────────────────────
# Regret bound utility (paper Eq 11)
# ─────────────────────────────────────────────────────────────────────

def optimal_eta(T: int, K: int) -> float:
    """eta* = sqrt(8 ln K / T) — optimal Hedge learning rate."""
    return np.sqrt(8.0 * np.log(K) / T)


def regret_upper_bound(T: int, K: int, eta: float = None) -> float:
    """
    Hedge regret bound: ln(K)/eta + eta*T/8.
    Paper Eq (11).
    """
    if eta is None:
        eta = optimal_eta(T, K)
    return np.log(K) / eta + eta * T / 8.0
