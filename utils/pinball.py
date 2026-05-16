"""
utils/pinball.py — Pinball loss and Chernozhukov rearrangement.
Paper: Eq (3) pinball loss, Eq (4) rearrangement.
"""
import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Scalar pinball (tick) loss for a single quantile tau.
    rho_tau(u) = u * (tau - 1{u < 0})
    Paper Eq (3).
    """
    u = y_true - y_pred
    return float(np.mean(np.where(u >= 0, tau * u, (tau - 1.0) * u)))


def pinball_loss_grid(
    y_true: np.ndarray,
    q_preds: dict[float, np.ndarray],
) -> dict[float, float]:
    """
    Compute pinball loss for every tau in q_preds dict.
    q_preds: {tau: array of predictions}
    Returns: {tau: loss}
    """
    return {tau: pinball_loss(y_true, preds, tau)
            for tau, preds in q_preds.items()}


def rearrange_quantiles(
    q_preds: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    """
    Post-hoc Chernozhukov et al. (2010) rearrangement:
    enforce monotone non-decreasing quantile function
    across the tau grid at every time step.

    Input:  {tau: array shape (T,)}
    Output: {tau: array shape (T,)} — sorted ascending across taus
    """
    taus = sorted(q_preds.keys())
    T = len(next(iter(q_preds.values())))
    # Stack into (T, K) matrix, sort each row
    matrix = np.column_stack([q_preds[t] for t in taus])  # (T, K)
    matrix = np.sort(matrix, axis=1)                       # sort taus dim
    return {tau: matrix[:, i] for i, tau in enumerate(taus)}


def cvar_from_quantiles(
    q_preds: dict[float, np.ndarray],
    alpha: float = 0.05,
) -> np.ndarray:
    """
    CVaR via quantile averaging (Paper Eq after Eq 11).
    Averages all tau <= alpha after rearrangement.
    Returns array shape (T,).
    """
    rearranged = rearrange_quantiles(q_preds)
    tail_taus = [t for t in sorted(rearranged.keys()) if t <= alpha]
    if not tail_taus:
        raise ValueError(f"No quantiles <= alpha={alpha} in grid.")
    stacked = np.column_stack([rearranged[t] for t in tail_taus])  # (T, K)
    return stacked.mean(axis=1)
