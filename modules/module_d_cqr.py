"""
modules/module_d_cqr.py — Module D: Conformal Quantile Regression.

Implements split-conformal CQR (Romano et al., 2019).
Paper: Eq (3) nonconformity scores, Eq (4) CQR threshold,
       Theorem: marginal coverage guarantee >= 1 - alpha.
"""
from __future__ import annotations

import numpy as np
from utils.pinball import pinball_loss


class ConformalQuantileRegressor:
    """
    Split-conformal CQR wrapper for any base quantile estimator.

    Usage:
        cqr = ConformalQuantileRegressor(alpha=0.05)
        cqr.calibrate(y_calib, q_calib)    # Stage 1: fit on calibration
        var_cqr = cqr.correct(q_test)       # Stage 2: correct test preds
        print(cqr.empirical_coverage(y_test, var_cqr))
    """

    def __init__(self, alpha: float = 0.05):
        """
        alpha: miscoverage level (e.g. 0.05 for 95% coverage).
        tau  = 1 - alpha used for nonconformity scores.
        """
        self.alpha   = alpha
        self.tau     = 1.0 - alpha
        self._q_corr = None    # stored correction quantile

    def calibrate(
        self,
        y_calib:   np.ndarray,   # (n,) actual returns on calibration set
        q_calib:   np.ndarray,   # (n,) base model's tau-quantile predictions
    ) -> float:
        """
        Split-conformal calibration for a LOWER-tail quantile (CQR).
        Romano et al. (2019): the conformity score for a one-sided
        lower bound is the SIGNED residual E_i = q_hat(x_i) - y_i,
        and the correction is its empirical (1-alpha) quantile.

        A return that falls below the predicted quantile produces a
        positive E_i; the (1-alpha) quantile of E_i is the amount the
        band must shift down to reach nominal coverage.
        """
        # Signed conformity residual for a lower bound
        E = q_calib - y_calib                       # (n,)

        n     = len(E)
        level = min(np.ceil((1.0 - self.alpha) * (n + 1)) / n, 1.0)
        self._q_corr = float(np.quantile(E, level))
        return self._q_corr

    def correct(
        self,
        q_test: np.ndarray,   # (T,) base model predictions on test set
    ) -> np.ndarray:
        """
        Apply CQR correction.
        Paper Eq (4): q_hat^CQR_alpha(x) = q_hat_tau(x) + Q^+_{1-alpha}

        Returns corrected VaR estimates (T,).
        Negative values → positive VaR via sign convention: VaR = -q.
        """
        if self._q_corr is None:
            raise RuntimeError("Call calibrate() before correct().")
        # Lower-tail quantile: the conformal correction must WIDEN the
        # band downward (make the quantile more negative), so subtract.
        return q_test - self._q_corr

    def empirical_coverage(
        self,
        y_test:  np.ndarray,   # (T,) actual returns
        var_cqr: np.ndarray,   # (T,) CQR-corrected thresholds
    ) -> float:
        """
        Empirical marginal coverage = P(y_{t} >= var_cqr_t).
        Should be >= 1 - alpha (paper Theorem).
        """
        covered = y_test >= var_cqr
        return float(covered.mean())

    def coverage_gap(
        self,
        y_test:  np.ndarray,
        var_cqr: np.ndarray,
    ) -> float:
        """
        Coverage gap = empirical_coverage - (1 - alpha).
        Positive = conservative (over-covered).
        Negative = anti-conservative (under-covered).
        """
        return self.empirical_coverage(y_test, var_cqr) - (1.0 - self.alpha)

    def _nonconformity_scores(
        self,
        y:  np.ndarray,
        q:  np.ndarray,
    ) -> np.ndarray:
        """
        s_i = rho_tau(y_i - q_hat_tau(x_i))
        where rho_tau is the pinball check function.
        """
        u      = y - q
        tau    = self.tau
        scores = np.where(u >= 0, tau * u, (tau - 1.0) * u)
        return scores


def run_cqr_pipeline(
    y_calib:  np.ndarray,
    q_calib:  np.ndarray,
    y_test:   np.ndarray,
    q_test:   np.ndarray,
    alpha:    float = 0.05,
) -> dict[str, float | np.ndarray]:
    """
    Convenience wrapper: calibrate + correct + report coverage.

    Returns dict with:
        correction    : scalar correction applied
        var_cqr       : (T,) corrected VaR series
        coverage      : empirical coverage
        coverage_gap  : deviation from nominal
    """
    cqr = ConformalQuantileRegressor(alpha=alpha)
    corr = cqr.calibrate(y_calib, q_calib)
    var_cqr = cqr.correct(q_test)

    return {
        "correction":   corr,
        "var_cqr":      var_cqr,
        "coverage":     cqr.empirical_coverage(y_test, var_cqr),
        "coverage_gap": cqr.coverage_gap(y_test, var_cqr),
    }
