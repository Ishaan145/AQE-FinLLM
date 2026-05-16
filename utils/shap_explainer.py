"""
utils/shap_explainer.py — SHAP explainability for AQE-FinLLM.
Computes mean |SHAP| values and interaction SHAP.
Paper Section VII-D: cross-market SHAP explainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def compute_shap_lgbm(
    model,                     # trained LightGBM model
    X:         np.ndarray,     # (T, 12) test features
    feature_names: list[str],
    max_samples: int = 500,    # subsample for speed
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for LightGBM quantile model.
    Returns:
        shap_vals  : (n, 12) SHAP value matrix
        summary_df : DataFrame with mean |SHAP| per feature, sorted desc
    """
    idx = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
    X_sub = X[idx]

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sub)   # (n, 12)

    mean_abs   = np.abs(shap_vals).mean(axis=0)
    summary_df = pd.DataFrame({
        "feature":   feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_vals, summary_df


def compute_shap_interaction(
    model,
    X:           np.ndarray,
    feature_names: list[str],
    max_samples: int = 200,
) -> np.ndarray:
    """
    Compute SHAP interaction values (n, 12, 12).
    Used to identify sentiment × VIX divergence windows.
    Paper: phi_{ij}(x) — interaction between S_t and VIX.
    """
    idx       = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    return explainer.shap_interaction_values(X[idx])   # (n, 12, 12)


def sentiment_vix_divergence(
    shap_interaction: np.ndarray,  # (n, 12, 12)
    sentiment_idx:    int,          # column index of S_t feature
    vix_idx:          int,          # column index of VIX feature
) -> np.ndarray:
    """
    Extract phi_{sentiment, vix}(x_t) interaction SHAP values.
    Positive and large when sentiment diverges from VIX.
    Returns (n,) array of interaction magnitudes.
    """
    return shap_interaction[:, sentiment_idx, vix_idx]


def cross_market_shap_summary(
    models_by_market: dict[str, object],   # {market: lgbm model}
    X_by_market:      dict[str, np.ndarray],
    feature_names:    list[str],
) -> pd.DataFrame:
    """
    Compute mean |SHAP| ranking per market.
    Returns DataFrame: rows = features, cols = markets.
    """
    records = {}
    for market, model in models_by_market.items():
        _, summary = compute_shap_lgbm(
            model, X_by_market[market], feature_names
        )
        records[market] = summary.set_index("feature")["mean_abs_shap"]

    df = pd.DataFrame(records).fillna(0.0)
    df["mean_across_markets"] = df.mean(axis=1)
    return df.sort_values("mean_across_markets", ascending=False)


def plot_shap_bar(
    summary_df: pd.DataFrame,
    title:      str = "SHAP Feature Importance",
    save_path:  str | None = None,
) -> None:
    """Horizontal bar chart of mean |SHAP| — matches Fig 4 in paper."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["#1a5276" if i < 2 else "#2980b9" if i < 4 else "#85c1e9"
               for i in range(len(summary_df))]
    ax.barh(
        summary_df["feature"][::-1],
        summary_df["mean_abs_shap"][::-1],
        color=colors[::-1],
    )
    ax.set_xlabel("Mean |SHAP value| (impact on tail-quantile prediction)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
