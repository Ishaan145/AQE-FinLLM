"""
main.py — AQE-FinLLM full pipeline orchestrator.
Runs Modules A → B → C → D → Backtesting → SHAP for one market.

Usage:
    python main.py --market NIFTY50
    python main.py --market SP500 --no_sentiment
    python main.py --all_markets
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

import config as cfg
from data.fetch_market_data import fetch_all_for_market
from utils.features import build_features, check_all_stationary
from modules.module_b_base_learners import (
    train_all_base_learners, predict_base_learners, save_learners,
)
from modules.module_c_hedge_ensemble import HedgeEnsemble, regret_upper_bound
from modules.module_d_cqr import run_cqr_pipeline
from backtesting.backtest_suite import (
    run_backtest, run_all_models_backtest, diebold_mariano,
)

import warnings
warnings.filterwarnings("ignore")

os.makedirs(cfg.DATA_DIR,    exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(cfg.MODEL_DIR,   exist_ok=True)


def load_sentiment(market: str) -> tuple[pd.Series, pd.Series]:
    """Load precomputed sentiment CSVs for a market."""
    local_path  = Path(cfg.DATA_DIR) / f"sentiment_{market}.csv"
    global_path = Path(cfg.DATA_DIR) / "sentiment_global.csv"

    if local_path.exists():
        local = pd.read_csv(local_path, index_col=0, parse_dates=True
                            ).squeeze("columns")
        local = pd.to_numeric(local, errors="coerce").rename("sentiment_local")
    else:
        print(f"  [!] No local sentiment found for {market}. Using zeros.")
        local = pd.Series(dtype=float, name="sentiment_local")

    if global_path.exists():
        glb = pd.read_csv(global_path, index_col=0, parse_dates=True
                          ).squeeze("columns")
        glb = pd.to_numeric(glb, errors="coerce").rename("sentiment_global")
    else:
        glb = pd.Series(dtype=float, name="sentiment_global")

    return local, glb


def run_pipeline(market: str, use_sentiment: bool = True) -> dict:
    """Full AQE-FinLLM pipeline for one market."""

    print(f"\n{'='*60}")
    print(f"  AQE-FinLLM | Market: {market}")
    print(f"{'='*60}")

    # ── 1. Data ingestion ─────────────────────────────────────────
    print("[1] Fetching market data...")
    data = fetch_all_for_market(market)
    ohlcv, vix, fx, flow = (
        data["ohlcv"], data["vix"], data["fx"], data["capital_flow"]
    )

    # ── 2. Sentiment loading (Module A output) ───────────────────
    print("[2] Loading sentiment features...")
    sentiment_local, sentiment_global = load_sentiment(market)

    if not use_sentiment or len(sentiment_local) == 0:
        idx = ohlcv.index
        sentiment_local  = pd.Series(0.0, index=idx, name="sentiment_local")
        sentiment_global = pd.Series(0.0, index=idx, name="sentiment_global")

    # ── 3. Feature engineering ────────────────────────────────────
    print("[3] Building 12-feature matrix...")
    feat_df = build_features(
        ohlcv, vix, flow, fx,
        sentiment_local, sentiment_global,
    )

    # Stationarity check
    pvals = check_all_stationary(feat_df)
    failed = {c: p for c, p in pvals.items() if p > 0.05}
    if failed:
        print(f"  [!] Non-stationary features: {list(failed.keys())}")

    feature_names = list(feat_df.columns)
    target_col    = "log_return"
    feature_cols  = [c for c in feature_names if c != target_col]

    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)
    dates = feat_df.index

    # ── Chronological splits ──────────────────────────────────────
    def mask(start, end):
        return (dates >= start) & (dates <= end)

    tr  = mask(cfg.TRAIN_START,  cfg.TRAIN_END)
    val = mask(cfg.VAL_START,    cfg.VAL_END)
    cal = mask(cfg.CALIB_START,  cfg.CALIB_END)
    te  = mask(cfg.TEST_START,   cfg.TEST_END)

    X_tr,  y_tr  = X_all[tr],  y_all[tr]
    X_val, y_val = X_all[val], y_all[val]
    X_cal, y_cal = X_all[cal], y_all[cal]
    X_te,  y_te  = X_all[te],  y_all[te]

    print(f"  Train: {tr.sum()}  Val: {val.sum()}  "
          f"Calib: {cal.sum()}  Test: {te.sum()} days")

    # ── 4. Module B: Train base learners ─────────────────────────
    print("[4] Training base learners (this may take a while)...")
    learners = train_all_base_learners(
        X_tr, y_tr, X_val, y_val,
        tau_grid=cfg.TAU_GRID,
        n_trials=cfg.OPTUNA_TRIALS,
        verbose=True,
    )
    save_learners(learners, f"{cfg.MODEL_DIR}/{market}")
    print(f"  Models saved → {cfg.MODEL_DIR}/{market}/")

    # ── 5. Module C: Hedge ensemble on test set ───────────────────
    print("[5] Running Hedge adaptive ensemble (Module C)...")
    test_preds = predict_base_learners(learners, X_te, cfg.TAU_GRID)

    hedge = HedgeEnsemble(
        learner_names = list(learners.keys()),
        tau_grid      = cfg.TAU_GRID,
        eta           = cfg.HEDGE_ETA,
        window        = cfg.HEDGE_WINDOW,
        alpha         = cfg.TAU_VAR,
    )
    ensemble_q, var_aqe, cvar_aqe, weight_mat = hedge.run(
        test_preds, y_te
    )

    K  = len(learners)
    T  = len(y_te)
    rb = regret_upper_bound(T, K)
    print(f"  Hedge regret bound (T={T}, K={K}): {rb:.4f}")

    # ── 6. Module D: Conformal calibration ───────────────────────
    print("[6] Conformal calibration (Module D / CQR)...")
    cal_preds  = predict_base_learners(learners, X_cal, cfg.TAU_GRID)
    hedge_cal  = HedgeEnsemble(
        learner_names=list(learners.keys()),
        tau_grid=cfg.TAU_GRID, eta=cfg.HEDGE_ETA,
        window=cfg.HEDGE_WINDOW, alpha=cfg.TAU_VAR,
    )
    _, var_cal, _, _ = hedge_cal.run(cal_preds, y_cal)

    # var_cal / var_aqe are POSITIVE VaR (loss) numbers.
    # CQR works on the lower-tail QUANTILE, which is the negative
    # of VaR: q = -VaR. Convert once, correct, convert back once.
    q_cal = -var_cal          # calibration lower-tail quantile
    q_te  = -var_aqe          # test lower-tail quantile

    cqr_result = run_cqr_pipeline(
        y_cal, q_cal, y_te, q_te, alpha=cfg.CQR_ALPHA
    )
    # run_cqr_pipeline returns the corrected QUANTILE; VaR = -quantile
    var_cqr = -cqr_result["var_cqr"]
    print(f"  CQR correction: {cqr_result['correction']:.5f}")
    print(f"  Empirical coverage: {cqr_result['coverage']:.4f} "
          f"(gap: {cqr_result['coverage_gap']:+.4f})")

    # ── 7. Backtesting ────────────────────────────────────────────
    print("[7] Running backtesting suite...")

    # Only backtest the ensemble + CQR variant (skip per-learner rows)
    single_vars = {
        "AQE_FinLLM":     var_aqe,
        "AQE_FinLLM_CQR": var_cqr,
    }

    results = run_all_models_backtest(
        single_vars, y_te, market=market,
        extra_X=X_te[:, feature_cols.index("sentiment_local")]
               .reshape(-1, 1) if "sentiment_local" in feature_cols else None,
        alpha=cfg.TAU_VAR,
    )

    print("\n  Backtesting Results:")
    for name, res in results.items():
        print(f"  {res}")

    # ── 8. Diebold-Mariano vs. baseline ───────────────────────────
    # Pull LightGBM VaR directly from base predictions for the DM test,
    if "lgbm" in test_preds:
        lgbm_q = test_preds["lgbm"][cfg.TAU_VAR]
        lgbm_q = np.where(np.isnan(lgbm_q), np.nanmean(lgbm_q), lgbm_q)
        lgbm_var = -lgbm_q
        dm_stat, dm_p = diebold_mariano(
            y_te, var_aqe, lgbm_var, alpha=cfg.TAU_VAR
        )
        print(f"\n  DM test (AQE-FinLLM vs LightGBM): "
              f"stat={dm_stat:.3f}  p={dm_p:.4f}")

    # ── Save results ──────────────────────────────────────────────
    out = {
        "market":       market,
        "var_aqe":      var_aqe,
        "cvar_aqe":     cvar_aqe,
        "var_cqr":      var_cqr,
        "cqr_coverage": cqr_result["coverage"],
        "weight_mat":   weight_mat,
        "backtest":     results,
        "dates_test":   dates[te],
        "y_test":       y_te,
    }

    results_path = Path(cfg.RESULTS_DIR) / f"{market}_results.npz"
    np.savez(
        results_path,
        var_aqe=var_aqe, cvar_aqe=cvar_aqe, var_cqr=var_cqr,
        weight_mat=weight_mat, y_test=y_te,
    )
    print(f"\n  Results saved → {results_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="AQE-FinLLM Pipeline")
    parser.add_argument("--market", type=str, default="NIFTY50",
                        choices=list(cfg.MARKETS.keys()))
    parser.add_argument("--all_markets", action="store_true")
    parser.add_argument("--no_sentiment", action="store_true",
                        help="Ablation: run without FinBERT sentiment")
    args = parser.parse_args()

    markets = list(cfg.MARKETS.keys()) if args.all_markets else [args.market]
    use_s   = not args.no_sentiment

    all_results = {}
    for mkt in markets:
        all_results[mkt] = run_pipeline(mkt, use_sentiment=use_s)

    print("\n" + "="*60)
    print("  Summary Table")
    print("="*60)
    print(f"{'Market':<12} {'Viol%':>7} {'Kupiec':>10} {'Chr':>10} {'DQ':>10} {'CQR':>8}")
    for mkt, res in all_results.items():
        r = res["backtest"].get("AQE_FinLLM")
        if r:
            cov = res["cqr_coverage"]
            print(f"{mkt:<12} {r.violation_rate*100:>6.2f}% "
                  f"{r.p_kupiec:>10.3f} {r.p_chr_ind:>10.3f} "
                  f"{r.p_dq:>10.3f} {cov:>8.3f}")


if __name__ == "__main__":
    main()
