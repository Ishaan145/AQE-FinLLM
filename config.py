"""
config.py — AQE-FinLLM
All hyperparameters, market definitions, and paths in one place.
Match values to paper Section V (Experimental Design).
"""

# ── Reproducibility ────────────────────────────────────────────────
SEED = 42

# ── Markets ────────────────────────────────────────────────────────
MARKETS = {
    "SP500":     {"ticker": "^GSPC",    "vix": "^VIX",      "fx": "JPY=X",   "lang": "en"},
    "NIKKEI":    {"ticker": "^N225",    "vix": "^JNIV",     "fx": "JPY=X",   "lang": "ja"},
    "BITCOIN":   {"ticker": "BTC-USD",  "vix": None,        "fx": "DX-Y.NYB","lang": "en"},
    "NIFTY50":   {"ticker": "^NSEI",    "vix": "^INDIAVIX", "fx": "INR=X",   "lang": "en"},
    "BANKNIFTY": {"ticker": "^NSEBANK", "vix": "^INDIAVIX", "fx": "INR=X",   "lang": "en"},
}

# ── Data splits ─────────────────────────────────────────────────────
TRAIN_START      = "2015-01-01"
TRAIN_END        = "2019-12-31"
VAL_START        = "2020-01-01"
VAL_END          = "2021-12-31"
CALIB_START      = "2021-01-01"   # CQR calibration set
CALIB_END        = "2021-12-31"
TEST_START       = "2022-01-01"
TEST_END         = "2025-12-31"

# ── Quantile grid (Module B/C) ─────────────────────────────────────
TAU_GRID = [round(0.005 * i, 3) for i in range(1, 11)]   # 0.005 … 0.050
TAU_VAR  = 0.05    # primary VaR level

# ── Pinball / Learners ─────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":       "quantile",
    "metric":          "quantile",
    "verbosity":       -1,
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_child_samples": 20,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
}
XGB_PARAMS = {
    "objective":       "reg:quantileerror",
    "tree_method":     "hist",
    "n_estimators":    500,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree":0.8,
}
CATBOOST_PARAMS = {
    "loss_function":   "Quantile",
    "iterations":      500,
    "learning_rate":   0.05,
    "depth":           6,
    "l2_leaf_reg":     3.0,
    "bagging_temperature": 1.0,
    "verbose":         0,
}
LSTM_PARAMS = {
    "hidden_dim":      128,
    "num_layers":      2,
    "dropout":         0.2,
    "seq_len":         20,
    "batch_size":      64,
    "epochs":          50,
    "lr":              1e-3,
    "bidirectional":   True,
}

# ── Optuna ─────────────────────────────────────────────────────────
OPTUNA_TRIALS = 300

# ── Module C: Hedge ensemble ───────────────────────────────────────
HEDGE_ETA        = 0.1          # learning rate (fixed; paper eq 9)
HEDGE_WINDOW     = 60           # rolling Kupiec window (days)

# ── Module D: CQR ──────────────────────────────────────────────────
CQR_ALPHA        = 0.05         # target miscoverage level

# ── Backtesting ────────────────────────────────────────────────────
KUPIEC_ALPHA     = 0.05         # nominal violation rate
CHI2_CRIT_1      = 3.841        # chi2(1) 95th percentile
CHI2_CRIT_2      = 5.991        # chi2(2) 95th percentile
DQ_LAGS          = 4            # lagged hits in DQ regression

# ── Regime windows ─────────────────────────────────────────────────
REGIMES = {
    "R1_Fed_hike":    ("2022-01-01", "2022-12-31"),
    "R2_FTX":         ("2022-11-01", "2022-11-30"),
    "R3_SVB":         ("2023-03-01", "2023-03-31"),
    "R4_Adani":       ("2023-01-01", "2023-03-31"),
    "R5_BOJ_YCC":     ("2024-07-01", "2024-08-31"),
    "R6_India_Elec":  ("2024-04-01", "2024-06-30"),
}

# ── FinBERT ────────────────────────────────────────────────────────
FINBERT_MODEL_EN = "ProsusAI/finbert"
FINBERT_MODEL_JA = "bardsai/finance-sentiment-ja-base"
FINBERT_MAX_LEN  = 512
FINBERT_BATCH    = 32

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR    = "data/cache/"
RESULTS_DIR = "results/"
MODEL_DIR   = "results/models/"
