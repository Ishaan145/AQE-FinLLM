"""
modules/module_b_base_learners.py — Module B: Multi-Quantile Base Learners.
LightGBM, XGBoost, CatBoost, Bidirectional LSTM.
Each trained per quantile tau in tau_grid, per market.
Optuna hyperparameter optimization on validation split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    LGBM_PARAMS, XGB_PARAMS, CATBOOST_PARAMS, LSTM_PARAMS,
    TAU_GRID, OPTUNA_TRIALS, SEED,
)
from utils.pinball import pinball_loss

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# LightGBM quantile regressor
# ─────────────────────────────────────────────────────────────────────

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    tau:     float,
    n_trials: int = OPTUNA_TRIALS,
) -> lgb.LGBMRegressor:
    """
    Train LightGBM quantile model with Optuna HPO.
    Monotonicity constraints on Features 4 (VIX) and 11 (sentiment)
    to enforce non-decreasing risk with rising fear signals.
    """
    def objective(trial):
        params = {
            "objective":         "quantile",
            "alpha":             tau,
            "metric":            "quantile",
            "verbosity":         -1,
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 1.0),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state":      SEED,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        return pinball_loss(y_val, preds, tau)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best = {
        "objective":         "quantile",
        "alpha":             tau,
        "metric":            "quantile",
        "verbosity":         -1,
        "n_estimators":      bp["n_estimators"],
        "learning_rate":     bp["lr"],
        "num_leaves":        bp["num_leaves"],
        "min_child_samples": bp["min_child_samples"],
        "reg_alpha":         bp["reg_alpha"],
        "reg_lambda":        bp["reg_lambda"],
        "subsample":         bp["subsample"],
        "colsample_bytree":  bp["colsample_bytree"],
        "random_state":      SEED,
    }
    final = lgb.LGBMRegressor(**best)
    final.fit(X_train, y_train)
    return final


# ─────────────────────────────────────────────────────────────────────
# XGBoost quantile regressor
# ─────────────────────────────────────────────────────────────────────

def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    tau:     float,
    n_trials: int = OPTUNA_TRIALS,
) -> xgb.XGBRegressor:
    """
    Train XGBoost with native reg:quantileerror (v2.0+).
    Monotonicity constraints on VIX (col 3) and realized vol (col 2).
    """
    n_features = X_train.shape[1]
    # Columns 2=realized_vol, 3=vix, 10=sentiment_local: constrain >=0
    mono = [0] * n_features
    for col_idx in [2, 3, 10]:
        if col_idx < n_features:
            mono[col_idx] = 1   # non-decreasing

    def objective(trial):
        params = dict(
            objective          = "reg:quantileerror",
            quantile_alpha     = tau,
            tree_method        = "hist",
            n_estimators       = trial.suggest_int("n_estimators", 100, 800),
            learning_rate      = trial.suggest_float("lr", 0.01, 0.15, log=True),
            max_depth          = trial.suggest_int("max_depth", 3, 8),
            subsample          = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree   = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha          = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda         = trial.suggest_float("reg_lambda", 0.0, 2.0),
            monotone_constraints = tuple(mono),
            random_state       = SEED,
            verbosity          = 0,
        )
        params["early_stopping_rounds"] = 50
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        preds = model.predict(X_val)
        return pinball_loss(y_val, preds, tau)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best = dict(
        objective            = "reg:quantileerror",
        quantile_alpha       = tau,
        tree_method          = "hist",
        n_estimators         = bp["n_estimators"],
        learning_rate        = bp["lr"],
        max_depth            = bp["max_depth"],
        subsample            = bp["subsample"],
        colsample_bytree     = bp["colsample_bytree"],
        reg_alpha            = bp["reg_alpha"],
        reg_lambda           = bp["reg_lambda"],
        monotone_constraints = tuple(mono),
        random_state         = SEED,
        verbosity            = 0,
    )
    final = xgb.XGBRegressor(**best)
    final.fit(X_train, y_train)
    return final


# ─────────────────────────────────────────────────────────────────────
# CatBoost quantile regressor
# ─────────────────────────────────────────────────────────────────────

def train_catboost(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    tau:         float,
    cat_features: list[int] | None = None,  # column indices of categoricals
    n_trials:    int = OPTUNA_TRIALS,
) -> CatBoostRegressor:
    """
    Train CatBoost with Quantile loss.
    Handles categorical regime features (e.g., Election Cycle flag)
    natively via ordered boosting — no one-hot encoding.
    """
    def objective(trial):
        params = dict(
            loss_function      = f"Quantile:alpha={tau}",
            iterations         = trial.suggest_int("iterations", 100, 800),
            learning_rate      = trial.suggest_float("lr", 0.01, 0.15, log=True),
            depth              = trial.suggest_int("depth", 3, 8),
            l2_leaf_reg        = trial.suggest_float("l2_leaf_reg", 0.5, 10.0),
            bagging_temperature= trial.suggest_float("bagging_temp", 0.0, 2.0),
            random_seed        = SEED,
            verbose            = 0,
        )
        if cat_features:
            params["cat_features"] = cat_features
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  early_stopping_rounds=50,
                  verbose=False)
        preds = model.predict(X_val)
        return pinball_loss(y_val, preds, tau)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best = dict(
        loss_function       = f"Quantile:alpha={tau}",
        iterations          = bp["iterations"],
        learning_rate       = bp["lr"],
        depth               = bp["depth"],
        l2_leaf_reg         = bp["l2_leaf_reg"],
        bagging_temperature = bp["bagging_temp"],
        random_seed         = SEED,
        verbose             = 0,
    )
    if cat_features:
        best["cat_features"] = cat_features
    final = CatBoostRegressor(**best)
    final.fit(X_train, y_train)
    return final


# ─────────────────────────────────────────────────────────────────────
# Bidirectional LSTM quantile regressor
# ─────────────────────────────────────────────────────────────────────

class BiLSTMQuantile(nn.Module):
    """
    2-layer Bidirectional LSTM for deep quantile regression.
    Paper Section V-B: hidden_dim=128, seq_len=20, pinball loss via BPTT.
    """
    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)          # (B, seq_len, out_dim)
        last    = out[:, -1, :]        # take last timestep
        return self.head(last).squeeze(-1)   # (B,)


def pinball_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau:    float,
) -> torch.Tensor:
    """Differentiable pinball loss for PyTorch BPTT."""
    u = y_true - y_pred
    return torch.mean(torch.where(u >= 0, tau * u, (tau - 1.0) * u))


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape (T, F) → (T-seq_len, seq_len, F) sequences."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    tau:     float,
    params:  dict | None = None,
) -> BiLSTMQuantile:
    """
    Train BiLSTM with pinball loss via BPTT.
    MinMax scaling applied inside (matches paper Section IV-C).
    """
    cfg = params or LSTM_PARAMS

    # MinMax scale per feature
    X_min  = X_train.min(axis=0)
    X_max  = X_train.max(axis=0)
    X_rng  = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
    X_tr_s = (X_train - X_min) / X_rng
    X_vl_s = (X_val   - X_min) / X_rng

    seq_len = cfg["seq_len"]
    Xtr_s, ytr = make_sequences(X_tr_s, y_train, seq_len)
    Xvl_s, yvl = make_sequences(X_vl_s, y_val,   seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMQuantile(
        input_dim    = X_train.shape[1],
        hidden_dim   = cfg["hidden_dim"],
        num_layers   = cfg["num_layers"],
        dropout      = cfg["dropout"],
        bidirectional= cfg["bidirectional"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    tr_ds = TensorDataset(
        torch.tensor(Xtr_s), torch.tensor(ytr)
    )
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)

    best_val  = float("inf")
    best_state = None

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = pinball_torch(yb, pred, tau)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xvl_s).to(device)
            yv = torch.tensor(yvl).to(device)
            val_loss = pinball_torch(yv, model(xv), tau).item()

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in
                          model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    # Store scaling params on model for inference
    model.X_min   = X_min
    model.X_rng   = X_rng
    model.seq_len = seq_len
    model.eval()
    return model


def predict_lstm(
    model: BiLSTMQuantile,
    X:     np.ndarray,
) -> np.ndarray:
    """Inference: scale → sequence → predict."""
    device  = next(model.parameters()).device
    X_s     = (X - model.X_min) / model.X_rng
    seq_len = model.seq_len
    Xs, _   = make_sequences(X_s, np.zeros(len(X_s)), seq_len)

    with torch.no_grad():
        xb   = torch.tensor(Xs).to(device)
        preds= model(xb).cpu().numpy()

    # Prepend NaNs for the first seq_len steps (no prediction)
    full = np.full(len(X), np.nan)
    full[seq_len:] = preds
    return full


# ─────────────────────────────────────────────────────────────────────
# Multi-quantile training loop
# ─────────────────────────────────────────────────────────────────────

def train_all_base_learners(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    tau_grid: list[float] = TAU_GRID,
    n_trials: int = OPTUNA_TRIALS,
    verbose:  bool = True,
) -> dict[str, dict[float, object]]:
    """
    Train all four learners across the full quantile grid.
    Returns:
        {
          "lgbm":     {tau: model},
          "xgb":      {tau: model},
          "catboost": {tau: model},
          "lstm":     {tau: model},
        }
    """
    learners = {k: {} for k in ["lgbm", "xgb", "catboost", "lstm"]}

    for tau in tau_grid:
        if verbose:
            print(f"  Training tau={tau:.3f}...")

        learners["lgbm"][tau]     = train_lgbm(
            X_train, y_train, X_val, y_val, tau, n_trials)
        learners["xgb"][tau]      = train_xgb(
            X_train, y_train, X_val, y_val, tau, n_trials)
        learners["catboost"][tau] = train_catboost(
            X_train, y_train, X_val, y_val, tau, n_trials=n_trials)
        learners["lstm"][tau]     = train_lstm(
            X_train, y_train, X_val, y_val, tau)

    return learners


def predict_base_learners(
    learners: dict[str, dict[float, object]],
    X:        np.ndarray,
    tau_grid: list[float] = TAU_GRID,
) -> dict[str, dict[float, np.ndarray]]:
    """
    Generate predictions for all learners × quantiles.
    Returns: {learner_name: {tau: predictions array}}
    """
    preds = {}
    for name, tau_models in learners.items():
        preds[name] = {}
        for tau, model in tau_models.items():
            if name == "lstm":
                preds[name][tau] = predict_lstm(model, X)
            else:
                preds[name][tau] = model.predict(X)
    return preds


def save_learners(learners: dict, path: str) -> None:
    """Serialize all models (tree models via joblib, LSTM via torch)."""
    import os
    os.makedirs(path, exist_ok=True)
    for name, tau_models in learners.items():
        for tau, model in tau_models.items():
            fname = f"{path}/{name}_tau{tau:.3f}"
            if name == "lstm":
                torch.save(model.state_dict(), fname + ".pt")
            else:
                joblib.dump(model, fname + ".pkl")


def load_learners(
    path:     str,
    tau_grid: list[float] = TAU_GRID,
    input_dim: int = 12,
) -> dict[str, dict[float, object]]:
    """Load serialized models back from disk."""
    learners = {k: {} for k in ["lgbm", "xgb", "catboost", "lstm"]}
    for tau in tau_grid:
        t = f"{tau:.3f}"
        learners["lgbm"][tau]     = joblib.load(f"{path}/lgbm_tau{t}.pkl")
        learners["xgb"][tau]      = joblib.load(f"{path}/xgb_tau{t}.pkl")
        learners["catboost"][tau] = joblib.load(f"{path}/catboost_tau{t}.pkl")
        lstm_model = BiLSTMQuantile(input_dim=input_dim)
        lstm_model.load_state_dict(
            torch.load(f"{path}/lstm_tau{t}.pt", map_location="cpu"))
        lstm_model.eval()
        learners["lstm"][tau] = lstm_model
    return learners
