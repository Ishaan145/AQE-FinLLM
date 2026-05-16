# AQE-FinLLM: Adaptive Quantile Ensemble with Multimarket LLM Sentiment Fusion

**CIFEr 2026** | Cross-Market Tail Risk Estimation

---
## Markets Evaluated
S&P 500 (`^GSPC`), Nikkei 225 (`^N225`), Bitcoin (`BTC-USD`),
NIFTY 50 (`^NSEI`), Bank Nifty (`^NSEBANK`)

## Architecture

Raw Market Data + Financial News
        ↓
FinBERT Sentiment Pipeline
        ↓
12-Feature Engineering
        ↓
4 Quantile ML Models
        ↓
Adaptive Hedge Ensemble
        ↓
Conformal Calibration (CQR)
        ↓
VaR / CVaR Forecasts
        ↓
Backtesting + SHAP Explainability

---
## Repository Structure

```
aqe_finllm/
├── data/
│   ├── fetch_market_data.py      # yfinance + NSEPython + FX ingestion
│   └── fetch_sentiment.py        # FinBERT corpus pipeline
├── modules/
│   ├── module_a_finllm.py        # Stage 1-3: tokenize → encode → fuse
│   ├── module_b_base_learners.py # LightGBM, XGBoost, CatBoost, BiLSTM
│   ├── module_c_hedge_ensemble.py# Online Hedge weighting + CVaR
│   └── module_d_cqr.py           # Conformal Quantile Regression
├── backtesting/
│   └── backtest_suite.py         # Kupiec, Christoffersen, DQ, DM tests
├── utils/
│   ├── features.py               # All 12 features + ADF stationarity
│   ├── pinball.py                # Pinball loss + Chernozhukov rearrangement
│   └── shap_explainer.py         # SHAP + interaction decomposition
├── notebooks/
│   └── full_pipeline_demo.ipynb  # End-to-end Colab-ready demo
├── main.py                       # Orchestrator — run full pipeline
├── config.py                     # All hyperparameters, markets, paths
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install feedparser
pip install fugashi unidic-lite
python -m data.build_corpus --output articles.csv
python -m data.fetch_sentiment --input articles.csv --output data\cache\
python main.py --market NSEI --start 2015-01-01 --test_start 2022-01-01
python main.py --market NIFTY50
```


main.py reads data/cache/sentiment_NIFTY50.csv; if that file doesn't exist, it silently substitutes zeros and continues. Module A (data/fetch_sentiment.py) is the separate script that creates that CSV.
# Step 1 — run ONCE (slow, needs news corpus + FinBERT)
python data\fetch_sentiment.py --input articles.csv --output data\cache\

# Step 2 — run many times (fast, reads the cached sentiment CSVs)
python main.py --market NIFTY50
Quick run: python main.py --market NIFTY50 → fetches data → trains all models → runs Hedge → CQR → full backtesting table.

## Reproducibility

All experiments are implemented in Python~3.11 using PyTorch~2.1, LightGBM~4.2, XGBoost~2.0, CatBoost~1.2, and MAPIE~0.8 for conformal quantile calibration. Random seeds are fixed to 42 across all experiments to ensure reproducibility. Hyperparameter optimization is performed using Optuna with deterministic TPE sampler initialization.

## Citation
```
@inproceedings{aqefinllm2026,
  title  = {AQE-FinLLM: Adaptive Quantile Ensemble with Multimarket
             LLM Sentiment Fusion for Cross-Market Tail Risk Estimation},
  author = {Anonymous},
  booktitle = {CIFEr 2026},
}
```
