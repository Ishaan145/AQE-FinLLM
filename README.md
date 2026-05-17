# AQE-FinLLM: Adaptive Quantile Ensemble with Multimarket LLM Sentiment Fusion
Cross-Market Tail Risk Estimation |  **CIFEr 2026** 

---
## Markets Evaluated
S&P 500 (`^GSPC`), Nikkei 225 (`^N225`), Bitcoin (`BTC-USD`),
NIFTY 50 (`^NSEI`), Bank Nifty (`^NSEBANK`)

## Architecture

```bash
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
```
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

# 1. Download both datasets from Kaggle, unzip, place CSVs in project root AQE-FINLLM, Downloaded Kaggle datasets with renaming already available in project repository for reference

Kaggle 2008–2024, daily headlines, perfectly covers your full test window
URL:  kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024

Kaggle Full-text Indian financial news,2003–2020 (Economic Times). Covers most of your test 
URL:  https://www.kaggle.com/datasets/hkapoor/indian-financial-news-articles-20032020

#    Rename them to:
      sp500_headlines_2008_2024.csv
      nifty_news_2003_2020.csv

# 2. Run pipeline

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install feedparser
pip install fugashi unidic-lite
python -m data.build_corpus --output articles.csv
python -m data.normalize_corpus --input raw_news.csv --output articles.csv
python -m data.fetch_sentiment --input articles.csv --output data\cache\
python main.py --market NSEI --start 2015-01-01 --test_start 2022-01-01
python main.py --market NIFTY50
python main.py --market SP500
```

## Note: 

main.py reads data/cache/sentiment_NIFTY50.csv; if that file doesn't exist, it silently substitutes zeros and continues. Module A (data/fetch_sentiment.py) is the separate script that creates that CSV.

# Here authors have given option to skip datasets if one want to run model only on kaggle downloaded dataset can skip real time news fetching, normalize just prints (skip) not found: rss_live.csv and proceeds with Kaggle only. If you skip Kaggle, it uses RSS only. Any combination works it merges whatever exists. One command auto pull (if you want RSS fetched automatically every run)

Add this near the top of normalize_corpus.py's main(), right after args = parser.parse_args():
```bash
# Auto-refresh live RSS before combining (best-effort, non-fatal)
    if not args.input:
        try:
            from data.build_corpus import fetch_feed, FEEDS
            import pandas as _pd
            rows = []
            for _name, _url in FEEDS.items():
                rows.extend(fetch_feed(_name, _url))
            if rows:
                _df = _pd.DataFrame(rows)[["date", "text"]]
                _df.to_csv(_PROJECT_ROOT / "rss_live.csv", index=False)
                print(f"[normalize] Auto-pulled {len(_df)} live RSS rows")
        except Exception as e:
            print(f"[normalize] RSS auto-pull skipped: {e}")
```
Then a single python -m data.normalize_corpus pulls fresh RSS and merges with Kaggle in one shot. Requires build_corpus.py's fetch_feed/FEEDS to be importable (they already are if you applied the earlier sys.path pattern).

#  (optional, anytime) pull fresh live news
python -m data.build_corpus                 # -> rss_live.csv

#  combine ALL available sources (Kaggle history + RSS live)
python -m data.normalize_corpus             # -> articles.csv

#  sentiment + pipeline
python -m data.fetch_sentiment --input articles.csv --output data\cache\
python main.py --market NIFTY50

# Run ONCE (slow, needs news corpus + FinBERT)
Run FinBERT over the combined corpus
python -m data.fetch_sentiment

# Run many times (fast, reads the cached sentiment CSVs)
python main.py --market NIFTY50
Quick run: python main.py --market NIFTY50 → fetches data → trains all models → runs Hedge → CQR → full backtesting table.

## Architecture Image
![alt text](architecture.png)

## Reproducibility

All experiments are implemented in Python~3.11 using PyTorch~2.1, LightGBM~4.2, XGBoost~2.0, CatBoost~1.2, and MAPIE~0.8 for conformal quantile calibration. Random seeds are fixed to 42 across all experiments to ensure reproducibility. Hyperparameter optimization is performed using Optuna with deterministic TPE sampler initialization.

## Citation
```
@inproceedings{aqefinllm2026,
  title  = {Cross-Market Tail Risk Estimation using Adaptive Quantile Ensembles, FinLLM Sentiment Fusion, and Conformal Calibration},
  author = {Anonymous},
  booktitle = {CIFEr 2026},
}
```
