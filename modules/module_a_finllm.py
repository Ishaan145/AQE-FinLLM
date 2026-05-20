"""
modules/module_a_finllm.py — Module A: Multimarket FinLLM CI Pipeline.

Three-stage transformation (Paper Section V-A):
  Stage 1: Corpus filtering and market tagging         [Eq 5]
  Stage 2: Domain-adapted FinBERT/FinBERT-ja encoding  [Eq 6]
  Stage 3: Per-market aggregation + cross-market
           attention fusion                            [Eq 7, 8]
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    FINBERT_MODEL_EN, FINBERT_MODEL_JA,
    FINBERT_MAX_LEN, FINBERT_BATCH, DATA_DIR, SEED,
)
from utils.features import enforce_stationarity

torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Corpus filtering & market tagging (Paper Eq 5)
# ─────────────────────────────────────────────────────────────────────

# Market keyword vocabularies for article tagging
MARKET_KEYWORDS: dict[str, list[str]] = {
    "SP500": ["S&P", "S&P 500", "SPX", "GSPC", "Wall Street", "Dow Jones",
              "Nasdaq", "Fed", "Federal Reserve", "FOMC", "Powell",
              "US stocks", "US equities", "SEC", "Treasury yield",
              "US inflation", "CPI", "nonfarm payroll", "rate hike",
              "rate cut", "US recession", "S&P500", "bear market", 
              "bull market", "record high", "market rally", "earnings"],
    "NIKKEI": ["Nikkei", "Nikkei 225", "N225", "TSE", "Tokyo Stock",
               "BOJ", "Bank of Japan", "Ueda", "Kuroda", "yen", "JPY",
               "Japan equities", "Japanese stocks", "Topix",
               "yield curve control", "YCC", "Japan inflation",
               "Japan GDP", "Abenomics", "bear market", 
               "bull market", "record high", "market rally", "earnings"],
    "BITCOIN": ["Bitcoin", "BTC", "crypto", "cryptocurrency", "blockchain",
                "Coinbase", "Binance", "FTX", "Ethereum", "ETH",
                "digital asset", "stablecoin", "Tether", "USDT",
                "crypto exchange", "halving", "Satoshi", "bitcoin ETF",
                "Grayscale", "MicroStrategy", "crypto crash"],
    "NIFTY50": ["Nifty", "Nifty 50", "Nifty50", "NSE", "BSE", "Sensex", "rupee", "INR",
                "Dalal Street", "FII", "DII", "India GDP", "results",
                "India inflation", "Indian economy", "net profit", "share price", 
                "crude oil", "oil prices", "gold", "public sector", "private sector"
                "Adani", "Reliance", "TCS", "Infosys", "Nifty index", "bear market", 
                "bull market", "record high", "market rally", "earnings", "SEBI", "RBI", "Reserve Bank of India", "India equities",
                "Indian stock market", "Indian markets"],
    "BANKNIFTY": ["Bank Nifty", "BankNifty", "Nifty Bank", "HDFC Bank", "RBI",
                  "ICICI Bank", "SBI", "State Bank of India", "Axis Bank",
                  "Kotak Mahindra", "IndusInd", "PNB", "Indian banking", "Reserve Bank of India",
                  "Indian banks", "bank stocks India", "NPA","RBI repo rate", "NPA",
                  "banking sector India", "Yes Bank", "Bandhan Bank", "bear market",
                  "bull market", "record high", "market rally", "earnings", "SEBI", "India equities",
                  "Indian stock market", "Indian markets", "PSU banks", "basis points", "central bank",
                  "credit growth", "banking crisis"],
}


def tag_article_to_markets(
    text: str,
    keywords: dict[str, list[str]] = MARKET_KEYWORDS,
) -> list[str]:
    """
    Paper Eq (5): D_t^(m) = {d : tag(d)=m, date(d)=t}.
    Returns list of market keys this article is tagged to.
    If no keywords matched, tags to all markets (global article).
    """
    text_lower = text.lower()
    matched = [
        mkt for mkt, kws in keywords.items()
        if any(kw.lower() in text_lower for kw in kws)
    ]
    return matched if matched else list(keywords.keys())


def build_corpus_partition(
    articles: pd.DataFrame,           # columns: date, text, (optional: source)
) -> dict[str, dict[str, list[str]]]: # {market: {date_str: [text, ...]}}
    """
    Partition articles into per-market, per-date buckets.
    Implements the D_t^(m) filtering of Paper Eq (5).
    """
    partition: dict[str, dict[str, list[str]]] = {
        mkt: {} for mkt in MARKET_KEYWORDS
    }
    for _, row in articles.iterrows():
        date_str = str(row["date"])[:10]
        text     = str(row["text"])
        markets  = tag_article_to_markets(text)
        for mkt in markets:
            partition[mkt].setdefault(date_str, []).append(text)
    return partition


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Domain-adapted transformer encoding (Paper Eq 6)
# ─────────────────────────────────────────────────────────────────────

class ArticleDataset(Dataset):
    """Simple dataset for batch inference."""
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.enc = tokenizer(
            texts,
            padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


class FinBERTEncoder:
    """
    Wraps FinBERT / FinBERT-ja for batch inference.
    Produces (p_pos, p_neg, p_neu) per article — Paper Eq (6):
        [p_pos, p_neg, p_neu] = softmax(W_c * h_[CLS] + b_c)
    """

    def __init__(self, lang: Literal["en", "ja"] = "en"):
        model_name = FINBERT_MODEL_EN if lang == "en" else FINBERT_MODEL_JA
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Determine label order (finbert: positive=0, negative=1, neutral=2)
        self._id2label = self.model.config.id2label

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts.
        Returns array of shape (N, 3): [p_pos, p_neg, p_neu].
        """
        if not texts:
            return np.empty((0, 3))

        dataset = ArticleDataset(texts, self.tokenizer, FINBERT_MAX_LEN)
        loader  = DataLoader(dataset, batch_size=FINBERT_BATCH, shuffle=False)
        probs_list = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits          # (B, 3)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

        return np.vstack(probs_list)                     # (N, 3)

    def polarity(self, texts: list[str]) -> np.ndarray:
        """
        Returns delta^(d) = p_pos - p_neg for each article.
        Shape: (N,)
        """
        probs = self.encode_batch(texts)
        if probs.shape[0] == 0:
            return np.array([])
        # Map label indices to positive/negative columns
        label_to_idx = {v.lower(): k for k, v in self._id2label.items()}
        pos_idx = label_to_idx.get("positive", 0)
        neg_idx = label_to_idx.get("negative", 1)
        return probs[:, pos_idx] - probs[:, neg_idx]


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Per-market aggregation + cross-market attention (Eq 7, 8)
# ─────────────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    partition:  dict[str, dict[str, list[str]]],
    encoder_en: FinBERTEncoder,
    encoder_ja: FinBERTEncoder,
    lang_map:   dict[str, str] = None,   # {market: "en"/"ja"}
    dates:      list[str] = None,
) -> dict[str, pd.Series]:
    """
    Paper Eq (7):
        S_t^(m) = (1 / |D_t^(m)|) * sum_{d in D_t^(m)} delta^(d)

    Returns {market: pd.Series(index=DatetimeIndex, values=S_t^(m))}.
    """
    if lang_map is None:
        lang_map = {
            "SP500": "en", "NIKKEI": "ja,en", "BITCOIN": "en",
            "NIFTY50": "en", "BANKNIFTY": "en",
        }

    result: dict[str, pd.Series] = {}

    def _is_japanese(s: str) -> bool:
        # crude but effective: any Hiragana/Katakana/CJK char
        return any(
            "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff"
            for ch in s
        )

    for mkt, daily_texts in partition.items():
        mode = lang_map.get(mkt, "en")
        records = {}

        for date_str, texts in daily_texts.items():
            if not texts:
                records[date_str] = 0.0
                continue

            if mode == "ja":
                deltas = encoder_ja.polarity(texts)
            elif mode in ("ja,en", "mixed", "both"):
                # route each article by detected language
                ja_txt = [t for t in texts if _is_japanese(t)]
                en_txt = [t for t in texts if not _is_japanese(t)]
                parts = []
                if ja_txt:
                    parts.append(encoder_ja.polarity(ja_txt))
                if en_txt:
                    parts.append(encoder_en.polarity(en_txt))
                deltas = (np.concatenate(parts)
                          if parts else np.array([]))
            else:  # "en"
                deltas = encoder_en.polarity(texts)

            records[date_str] = (
                float(deltas.mean()) if len(deltas) > 0 else 0.0
            )

        s = pd.Series(records, name=f"sentiment_{mkt}")
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()

        # Enforce stationarity (Paper Section IV-C)
        s = enforce_stationarity(s)
        result[mkt] = s

    return result


class CrossMarketAttentionFusion(nn.Module):
    """
    Paper Eq (8):
        S_bar_t = sum_m a_t^(m) * S_t^(m)
        a_t^(m) = softmax(w^T h_t^(m))
        h_t^(m) = [S_t^(m), sigma_t^(m), rho_{tm}]

    Learned weight vector w in R^3, trained on validation split.
    """

    def __init__(self, context_dim: int = 3):
        super().__init__()
        self.w = nn.Parameter(torch.ones(context_dim) / context_dim)

    def forward(
        self,
        S:      torch.Tensor,    # (T, M) per-market sentiment
        sigma:  torch.Tensor,    # (T, M) realized vol per market
        rho:    torch.Tensor,    # (T, M) cross-market correlation
    ) -> torch.Tensor:           # (T,) global sentiment bar_S_t
        """
        Compute attention-weighted global sentiment index.
        """
        # h_t^(m) = [S_t^(m), sigma_t^(m), rho_{tm}]  -> (T, M, 3)
        H = torch.stack([S, sigma, rho], dim=-1)
        # scores: (T, M)
        scores = (H * self.w).sum(dim=-1)
        # attention weights: softmax over M markets at each t
        weights = torch.softmax(scores, dim=-1)           # (T, M)
        # weighted sum: (T,)
        return (weights * S).sum(dim=-1)

    def fit(
        self,
        S_val:     np.ndarray,   # (T_val, M)
        sigma_val: np.ndarray,   # (T_val, M)
        rho_val:   np.ndarray,   # (T_val, M)
        target:    np.ndarray,   # (T_val,) — e.g. mean across markets
        epochs:    int = 200,
        lr:        float = 1e-2,
    ) -> None:
        """Train attention weights on validation split."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        S_t  = torch.tensor(S_val,     dtype=torch.float32)
        sig  = torch.tensor(sigma_val, dtype=torch.float32)
        rho_ = torch.tensor(rho_val,   dtype=torch.float32)
        y    = torch.tensor(target,    dtype=torch.float32)

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self(S_t, sig, rho_)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()


def build_global_sentiment(
    sentiment_dict: dict[str, pd.Series],
    realized_vols:  dict[str, pd.Series],
    index:          pd.DatetimeIndex,
    fusion_model:   CrossMarketAttentionFusion | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align per-market sentiments onto common index, compute
    cross-market correlation matrix, and produce global bar_S_t.

    Returns:
        local_df   : DataFrame (T, M) of per-market S_t^(m)
        global_s   : Series (T,) of bar_S_t
    """
    markets = list(sentiment_dict.keys())
    M = len(markets)

    # Align to common index
    local_df = pd.DataFrame(
        {mkt: sentiment_dict[mkt].reindex(index).ffill().fillna(0.0)
         for mkt in markets},
        index=index,
    )

    if fusion_model is None:
        # Simple equal-weight average (non-trained fallback)
        global_s = local_df.mean(axis=1).rename("global_sentiment")
        return local_df, global_s

    # Build context tensors
    sigma_df = pd.DataFrame(
        {mkt: realized_vols[mkt].reindex(index).ffill().fillna(0.02)
         for mkt in markets},
        index=index,
    )

    # Rolling 60-day correlation of each market return with target (NIFTY50)
    # Used as rho_{tm} context feature
    rho_df = local_df.rolling(60).corr(
        local_df.mean(axis=1)
    ).fillna(0.0)

    S_np     = local_df.values.astype(np.float32)     # (T, M)
    sig_np   = sigma_df.values.astype(np.float32)
    rho_np   = rho_df.values.astype(np.float32)

    with torch.no_grad():
        bar_S = fusion_model(
            torch.tensor(S_np),
            torch.tensor(sig_np),
            torch.tensor(rho_np),
        ).numpy()

    global_s = pd.Series(bar_S, index=index, name="global_sentiment")
    return local_df, global_s


# ─────────────────────────────────────────────────────────────────────
# Demo / standalone run
# ─────────────────────────────────────────────────────────────────────

def demo_from_csv(
    articles_csv: str,
    output_dir:   str = DATA_DIR,
) -> dict[str, pd.Series]:
    """
    Run full Module A pipeline on a CSV of articles.
    CSV must have columns: date, text.
    Returns per-market sentiment dict.
    """
    print("[Module A] Loading articles...")
    articles = pd.read_csv(articles_csv, parse_dates=["date"])

    print("[Module A] Stage 1: Corpus partitioning...")
    partition = build_corpus_partition(articles)

    print("[Module A] Stage 2: FinBERT encoding (EN)...")
    enc_en = FinBERTEncoder("en")
    print("[Module A] Stage 2: FinBERT-ja encoding (JA)...")
    enc_ja = FinBERTEncoder("ja")

    print("[Module A] Stage 3: Per-market aggregation...")
    sentiment_dict = aggregate_daily_sentiment(
        partition, enc_en, enc_ja,
    )

    Path(output_dir).mkdir(exist_ok=True)
    for mkt, series in sentiment_dict.items():
        fpath = Path(output_dir) / f"sentiment_{mkt}.csv"
        series.to_csv(fpath)
        print(f"  Saved {mkt} sentiment → {fpath}")

    return sentiment_dict


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        demo_from_csv(sys.argv[1])
    else:
        print("Usage: python module_a_finllm.py <articles.csv>")
        print("CSV columns required: date, text")
