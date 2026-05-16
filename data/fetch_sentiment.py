"""
data/fetch_sentiment.py — Standalone Module A runner.
Reads articles CSV, runs FinBERT pipeline, saves per-market sentiment.

Usage:
    python fetch_sentiment.py --input articles.csv --output data/cache/

Input CSV format:
    date, text, [source]     <- source optional

Output:
    data/cache/sentiment_SP500.csv
    data/cache/sentiment_NIKKEI.csv
    ... etc.
    data/cache/sentiment_global.csv
"""
import argparse
import pandas as pd
from pathlib import Path
from modules.module_a_finllm import (
    FinBERTEncoder,
    build_corpus_partition,
    aggregate_daily_sentiment,
    CrossMarketAttentionFusion,
    build_global_sentiment,
)
from config import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True,
                        help="Path to articles CSV (columns: date, text)")
    parser.add_argument("--output", type=str, default=DATA_DIR)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("[fetch_sentiment] Loading articles...")
    articles = pd.read_csv(args.input, parse_dates=["date"])
    print(f"  {len(articles):,} articles loaded")

    print("[fetch_sentiment] Stage 1: Corpus partitioning...")
    partition = build_corpus_partition(articles)
    for mkt, daily in partition.items():
        total = sum(len(v) for v in daily.values())
        print(f"  {mkt}: {total:,} articles across {len(daily)} days")

    print("[fetch_sentiment] Stage 2: Loading FinBERT models...")
    enc_en = FinBERTEncoder("en")
    enc_ja = FinBERTEncoder("ja")

    print("[fetch_sentiment] Stage 3: Aggregating daily sentiment...")
    sentiment_dict = aggregate_daily_sentiment(partition, enc_en, enc_ja)

    # Save per-market sentiment
    for mkt, series in sentiment_dict.items():
        fpath = Path(args.output) / f"sentiment_{mkt}.csv"
        series.to_csv(fpath)
        print(f"  Saved {mkt} → {fpath}  (n={len(series)})")

    # Save global sentiment (simple mean fallback — train attention on val set)
    idx    = pd.date_range(
        min(s.index.min() for s in sentiment_dict.values()),
        max(s.index.max() for s in sentiment_dict.values()),
        freq="B",
    )
    local_df = pd.DataFrame(
        {m: s.reindex(idx).ffill().fillna(0.0)
         for m, s in sentiment_dict.items()},
        index=idx,
    )
    global_s = local_df.mean(axis=1).rename("global_sentiment")
    global_s.to_csv(Path(args.output) / "sentiment_global.csv")
    print(f"  Saved global sentiment → {args.output}/sentiment_global.csv")

    print("\n[fetch_sentiment] Done.")


if __name__ == "__main__":
    main()
