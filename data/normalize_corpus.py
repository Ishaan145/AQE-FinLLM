"""
Usage:
    python -m data.normalize_corpus --input raw_news.csv --output articles.csv
    python -m data.normalize_corpus --input news.jsonl --output articles.csv
"""
"""
data/normalize_corpus.py — Convert and COMBINE downloaded news datasets
(Kaggle CSV / HuggingFace JSONL) into the single date,text articles.csv
that fetch_sentiment.py + Module A consume.

Auto-loads known dataset filenames from the project root if present,
and/or accepts explicit --input files. All are merged, normalized,
deduplicated, and written to articles.csv.

Usage:
    # Auto mode: picks up known files in project root
    python -m data.normalize_corpus

    # Explicit mode: one or more files
    python -m data.normalize_corpus --input sp500_headlines_2008_2024.csv nifty_news_2003_2020.csv

    # Override columns if auto-detect fails
    python -m data.normalize_corpus --input foo.csv --date_col publishedAt --text_col headline
"""
import argparse
import sys
import json
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Known dataset filenames expected in the project root.
# Add more here as you download them.
KNOWN_DATASETS = [
    "sp500_headlines_2008_2024.csv",   # dyutidasmahaptra S&P 500 2008-2024
    "nifty_news_2003_2020.csv",        # hkapoor Indian financial news 2003-2020
]

DATE_CANDIDATES = [
    "date", "Date", "datetime", "published", "publish_date",
    "published_at", "publishedAt", "time", "timestamp", "Datetime",
    "pub_date", "DATE", "Date/Time",
]
TEXT_CANDIDATES = [
    "text", "headline", "Headline", "title", "Title", "headlines",
    "Headlines", "news", "content", "Content", "summary", "article",
    "Article", "description", "Description", "body", "News",
]


def _pick(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows)
    return pd.read_csv(path, on_bad_lines="skip", engine="python",
                       encoding_errors="ignore")


def normalize_one(path: Path, date_col=None, text_col=None) -> pd.DataFrame:
    print(f"\n[normalize] Loading {path.name} ...")
    df = load_any(path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    dcol = date_col or _pick(list(df.columns), DATE_CANDIDATES)
    tcol = text_col or _pick(list(df.columns), TEXT_CANDIDATES)

    if dcol is None or tcol is None:
        print(f"  [!] Could not detect columns for {path.name} "
              f"(date={dcol}, text={tcol}). Skipping this file. "
              f"Re-run with --date_col / --text_col to force.")
        return pd.DataFrame(columns=["date", "text"])

    print(f"  Using date='{dcol}', text='{tcol}'")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[dcol], errors="coerce", utc=True)
    out["text"] = df[tcol].astype(str).str.strip()

    # Concat extra descriptive text if a secondary column exists
    extra = _pick(
        [c for c in df.columns if c != tcol],
        ["summary", "description", "abstract", "body", "content"],
    )
    if extra:
        out["text"] = out["text"] + ". " + df[extra].astype(str).fillna("")

    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out[out["text"].str.len() > 20]
    print(f"  -> {len(out):,} usable rows")
    return out[["date", "text"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", default=None,
                        help="One or more raw dataset files. "
                             "If omitted, auto-loads KNOWN_DATASETS.")
    parser.add_argument("--output", default="articles.csv")
    parser.add_argument("--date_col", default=None)
    parser.add_argument("--text_col", default=None)
    args = parser.parse_args()

    # Build the list of files to process
    if args.input:
        files = [Path(f) for f in args.input]
    else:
        files = []
        for name in KNOWN_DATASETS:
            p = _PROJECT_ROOT / name
            if p.exists():
                files.append(p)
            else:
                print(f"[normalize] (skip) not found: {name}")

    if not files:
        print("[normalize] No input files found. Download a dataset to "
              f"{_PROJECT_ROOT} and name it one of: {KNOWN_DATASETS}")
        sys.exit(1)

    frames = []
    for f in files:
        if not f.exists():
            print(f"[normalize] (skip) missing: {f}")
            continue
        frames.append(
            normalize_one(f, args.date_col, args.text_col)
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date"])
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.sort_values("date").reset_index(drop=True)

    out_path = _PROJECT_ROOT / args.output
    combined.to_csv(out_path, index=False)

    print(f"\n[normalize] COMBINED {len(frames)} dataset(s)")
    print(f"  Saved {len(combined):,} articles -> {out_path}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    n_days = combined["date"].nunique()
    print(f"  Unique days: {n_days:,}")
    if n_days < 200:
        print("  [!] WARNING: few unique days — sentiment will be sparse.")
    else:
        print("  OK: dense enough for meaningful sentiment training.")


if __name__ == "__main__":
    main()