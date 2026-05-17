"""
data/build_corpus.py — Build a free, license-clean news corpus
from public NSE / SEBI / RBI announcement RSS feeds.

Produces articles.csv with columns: date, text
Usage:
    python -m data.build_corpus --output articles.csv --years 2015 2025
"""
import argparse
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import feedparser   # pip install feedparser

# Public, free, no-auth feeds (India-market relevant)
FEEDS = {
    "NSE": "https://www.nseindia.com/feeds/announcements.xml",
    "RBI_press":   "https://www.rbi.org.in/pressreleases_rss.xml",
    "SEBI":        "https://www.sebi.gov.in/sebirss.xml",
    # Economic Times (these worked for you — keep)
    "ET_markets":   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET_econ":      "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "ET_stocks":    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET_forex":     "https://economictimes.indiatimes.com/markets/forex/rssfeeds/1898055246.cms",
    # Moneycontrol (India markets)
    "MC_markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "MC_business":  "https://www.moneycontrol.com/rss/business.xml",
    "MC_economy":   "https://www.moneycontrol.com/rss/economy.xml",
    # Livemint
    "Mint_markets": "https://www.livemint.com/rss/markets",
    "Mint_money":   "https://www.livemint.com/rss/money",
    # The Hindu BusinessLine
    "BL_markets":   "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    # Yahoo Finance (global)
    "YF_headline":  "https://finance.yahoo.com/news/rssindex",
    # CNBC (global markets)
    "CNBC_finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    # Investing.com (global)
    "INV_news":     "https://www.investing.com/rss/news_25.rss",
}


def fetch_feed(name: str, url: str) -> list[dict]:
    """Parse one RSS feed → list of {date, text} dicts."""
    rows = []
    try:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            # Date: prefer published, fall back to updated
            dt = entry.get("published_parsed") or entry.get("updated_parsed")
            if dt is None:
                continue
            date = datetime(*dt[:6]).strftime("%Y-%m-%d")
            # Text: title + summary (strip HTML crudely)
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title}. {summary}".replace("<p>", " ").replace("</p>", " ")
            text = " ".join(text.split())  # collapse whitespace
            if len(text) > 20:
                rows.append({"date": date, "text": text, "source": name})
        print(f"  [{name}] {len(rows)} articles")
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="articles.csv")
    args = parser.parse_args()

    print("[build_corpus] Fetching public RSS feeds...")
    all_rows = []
    for name, url in FEEDS.items():
        all_rows.extend(fetch_feed(name, url))
        time.sleep(1)  # be polite to servers

    if not all_rows:
        print("[build_corpus] No articles fetched. Check network/feeds.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["text"]).sort_values("date")
    df = df[["date", "text"]]   # fetch_sentiment.py expects these
    df.to_csv(args.output, index=False)
    print(f"[build_corpus] Saved {len(df)} articles -> {args.output}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()