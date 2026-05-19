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
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import feedparser   # pip install feedparser

# Public, free, no-auth feeds (India-market relevant)
FEEDS = {
    # ========================================================
    # INDIA MARKETS
    # ========================================================
    
    # Economic Times — reliable, worked before
    "ET_markets":   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET_econ":      "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "ET_stocks":    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET_forex":     "https://economictimes.indiatimes.com/markets/forex/rssfeeds/1898055246.cms",
    # Moneycontrol
    "MC_markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "MC_business":  "https://www.moneycontrol.com/rss/business.xml",
    "MC_economy":   "https://www.moneycontrol.com/rss/economy.xml",
    # The Hindu BusinessLine
    "BL_markets":   "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    # NSE / SEBI relevant news
    "BusinessToday_markets": "https://www.businesstoday.in/rss/markets",
    # Livemint
    "Mint_markets": "https://www.livemint.com/rss/markets",
    "Mint_money":   "https://www.livemint.com/rss/money",
    
    # ========================================================
    # US / S&P 500 / MACRO GLOBAL
    # ========================================================

    # Yahoo Finance
    "YahooFinance_global": "https://finance.yahoo.com/news/rssindex",
    # CNBC Markets
    "CNBC_markets":  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    # CNBC Finance
    "CNBC_finance":  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    # MarketWatch
    "MarketWatch_topstories":  "http://feeds.marketwatch.com/marketwatch/topstories/",
    # Seeking Alpha
    "SeekingAlpha_market":  "https://seekingalpha.com/market_currents.xml",
    # Federal Reserve / Macro
    "Fed_news": "https://www.federalreserve.gov/feeds/press_all.xml",
    # Reuters Markets
    "Reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    # Bloomberg (limited public)
    "Bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",

    # ========================================================
    # JAPAN / NIKKEI 225 / BOJ
    # ========================================================

    # Nikkei Asia
    "NikkeiAsia":  "https://asia.nikkei.com/rss/feed/nar",
    # Japan Times Business
    "JapanTimes_business": "https://www.japantimes.co.jp/business/feed/",
    # BOJ Releases
    "BOJ_news": "https://www.boj.or.jp/en/rss/whatsnew.rdf",
    # NHK Business
    "NHK_business": "https://www3.nhk.or.jp/rss/news/cat5.xml",

    # ========================================================
    # CRYPTO / BITCOIN
    # ========================================================

    # CoinDesk
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    # CoinTelegraph
    "CoinTelegraph": "https://cointelegraph.com/rss",
    # Bitcoin Magazine
    "BitcoinMagazine": "https://bitcoinmagazine.com/.rss/full/",
    # CryptoCompare
    "CryptoCompare": "https://www.cryptocompare.com/media/rss/news/",

    # ========================================================
    # GLOBAL MACRO / MULTI-ASSET
    # ========================================================

    # Global
    "YF_headline":  "https://finance.yahoo.com/news/rssindex",
    "INV_news":     "https://www.investing.com/rss/news_25.rss",
    # Yahoo Finance (global)
    "YF_headline":  "https://finance.yahoo.com/news/rssindex",
    # CNBC (global markets)
    "CNBC_finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    # Investing.com (global)
    "INV_news":     "https://www.investing.com/rss/news_25.rss",
    # Investing.com
    "Investing_global": "https://www.investing.com/rss/news_25.rss",
    # IMF
    "IMF_news": "https://www.imf.org/en/News/RSS",
    # World Bank
    "WorldBank": "https://blogs.worldbank.org/feed",
    # OECD
    "OECD": "https://www.oecd.org/newsroom/rss.xml",
    # Financial Times
    "FT_markets": "https://www.ft.com/markets?format=rss",
    # WSJ Markets
    "WSJ_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}


def fetch_feed(name: str, url: str, timeout: int = 12) -> list[dict]:
    """Parse one RSS feed -> list of {date, text} dicts.
    Uses requests with a hard timeout so dead/blocking hosts
    (e.g. NSE) cannot hang the whole script."""
    rows = []
    headers = {
        # Some hosts (NSE, Investing) reject non-browser agents
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        for entry in parsed.entries:
            dt = entry.get("published_parsed") or entry.get("updated_parsed")
            if dt is None:
                continue
            date = datetime(*dt[:6]).strftime("%Y-%m-%d")
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title}. {summary}".replace("<p>", " ").replace("</p>", " ")
            text = " ".join(text.split())
            if len(text) > 20:
                rows.append({"date": date, "text": text, "source": name})
        print(f"  [{name}] {len(rows)} articles")
    except requests.exceptions.Timeout:
        print(f"  [{name}] TIMEOUT after {timeout}s — skipped")
    except requests.exceptions.RequestException as e:
        print(f"  [{name}] FAILED ({type(e).__name__}) — skipped")
    except Exception as e:
        print(f"  [{name}] ERROR: {e} — skipped")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="rss_live.csv")
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