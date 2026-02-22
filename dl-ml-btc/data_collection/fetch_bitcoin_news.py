"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BITCOIN NEWS MEGA COLLECTOR v4.0 — 20K+ ARTICLES                   ║
║                                                                              ║
║  Data sources (all VPN-compatible):                                         ║
║  ✅ NewsAPI          — Last 30 days (~858 articles)                         ║
║  ✅ RSS Feeds        — Unlimited history (CoinDesk, CoinTelegraph, etc.)    ║
║  ✅ Reddit           — Historical Bitcoin posts                              ║
║  ✅ CoinGecko        — Historical BTC prices (back to 2013)                 ║
║  ✅ Alpha Vantage    — Crypto news sentiment (free API key)                 ║
║                                                                              ║
║  Target: 20,000+ articles | J+1 label | No data leakage                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import time
import json
import hashlib
import feedparser
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from textblob import TextBlob
from dateutil import parser as dateparser
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
#  LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────

load_dotenv()

NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")

NEWSAPI_URL       = "https://newsapi.org/v2/everything"
COINGECKO_URL     = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

OUTPUT_DIR        = "./../data/raw"
OUTPUT_MAIN       = os.path.join(OUTPUT_DIR, "btc_news.csv")
OUTPUT_BACKUP     = os.path.join(OUTPUT_DIR, f"btc_news_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
OUTPUT_JSON       = os.path.join(OUTPUT_DIR, f"btc_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# ─── RSS Feeds Bitcoin/Crypto (unlimited history, free, no API key) ──────────
RSS_FEEDS = [
    ("CoinDesk",         "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph",    "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
    ("Decrypt",          "https://decrypt.co/feed"),
    ("The Block",        "https://www.theblock.co/rss.xml"),
    ("Bitcoinist",       "https://bitcoinist.com/feed/"),
    ("NewsBTC",          "https://www.newsbtc.com/feed/"),
    ("CryptoSlate",      "https://cryptoslate.com/feed/"),
    ("AMBCrypto",        "https://ambcrypto.com/feed/"),
    ("BeInCrypto",       "https://beincrypto.com/feed/"),
    ("CryptoNews",       "https://cryptonews.com/news/feed/"),
    ("U.Today",          "https://u.today/rss"),
    ("Investing.com",    "https://www.investing.com/rss/news_285.rss"),
    ("Yahoo Finance",    "https://finance.yahoo.com/rss/headline?s=BTC-USD"),
    ("CoinGape",         "https://coingape.com/feed/"),
    ("ZyCrypto",         "https://zycrypto.com/feed/"),
    ("FXStreet",         "https://www.fxstreet.com/rss/cryptocurrencies"),
    ("Cryptopotato",     "https://cryptopotato.com/feed/"),
    ("LiveBitcoinNews",  "https://www.livebitcoinnews.com/feed/"),
    ("UseTheBitcoin",    "https://usethebitcoin.com/feed/"),
]

# ─── NewsAPI search queries ───────────────────────────────────
NEWSAPI_QUERIES = [
    "Bitcoin BTC price",
    "Bitcoin rally crash",
    "Bitcoin regulation SEC",
    "Bitcoin ETF approval",
    "Bitcoin mining halving",
    "Bitcoin whale institutional",
    "Bitcoin bear bull market",
    "Bitcoin adoption payment",
    "BTC technical analysis",
    "Bitcoin Federal Reserve inflation",
    "Bitcoin Ethereum crypto market",
    "Bitcoin microstrategy blackrock",
    "cryptocurrency market crash",
    "Bitcoin all time high",
    "Bitcoin correction support resistance",
    "Bitcoin futures options",
    "Bitcoin Lightning Network",
    "Bitcoin scam hack exchange",
    "crypto winter bull run",
    "Bitcoin dollar DXY correlation",
]


# ─────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────

def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CryptoMegaCollector/4.0)"})
    return s


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)                           # remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s\'\-\$\%\.\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def compute_sentiment(text):
    if not text:
        return 0.0
    try:
        return round(TextBlob(text).sentiment.polarity, 3)
    except Exception:
        return 0.0


def categorize(text):
    t = text.lower()
    rules = [
        ("REGULATORY", ["regulation", "sec", "law", "legal", "government",
                        "ban", "approve", "etf", "court", "ruling", "policy",
                        "congress", "cftc", "irs", "tax", "compliance"]),
        ("SECURITY",   ["hack", "security", "breach", "exploit", "scam",
                        "fraud", "theft", "attack", "vulnerability", "stolen",
                        "phishing", "ransomware", "ponzi"]),
        ("MACRO",      ["inflation", "fed", "interest rate", "economy", "recession",
                        "gdp", "dollar", "federal reserve", "rate hike", "cpi",
                        "unemployment", "treasury", "yield"]),
        ("TECH",       ["upgrade", "network", "protocol", "lightning", "taproot",
                        "mining", "halving", "blockchain", "node", "fork",
                        "segwit", "ordinals", "runes", "layer 2"]),
        ("ADOPTION",   ["adoption", "accept", "payment", "tesla", "institutional",
                        "microstrategy", "blackrock", "fund", "etf", "corporate",
                        "country", "national", "reserve", "el salvador"]),
        ("MARKET",     ["price", "trading", "volume", "whale", "exchange",
                        "pump", "dump", "rally", "surge", "bull", "bear",
                        "ath", "support", "resistance", "breakout", "correction",
                        "liquidation", "long", "short", "futures", "options"]),
    ]
    for cat, kws in rules:
        if any(kw in t for kw in kws):
            return cat
    return "OTHER"


def make_event_id(date, url):
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"btc_{date.replace('-', '')}_{h}"


def parse_date(raw_date):
    """Parse any date format and return YYYY-MM-DD."""
    try:
        if not raw_date:
            return None, None
        dt = dateparser.parse(str(raw_date))
        if dt:
            return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    return None, None


def normalize_article(title, date_str, timestamp_str, source, url, description=""):
    """Normalize an article into the standard format."""
    full_text = f"{title}. {description}".strip()
    return {
        "timestamp":   timestamp_str or date_str,
        "date":        date_str,
        "title":       title,
        "description": description,
        "full_text":   full_text,
        "source":      source,
        "url":         url,
    }


# ─────────────────────────────────────────────────────────────
#  SOURCE 1 : NEWSAPI
# ─────────────────────────────────────────────────────────────

class NewsAPICollector:

    def __init__(self, session):
        self.session = session

    def fetch_all(self):
        print(f"\n{'='*60}")
        print("[SOURCE 1] NewsAPI — Last 30 days")
        print(f"  {len(NEWSAPI_QUERIES)} queries planned")
        print(f"{'='*60}")

        all_items = []
        seen_urls = set()

        for i, query in enumerate(NEWSAPI_QUERIES, 1):
            print(f"  [{i:>2}/{len(NEWSAPI_QUERIES)}] '{query}'", end=" ")
            items = self._fetch_query(query)

            new = 0
            for art in items:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_items.append(art)
                    new += 1

            print(f"-> {new} new | Total: {len(all_items)}")
            time.sleep(1.2)

        print(f"  TOTAL NewsAPI: {len(all_items)} articles")
        return all_items

    def _fetch_query(self, query):
        items = []
        try:
            params = {
                "q":        query,
                "sortBy":   "publishedAt",
                "language": "en",
                "pageSize": 100,
                "page":     1,
                "apiKey":   NEWSAPI_KEY,
            }
            resp = self.session.get(NEWSAPI_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                return items

            for art in data.get("articles", []):
                pub = art.get("publishedAt", "")
                if not pub:
                    continue
                title = art.get("title", "") or ""
                if title in ("[Removed]", ""):
                    continue
                desc = art.get("description", "") or ""
                items.append(normalize_article(
                    title=title,
                    date_str=pub[:10],
                    timestamp_str=pub,
                    source=art.get("source", {}).get("name", "unknown"),
                    url=art.get("url", ""),
                    description=desc,
                ))
        except Exception as e:
            print(f"[ERROR NewsAPI] {e}", end=" ")

        return items


# ─────────────────────────────────────────────────────────────
#  SOURCE 2 : RSS FEEDS (UNLIMITED HISTORY)
# ─────────────────────────────────────────────────────────────

class RSSCollector:

    def __init__(self):
        # feedparser does not use requests — no proxy issue
        pass

    def fetch_all(self):
        print(f"\n{'='*60}")
        print(f"[SOURCE 2] RSS Feeds — {len(RSS_FEEDS)} sources")
        print(f"{'='*60}")

        all_items = []
        seen_urls = set()

        for i, (source_name, rss_url) in enumerate(RSS_FEEDS, 1):
            print(f"  [{i:>2}/{len(RSS_FEEDS)}] {source_name:<20}", end=" ")

            try:
                feed    = feedparser.parse(rss_url)
                entries = feed.entries
                new     = 0

                for entry in entries:
                    url = entry.get("link", "")
                    if not url or url in seen_urls:
                        continue

                    title = entry.get("title", "") or ""
                    if not title.strip():
                        continue

                    # Parse date
                    raw_date = (entry.get("published") or
                                entry.get("updated") or
                                entry.get("created") or "")
                    try:
                        dt       = dateparser.parse(str(raw_date)) if raw_date else None
                        date_str = dt.strftime("%Y-%m-%d") if dt else datetime.now().strftime("%Y-%m-%d")
                        ts_str   = dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else ""
                    except Exception:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        ts_str   = ""

                    # Description / summary — strip HTML
                    desc = (entry.get("summary", "") or
                            entry.get("description", "") or "")
                    desc = re.sub(r'<[^>]+>', ' ', desc)[:500]

                    seen_urls.add(url)
                    all_items.append(normalize_article(
                        title=title,
                        date_str=date_str,
                        timestamp_str=ts_str,
                        source=source_name,
                        url=url,
                        description=desc,
                    ))
                    new += 1

                print(f"-> {new} articles | Total: {len(all_items)}")

            except Exception as e:
                print(f"-> ERROR: {e}")

            time.sleep(0.5)

        print(f"  TOTAL RSS: {len(all_items)} articles")
        return all_items


# ─────────────────────────────────────────────────────────────
#  SOURCE 3 : REDDIT (public API, no key required)
# ─────────────────────────────────────────────────────────────

class RedditCollector:

    def __init__(self, session):
        self.session    = session
        self.base       = "https://www.reddit.com"
        self.headers    = {
            "User-Agent": "CryptoCollector/4.0 (educational project)"
        }
        self.subreddits = [
            "Bitcoin", "CryptoCurrency", "BitcoinMarkets",
            "CryptoMarkets", "investing", "finance"
        ]
        self.sorts      = ["hot", "new", "top", "rising"]

    def fetch_all(self):
        print(f"\n{'='*60}")
        print(f"[SOURCE 3] Reddit — {len(self.subreddits)} subreddits")
        print(f"{'='*60}")

        all_items = []
        seen_urls = set()

        for subreddit in self.subreddits:
            for sort in self.sorts:
                print(f"  r/{subreddit:<20} [{sort:<7}]", end=" ")

                try:
                    url    = f"{self.base}/r/{subreddit}/{sort}.json"
                    params = {"limit": 100, "raw_json": 1}
                    resp   = self.session.get(url, params=params,
                                              headers=self.headers, timeout=15)
                    resp.raise_for_status()
                    data  = resp.json()
                    posts = data.get("data", {}).get("children", [])
                    new   = 0

                    for post in posts:
                        p        = post.get("data", {})
                        post_url = f"https://reddit.com{p.get('permalink', '')}"

                        if post_url in seen_urls:
                            continue

                        title = p.get("title", "") or ""
                        if not title.strip():
                            continue

                        # Filter only Bitcoin-related posts
                        if not any(kw in title.lower() for kw in
                                   ["bitcoin", "btc", "crypto", "blockchain",
                                    "satoshi", "halving", "lightning"]):
                            continue

                        created  = p.get("created_utc", 0)
                        dt       = datetime.fromtimestamp(created, tz=timezone.utc)
                        date_str = dt.strftime("%Y-%m-%d")
                        ts_str   = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        desc     = p.get("selftext", "")[:300] or ""

                        seen_urls.add(post_url)
                        all_items.append(normalize_article(
                            title=title,
                            date_str=date_str,
                            timestamp_str=ts_str,
                            source=f"Reddit/r/{subreddit}",
                            url=post_url,
                            description=desc,
                        ))
                        new += 1

                    print(f"-> {new} BTC posts | Total: {len(all_items)}")
                    time.sleep(2.0)   # Reddit API: 1 req/2sec

                except Exception as e:
                    print(f"-> ERROR: {e}")
                    time.sleep(3.0)

        print(f"  TOTAL Reddit: {len(all_items)} posts")
        return all_items


# ─────────────────────────────────────────────────────────────
#  SOURCE 4 : ALPHA VANTAGE (free API key)
# ─────────────────────────────────────────────────────────────

class AlphaVantageCollector:

    def __init__(self, session):
        self.session = session

    def fetch_all(self):
        print(f"\n{'='*60}")
        print("[SOURCE 4] Alpha Vantage — News & Sentiment")
        print(f"{'='*60}")

        if ALPHA_VANTAGE_KEY == "demo":
            print("  WARNING: 'demo' key in use — limited results")
            print("  -> Sign up for free at alphavantage.co to get a real key")

        all_items = []
        topics    = ["blockchain", "cryptocurrency", "financial_markets"]

        for topic in topics:
            print(f"  Topic: {topic}", end=" ")
            try:
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers":  "CRYPTO:BTC",
                    "topics":   topic,
                    "limit":    1000,
                    "apikey":   ALPHA_VANTAGE_KEY,
                }
                resp = self.session.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                feed = data.get("feed", [])
                new  = 0

                for art in feed:
                    title = art.get("title", "") or ""
                    url   = art.get("url", "") or ""
                    if not title or not url:
                        continue

                    raw_time = art.get("time_published", "")
                    try:
                        dt       = datetime.strptime(raw_time, "%Y%m%dT%H%M%S")
                        date_str = dt.strftime("%Y-%m-%d")
                        ts_str   = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        ts_str   = ""

                    desc     = art.get("summary", "")[:400]
                    av_score = art.get("overall_sentiment_score", 0)

                    item                = normalize_article(
                        title=title,
                        date_str=date_str,
                        timestamp_str=ts_str,
                        source=art.get("source", "AlphaVantage"),
                        url=url,
                        description=desc,
                    )
                    item["av_sentiment"] = round(float(av_score), 3)
                    all_items.append(item)
                    new += 1

                print(f"-> {new} articles | Total: {len(all_items)}")
                time.sleep(15)   # Alpha Vantage free tier: 5 req/min

            except Exception as e:
                print(f"-> ERROR: {e}")

        print(f"  TOTAL Alpha Vantage: {len(all_items)} articles")
        return all_items


# ─────────────────────────────────────────────────────────────
#  PRICE COLLECTOR — COINGECKO (full history)
# ─────────────────────────────────────────────────────────────

class PriceCollector:

    def __init__(self, session):
        self.session = session

    def fetch_range(self, start_date, end_date):
        print(f"\n{'='*60}")
        print(f"[CoinGecko] BTC prices: {start_date} -> {end_date}")
        print(f"{'='*60}")

        try:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=2)
            dt_end   = datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=2)

            # CoinGecko free plan: 90-day limit per range request
            # Split into chunks if period > 90 days
            all_rows    = []
            chunk_start = dt_start
            chunk_days  = 85

            while chunk_start < dt_end:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), dt_end)
                print(f"  Chunk {chunk_start.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}...", end=" ")

                try:
                    url    = f"{COINGECKO_URL}/coins/bitcoin/market_chart/range"
                    params = {
                        "vs_currency": "usd",
                        "from": int(chunk_start.timestamp()),
                        "to":   int(chunk_end.timestamp()),
                    }
                    resp = self.session.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    rows = data.get("prices", [])
                    all_rows.extend(rows)
                    print(f"{len(rows)} points")
                    time.sleep(2.0)   # CoinGecko free rate limit

                except Exception as e:
                    print(f"Chunk ERROR: {e}")
                    time.sleep(5.0)

                chunk_start = chunk_end + timedelta(days=1)

            if not all_rows:
                return pd.DataFrame()

            df             = pd.DataFrame(all_rows, columns=["ts_ms", "price"])
            df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms")
            df["date"]     = df["datetime"].dt.strftime("%Y-%m-%d")
            df             = df.sort_values("datetime")
            df             = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

            # Day-over-day price change (context only)
            df["price_change_24h"] = (df["price"].pct_change() * 100).round(4)

            # ✅ LABEL SOURCE: next day price change (J+1) — no data leakage
            df["price_change_next_day"] = df["price_change_24h"].shift(-1).round(4)
            df["price_next_day"]        = df["price"].shift(-1).round(2)
            df["price"]                 = df["price"].round(2)

            print(f"  TOTAL: {len(df)} days of prices | BTC current: ${df['price'].iloc[-1]:,.0f}")
            return df[["date", "price", "price_next_day",
                        "price_change_24h", "price_change_next_day"]]

        except Exception as e:
            print(f"  CoinGecko global ERROR: {e}")
            return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

class MegaPipeline:

    def __init__(self):
        self.session = make_session()
        self.newsapi = NewsAPICollector(self.session)
        self.rss     = RSSCollector()
        self.reddit  = RedditCollector(self.session)
        self.alphav  = AlphaVantageCollector(self.session)
        self.prices  = PriceCollector(self.session)

    def run(self):
        print("\n╔" + "="*60 + "╗")
        print("║   BITCOIN MEGA PIPELINE v4.0 — TARGET 20K+ ARTICLES     ║")
        print("╚" + "="*60 + "╝\n")

        # ── Collect from all sources ─────────────────────────────────
        all_articles = []
        seen_urls    = set()

        def add_batch(batch, source_name):
            new = 0
            for art in batch:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(art)
                    new += 1
            print(f"\n  >> {source_name}: +{new} | GLOBAL TOTAL: {len(all_articles)}")

        # Source 1: NewsAPI
        add_batch(self.newsapi.fetch_all(), "NewsAPI")

        # Source 2: RSS Feeds (richest source for historical data)
        add_batch(self.rss.fetch_all(), "RSS Feeds")

        # Source 3: Reddit
        add_batch(self.reddit.fetch_all(), "Reddit")

        # Source 4: Alpha Vantage
        add_batch(self.alphav.fetch_all(), "Alpha Vantage")

        print(f"\n{'='*60}")
        print(f"RAW TOTAL ALL SOURCES: {len(all_articles)} articles")
        print(f"{'='*60}")

        if not all_articles:
            print("ERROR: No articles collected.")
            return None

        # ── Global cleanup ───────────────────────────────────────────
        print("\nCleaning and deduplicating...")
        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset=["url"], keep="first")
        df = df.dropna(subset=["title", "date"])
        df = df[df["title"].str.strip() != ""]
        df = df[df["title"] != "[Removed]"]
        df = df[df["date"] >= "2017-01-01"]    # Keep data from 2017 onwards only
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  Articles after cleanup: {len(df)}")

        # ── Bitcoin prices ───────────────────────────────────────────
        print("\nFetching Bitcoin prices...")
        price_df = self.prices.fetch_range(df["date"].min(), df["date"].max())

        if not price_df.empty:
            df = df.merge(price_df, on="date", how="left")
        else:
            df["price"]                 = None
            df["price_next_day"]        = None
            df["price_change_24h"]      = 0.0
            df["price_change_next_day"] = 0.0

        # ── NLP Features ─────────────────────────────────────────────
        print("\nComputing NLP features...")
        print("  Cleaning text for BERT...")
        df["text_clean"]      = df["full_text"].apply(clean_text)
        print("  Computing TextBlob sentiment...")
        df["sentiment_score"] = df["text_clean"].apply(compute_sentiment)
        print("  Categorizing articles...")
        df["category"]        = df["full_text"].apply(categorize)
        print("  Generating event IDs...")
        df["event_id"]        = df.apply(lambda r: make_event_id(r["date"], r["url"]), axis=1)

        # ── Corrected J+1 label ──────────────────────────────────────
        print("\nLabeling (next-day price J+1)...")
        df["label"] = df["price_change_next_day"].apply(
            lambda x: 1 if pd.notna(x) and x > 0 else 0
        )
        df["severity"] = df["price_change_next_day"].abs().apply(
            lambda x: min(10, max(1, int(abs(x)))) if pd.notna(x) else 1
        )

        # ── Final column order ───────────────────────────────────────
        cols = [
            "event_id", "timestamp", "date",
            "title", "text_clean",
            "source", "url", "category",
            "sentiment_score",
            "price", "price_next_day",
            "price_change_24h", "price_change_next_day",
            "label", "severity",
        ]
        cols = [c for c in cols if c in df.columns]
        df   = df[cols]

        # ── Final report ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  FINAL DATASET — FULL REPORT")
        print(f"{'='*60}")
        print(f"  Total articles        : {len(df):,}")
        print(f"  Date range            : {df['date'].min()} -> {df['date'].max()}")
        print(f"  Labels UP    (1)      : {(df['label']==1).sum():,}  ({(df['label']==1).mean()*100:.1f}%)")
        print(f"  Labels DOWN  (0)      : {(df['label']==0).sum():,}  ({(df['label']==0).mean()*100:.1f}%)")
        print(f"  Average sentiment     : {df['sentiment_score'].mean():+.3f}")
        print(f"  Articles without price: {df['price'].isna().sum():,}")
        print(f"  Unique sources        : {df['source'].nunique()}")
        print(f"\n  Category breakdown:")
        for cat, n in df["category"].value_counts().items():
            bar = "#" * int(n / len(df) * 25)
            print(f"    {cat:<15} {bar:<25} {n:>5} ({n/len(df)*100:.1f}%)")
        print(f"\n  Top 10 sources:")
        for src, n in df["source"].value_counts().head(10).items():
            print(f"    {str(src):<30} {n:>5}")
        print(f"{'='*60}\n")

        return df


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    # Validate API keys loaded from .env
    if not NEWSAPI_KEY:
        print("ERROR: NEWSAPI_KEY not found in .env file")
        return
    if not ALPHA_VANTAGE_KEY:
        print("ERROR: ALPHA_VANTAGE_KEY not found in .env file")
        return

    pipeline = MegaPipeline()
    df       = pipeline.run()

    if df is None:
        print("Pipeline failed.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(OUTPUT_MAIN,   index=False, encoding="utf-8")
    df.to_csv(OUTPUT_BACKUP, index=False, encoding="utf-8")
    df.to_json(OUTPUT_JSON,  orient="records", indent=2, force_ascii=False)

    print("FILES SAVED:")
    print(f"  Main CSV    : {os.path.abspath(OUTPUT_MAIN)}")
    print(f"  Backup CSV  : {os.path.abspath(OUTPUT_BACKUP)}")
    print(f"  JSON        : {os.path.abspath(OUTPUT_JSON)}")

    print("\nPreview (first 5 rows):")
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 200)
    print(df[["date", "title", "label", "price_change_next_day",
              "sentiment_score", "category", "source"]].head())

    # Articles per year breakdown
    df["year"] = df["date"].str[:4]
    print("\nArticles per year:")
    print(df.groupby("year").size().to_string())

    print(f"\nPIPELINE COMPLETE — {len(df):,} articles collected!\n")


if __name__ == "__main__":
    main()