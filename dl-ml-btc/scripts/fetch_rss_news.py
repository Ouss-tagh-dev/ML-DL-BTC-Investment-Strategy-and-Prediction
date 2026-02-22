"""
Advanced RSS Crypto News Collector (Zero Cost, Uncapped)
Fetches thousands of news items from top 25 crypto feeds.
Enriches with BTC price action using Binance Public API for BERT training labels.
"""

import feedparser
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import time
import os
import re
import numpy as np

# --- CONFIGURATION ---
RSS_FEEDS = [
    # Major Outlets
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptopotato.com/feed/",
    "https://news.bitcoin.com/feed/",
    "https://beincrypto.com/feed/",
    "https://blockworks.co/feed",
    "https://thedefiant.io/feed",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed/",
    "https://dailyhodl.com/feed/",
    "https://ambcrypto.com/feed/",
    "https://zycrypto.com/feed/",
    "https://u.today/rss",
    "https://coinjournal.net/news/feed/",
    "https://bitcoinist.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://cryptonews.com/news/feed/",
    "https://blockchain.news/RSS/",
    # Aggregators / Specific
    # Reddit feeds removed due to noise
]

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news_rss_2024_2026.csv")

def clean_html(raw_html):
    """Remove HTML tags from summary"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def clean_summary(text):
    """Remove boilerplate text from summaries"""
    # Bitcoin Magazine Footer
    text = re.sub(r'This post .* first appeared on Bitcoin Magazine.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Blockchain News Read More
    text = re.sub(r'\(Read More\)', '', text, flags=re.IGNORECASE)
    # Source Headers
    text = re.sub(r'^(Bitcoin Magazine|CoinDesk|Decrypt)\n+', '', text, flags=re.IGNORECASE)
    return text.strip()

NOISE_PATTERNS = [
    r'Price Prediction Summary',
    r'Short-term target \(1 week\)',
    r'What Crypto Ana\.\.\.',
    r'Price Analysis',
]

def is_informative(text):
    """Filter out noise and price predictions"""
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False
    return True

def categorize_news(text):
    """Simple keyword-based categorization"""
    text = text.lower()
    if any(x in text for x in ['sec', 'regulation', 'law', 'ban', 'tax', 'legal', 'congress']):
        return 'REGULATORY'
    if any(x in text for x in ['inflation', 'fed', 'rate', 'macro', 'economy', 'recession']):
        return 'MACRO'
    if any(x in text for x in ['hack', 'exploit', 'scam', 'fraud', 'stolen', 'security']):
        return 'SECURITY'
    if any(x in text for x in ['upgrade', 'fork', 'taproot', 'lightning', 'miner', 'hashrate']):
        return 'TECH'
    if any(x in text for x in ['etf', 'blackrock', 'fidelity', 'adoption', 'payment', 'accept']):
        return 'ADOPTION'
    return 'MARKET'

def fetch_rss_news():
    """Fetch and parse all RSS feeds"""
    print(f"📡 Scanning {len(RSS_FEEDS)} RSS feeds...")
    all_news = []
    
    for url in RSS_FEEDS:
        try:
            print(f"  - Parsing: {url}")
            feed = feedparser.parse(url)
            source_name = getattr(feed.feed, 'title', url.split('/')[2])
            
            for entry in feed.entries:
                # Extract date
                pub_date = None
                if 'published_parsed' in entry:
                    try:
                        dt_obj = datetime(*entry.published_parsed[:6])
                        pub_date = pd.Timestamp(dt_obj).tz_localize('UTC')
                    except:
                        pub_date = pd.Timestamp.now(tz='UTC')
                elif 'updated_parsed' in entry:
                    try:
                        dt_obj = datetime(*entry.updated_parsed[:6])
                        pub_date = pd.Timestamp(dt_obj).tz_localize('UTC')
                    except:
                        pub_date = pd.Timestamp.now(tz='UTC')
                else:
                    pub_date = pd.Timestamp.now(tz='UTC')
                
                summary = entry.get('summary', entry.get('description', ''))
                summary = clean_html(summary)
                summary = clean_summary(summary) # Apply new cleaning
                
                # If summary is empty, use title
                if len(summary) < 20:
                    summary = entry.get('title', '')
                
                # Filter uninformative news
                if not is_informative(summary):
                    continue

                all_news.append({
                    'timestamp': pub_date,
                    'title': entry.get('title', ''),
                    'summary': summary,
                    'source': source_name,
                    'url': entry.get('link', ''),
                    'category': categorize_news(summary + " " + entry.get('title', ''))
                })
        except Exception as e:
            print(f"  ❌ Error parsing {url}: {e}")
            
    print(f"\n✅ Parsed {len(all_news)} items.")
    if not all_news:
        return pd.DataFrame()
    return pd.DataFrame(all_news)

def fetch_btc_prices_kraken(start_date_dt):
    """Fetch hourly BTC prices from Kraken Public API (no geo-restrictions)"""
    print(f"📈 Fetching BTC prices from Kraken (since {start_date_dt.date()})...")

    # Kraken uses pair name XBTUSD or XXBTZUSD
    url = "https://api.kraken.com/0/public/OHLC"
    all_klines = []
    
    # Kraken 'since' is in seconds
    since = int(start_date_dt.timestamp())
    end_ts = int(datetime.now().timestamp())
    
    # Safety break to prevent infinite loops
    max_loops = 100
    loops = 0

    while since < end_ts and loops < max_loops:
        loops += 1
        params = {
            'pair': 'XBTUSD',
            'interval': 60,  # 60 minutes = 1h
            'since': since
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                 print(f"  ⚠ Kraken error status: {resp.status_code}")
                 time.sleep(5)
                 continue

            data = resp.json()

            if data.get('error'):
                # Handle error (e.g. rate limit EAPI:Rate limit exceeded)
                errs = data['error']
                print(f"  ⚠ Kraken error: {errs}")
                if any("Rate limit" in e for e in errs):
                    time.sleep(10)
                    continue
                break

            result = data.get('result', {})
            # Kraken returns dictionary with pair name key (e.g. XXBTZUSD) and 'last'
            # We need to find the pair key
            keys = list(result.keys())
            if 'last' in keys: keys.remove('last')
            if not keys:
                break
                
            pair_key = keys[0]
            ohlc = result[pair_key]

            if not ohlc:
                break

            all_klines.extend(ohlc)

            # Kraken returns 'last' as the next since value (timestamp for next call)
            # The 'last' field is crucial for pagination
            last = result.get('last')
            
            # If last is same as since, we aren't moving forward
            if last and int(last) > since:
                since = int(last)
            else:
                # If 'last' didn't move, force it by taking the last entry's time + 1
                last_entry_time = int(ohlc[-1][0])
                if last_entry_time >= since:
                    since = last_entry_time + 1
                else:
                    break
            
            # Print progress from last chunk
            last_dt = datetime.fromtimestamp(int(ohlc[-1][0]))
            print(f"  - Fetched chunk ending {last_dt}...")
            
            time.sleep(1.5)  # Respect rate limit

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            break

    if not all_klines:
        return None

    print(f"  ✔ Fetched {len(all_klines)} klines using Kraken")

    data = []
    for k in all_klines:
        # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
        # time is unix timestamp (seconds, sometimes float)
        data.append({
            'timestamp': int(k[0]),
            'Open': float(k[1]),
            'Close': float(k[4])
        })

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df

def label_news_with_price_action(news_df, price_df):
    """Match news timestamps to future returns"""
    print("🏷️ Labeling news with market impact...")
    
    labeled_data = []
    
    # Ensure price_df is sorted
    price_df = price_df.sort_values('timestamp').reset_index(drop=True)
    price_times = price_df['timestamp'].values
    
    # Debug timestamps
    # print(f"News sample ts: {news_df.iloc[0]['timestamp']}")
    # print(f"Price sample ts: {price_df.iloc[0]['timestamp']}")

    for _, row in news_df.iterrows():
        news_ts = row['timestamp']
        
        # Normalize timezone to UTC
        # Normalize timezone to UTC then Strip for Numpy comparison (price_times is datetime64[ns] naive-like)
        if news_ts.tzinfo is None:
            news_ts = news_ts.replace(tzinfo=timezone.utc) # Assume UTC if naive
        else:
            news_ts = news_ts.astimezone(timezone.utc)
            
        # Crucial: Drop tzinfo because price_df['timestamp'].values is numpy.datetime64[ns] (naive wall time)
        news_ts = news_ts.replace(tzinfo=None)
            
        # Find index
        idx = np.searchsorted(price_times, news_ts)
        
        if idx >= len(price_df) - 24: 
            continue
            
        start_price_row = price_df.iloc[idx]
        current_price = start_price_row['Open']
        
        # 24h later
        end_idx = min(idx + 24, len(price_df) - 1)
        end_price_row = price_df.iloc[end_idx]
        price_24h = end_price_row['Close']
        
        pct_24h = ((price_24h - current_price) / current_price) * 100
        
        direction = 'NEUTRAL'
        if pct_24h > 1.5: direction = 'UP'
        elif pct_24h < -1.5: direction = 'DOWN'
        
        severity = min(10, int(abs(pct_24h) * 2)) 
        
        row['price_change_24h'] = round(pct_24h, 2)
        row['direction'] = direction
        row['severity'] = severity
        row['bitcoin_price_usd'] = round(current_price, 2)
        row['date'] = news_ts.strftime('%Y-%m-%d')
        
        labeled_data.append(row)
        
    return pd.DataFrame(labeled_data)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Fetch News
    df_news = fetch_rss_news()
    if df_news.empty:
        print("❌ No news found.")
        return

    df_news = df_news.drop_duplicates(subset=['title'])
    
    # Filter 2 years
    # Filter 2 years
    # Robust Comparison (All are UTC Timestamps now)
    try:
        min_date_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=720)
        
        # Filter
        df_news = df_news[df_news['timestamp'] > min_date_dt]
        print(f"📅 News count after date filter: {len(df_news)}")
        
    except Exception as e:
        print(f"⚠️ Filter Error: {e}")
        print("Fallback: Using all news.")
    
    if df_news.empty:
        print("❌ No news in date range.")
        return

    # 2. Fetch Prices (Kraken)
    df_prices = fetch_btc_prices_kraken(min_date_dt)
    
    if df_prices is None or df_prices.empty:
        print("❌ All price fetch methods failed. Saving unlabeled news...")
        # Save unlabeled
        unlabeled_file = OUTPUT_FILE.replace(".csv", "_unlabeled.csv")
        df_news.to_csv(unlabeled_file, index=False)
        print(f"💾 Saved {len(df_news)} unlabeled news items to {unlabeled_file}")
        return
        
    # 3. Label
    df_labeled = label_news_with_price_action(df_news, df_prices)
    
    if df_labeled.empty:
        print("❌ Labeling failed (no matching price data).")
        return
        
    # 4. Save
    cols = ['timestamp', 'date', 'summary', 'source', 'category', 'price_change_24h', 'direction', 'severity', 'bitcoin_price_usd', 'url', 'title']
    
    final_df = df_labeled[cols]
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved {len(final_df)} labeled news items to {OUTPUT_FILE}")
    print("Sample:")
    print(final_df[['date', 'summary', 'direction', 'severity']].head())

if __name__ == "__main__":
    main()
