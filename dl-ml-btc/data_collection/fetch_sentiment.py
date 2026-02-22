# api/fetch_sentiment.py - Fetching sentiment data

import requests
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENTIMENT_OUTPUT_FILE


class SentimentFetcher:
    """Fetches Fear & Greed Index crypto (FREE, Alternative.me)."""
    
    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        
    def fetch_fear_greed_index(self, days=None):
        """
        Fetches Fear & Greed Index from Alternative.me.
        FREE API - No key required.
        
        Args:
            days: Number of days (0 for today, or number of days)
        """
        print("😨 Fetching Fear & Greed Index...")
        
        try:
            # The API can return up to ~500 days of history
            params = {'limit': 0} if days is None else {'limit': min(days, 500)}
            
            response = requests.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('data'):
                df = pd.DataFrame(data['data'])
                # Convert timestamp to int first, handle errors
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                # Filter out invalid timestamps (too large or negative)
                df = df[df['timestamp'].between(0, 2147483647)]  # Max valid Unix timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Rename columns
                df.rename(columns={
                    'value': 'fear_greed_value',
                    'value_classification': 'fear_greed_label'
                }, inplace=True)
                
                # Convert value to float
                df['fear_greed_value'] = df['fear_greed_value'].astype(float)
                
                print(f"  ✅ {len(df)} points fetched")
                print(f"  📅 Period: {df.index.min()} → {df.index.max()}")
                
                return df[['fear_greed_value', 'fear_greed_label']]
            else:
                print("  ❌ No data in API response")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching: {e}")
            return pd.DataFrame()
    
    def save_data(self, df):
        """Saves data to CSV."""
        if df.empty:
            print("⚠️ No data to save")
            return
            
        os.makedirs(os.path.dirname(SENTIMENT_OUTPUT_FILE), exist_ok=True)
        df.to_csv(SENTIMENT_OUTPUT_FILE, index_label='date')
        print(f"\n✅ Sentiment data saved: {SENTIMENT_OUTPUT_FILE}")
        print(f"📊 {len(df)} days of data")


def main():
    """Main function."""
    print("=" * 60)
    print("  FETCHING SENTIMENT DATA")
    print("=" * 60)
    
    fetcher = SentimentFetcher()
    
    # Fetching (limited to 500 days by API)
    sentiment_df = fetcher.fetch_fear_greed_index()
    
    if not sentiment_df.empty:
        fetcher.save_data(sentiment_df)
    else:
        print("❌ No data fetched")


if __name__ == '__main__':
    main()
