# api/fetch_blockchain.py - Fetching blockchain metrics

import requests
import pandas as pd
import time
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLOCKCHAIN_OUTPUT_FILE, BLOCKCHAIN_DAYS


class BlockchainMetricsFetcher:
    """Fetches Bitcoin on-chain metrics from free APIs."""
    
    def __init__(self):
        self.blockchain_info_base = "https://api.blockchain.info"
        self.mempool_space_base = "https://mempool.space/api/v1"
        
    def fetch_blockchain_info_stats(self, days=None):
        """
        Fetches network stats from Blockchain.info (FREE).
        
        Args:
            days: Number of history days (default since 2018)
        """
        if days is None:
            days = BLOCKCHAIN_DAYS
            
        print(f"📡 Fetching Blockchain.info data for {days} days...")
        
        endpoints = {
            'market_price_usd': 'market-price',
            'hash_rate_th_s': 'hash-rate',
            'difficulty': 'difficulty',
            'tx_count_daily': 'n-transactions',
            'tx_fees_btc': 'transaction-fees',
            'avg_block_size_mb': 'avg-block-size',
            'mempool_size_bytes': 'mempool-size',
            'total_btc_supply': 'total-bitcoins'
        }
        
        data = {}
        
        for metric_name, endpoint in endpoints.items():
            try:
                url = f"{self.blockchain_info_base}/charts/{endpoint}?timespan={days}days&format=json"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                json_data = response.json()
                values = json_data.get('values', [])
                
                data[metric_name] = {
                    datetime.fromtimestamp(item['x']).strftime('%Y-%m-%d'): item['y']
                    for item in values
                }
                
                print(f"  ✅ {metric_name}: {len(values)} points")
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error for {metric_name}: {e}")
                data[metric_name] = {}
        
        return data
    
    def fetch_mempool_space_stats(self):
        """Fetches current mempool stats (FREE)."""
        try:
            url = f"{self.mempool_space_base}/mining/hashrate/1y"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            print("✅ Mempool.space data fetched")
            return response.json()
        except Exception as e:
            print(f"❌ Mempool.space error: {e}")
            return None
    
    def combine_to_dataframe(self, blockchain_data):
        """Converts data to unified DataFrame."""
        df = pd.DataFrame(blockchain_data)
        
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.ffill(inplace=True)  # Forward fill for missing values
        
        return df
    
    def save_data(self, df):
        """Saves DataFrame to CSV."""
        os.makedirs(os.path.dirname(BLOCKCHAIN_OUTPUT_FILE), exist_ok=True)
        df.to_csv(BLOCKCHAIN_OUTPUT_FILE, index_label='date')
        print(f"\n✅ Blockchain data saved: {BLOCKCHAIN_OUTPUT_FILE}")
        print(f"📊 {len(df)} days of data")


def main():
    """Main function."""
    print("=" * 60)
    print("  FETCHING BLOCKCHAIN METRICS")
    print("=" * 60)
    
    fetcher = BlockchainMetricsFetcher()
    
    # Fetching from Blockchain.info
    blockchain_data = fetcher.fetch_blockchain_info_stats()
    
    # Converting to DataFrame
    df = fetcher.combine_to_dataframe(blockchain_data)
    
    if not df.empty:
        fetcher.save_data(df)
        print(f"📅 Period: {df.index.min()} → {df.index.max()}")
    else:
        print("❌ No data fetched")


if __name__ == '__main__':
    main()
