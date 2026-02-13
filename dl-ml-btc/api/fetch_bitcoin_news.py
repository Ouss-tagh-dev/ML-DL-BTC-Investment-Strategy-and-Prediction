"""
Enhanced Real Crypto News & Market Data Collector
Target: 5,000 - 100,000+ REAL events
Uses CryptoPanic v2 API with pagination
"""

import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import time
import json

class MegaCryptoDataCollector:
    def __init__(self):
        # APIs
        self.cryptopanic_url = "https://cryptopanic.com/api/developer/v2/posts/"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.newsapi_url = "https://newsapi.org/v2/everything"
        
        self.cryptopanic_key = "2d98b6ac5d2f1bb00fa53dcbc3a04ab8120b7876"  
        self.newsapi_key = "4479de533c1e4408b22c5e1f2c52dfe8"  
        
        # Create a session that bypasses system proxy settings
        self.session = requests.Session()
        self.session.proxies = {'http': None, 'https': None}
        self.session.trust_env = False  # Ignore system proxy env vars
        
        # Add retry strategy for flaky connections
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,  # waits 2s, 4s, 8s between retries
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
    def get_crypto_news_cryptopanic_massive(self, target_events=5000):
        """
        Collecte MASSIVE avec pagination CryptoPanic
        Target: 5000+ événements
        """
        print(f"\n[MEGA COLLECTION] Target: {target_events} events from CryptoPanic")
        print("=" * 70)
        
        all_news = []
        page = 1
        next_url = None
        
        while len(all_news) < target_events:
            print(f"\n📄 Page {page} | Collected so far: {len(all_news)}")
            
            try:
                # Première requête ou suivre next URL
                if next_url:
                    response = self.session.get(next_url, timeout=45)
                else:
                    params = {
                        'auth_token': self.cryptopanic_key,
                        'public': 'true',
                        'currencies': 'BTC',
                        'kind': 'all',  # news + media + all
                        'filter': 'all'  # Récupérer TOUT
                    }
                    response = self.session.get(self.cryptopanic_url, params=params, timeout=45)
                
                response.raise_for_status()
                data = response.json()
                
                # Extraire les résultats
                results = data.get('results', [])
                if not results:
                    print("  ⚠ No more results, stopping...")
                    break
                
                # Parser chaque item
                for item in results:
                    news_item = {
                        'timestamp': item['published_at'],
                        'date': item['published_at'][:10],
                        'summary': item['title'],
                        'source': item['source']['title'],
                        'url': item['url'],
                        'votes_positive': item.get('votes', {}).get('positive', 0),
                        'votes_negative': item.get('votes', {}).get('negative', 0),
                        'votes_important': item.get('votes', {}).get('important', 0),
                        'kind': item.get('kind', 'news')
                    }
                    all_news.append(news_item)
                
                print(f"  ✓ Added {len(results)} items | Total: {len(all_news)}")
                
                # URL de la page suivante
                next_url = data.get('next')
                if not next_url:
                    print("  ⚠ No more pages available")
                    break
                
                page += 1
                
                # Respecter rate limits (2 req/sec pour DEVELOPER)
                time.sleep(0.6)
                
            except requests.exceptions.HTTPError as e:
                print(f"  ✗ HTTP Error: {e}")
                if e.response.status_code == 429:
                    print("  ⚠ Rate limit hit! Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                elif e.response.status_code == 403:
                    print("  ⚠ Monthly quota may be exceeded")
                    break
                else:
                    print(f"  Response: {e.response.text[:200]}")
                    break
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                break
        
        print(f"\n✅ CryptoPanic Collection Complete: {len(all_news)} events")
        return all_news
    
    def get_historical_news_batches(self, months_back=12):
        """
        Alternative: Collecter par batches de filtres différents
        Pour maximiser les données
        """
        print(f"\n[BATCH COLLECTION] Collecting {months_back} months with multiple filters")
        
        all_news = []
        filters = ['rising', 'hot', 'bullish', 'bearish', 'important']
        
        for filter_type in filters:
            print(f"\n🔍 Filter: {filter_type}")
            
            params = {
                'auth_token': self.cryptopanic_key,
                'public': 'true',
                'currencies': 'BTC',
                'filter': filter_type,
                'kind': 'all'
            }
            
            try:
                response = self.session.get(self.cryptopanic_url, params=params, timeout=45)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get('results', []):
                    news_item = {
                        'timestamp': item['published_at'],
                        'date': item['published_at'][:10],
                        'summary': item['title'],
                        'source': item['source']['title'],
                        'url': item['url'],
                        'votes_positive': item.get('votes', {}).get('positive', 0),
                        'votes_negative': item.get('votes', {}).get('negative', 0),
                        'filter': filter_type
                    }
                    all_news.append(news_item)
                
                print(f"  ✓ {filter_type}: {len(data.get('results', []))} items")
                time.sleep(0.6)
                
            except Exception as e:
                print(f"  ✗ Error for {filter_type}: {e}")
        
        # Dédupliquer par URL
        df = pd.DataFrame(all_news)
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        print(f"\n✅ Batch Collection: {len(df)} unique events")
        return df.to_dict('records')
    
    def get_bitcoin_price_history_bulk(self, start_date, end_date):
        """
        Récupérer TOUS les prix Bitcoin sur une période
        Plus efficace que requête par date
        """
        print(f"\n[PRICE DATA] Fetching Bitcoin prices from {start_date} to {end_date}")
        
        try:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            url = f"{self.coingecko_url}/coins/bitcoin/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convertir en DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp_ms', 'price'])
            prices_df['date'] = pd.to_datetime(prices_df['timestamp_ms'], unit='ms').dt.strftime('%Y-%m-%d')
            
            # Calculer change % par jour
            prices_df['price_change_24h'] = prices_df['price'].pct_change() * 100
            
            print(f"  ✓ Fetched {len(prices_df)} price points")
            
            return prices_df
            
        except Exception as e:
            print(f"  ✗ Price fetch error: {e}")
            return pd.DataFrame()
    
    def enrich_with_prices(self, news_list, prices_df):
        """
        Ajouter les prix à chaque événement
        """
        print("\n[ENRICHMENT] Adding Bitcoin prices to events...")
        
        enriched = []
        
        for news in news_list:
            date_str = news['date']
            
            # Trouver le prix pour cette date
            price_row = prices_df[prices_df['date'] == date_str]
            
            if not price_row.empty:
                news['bitcoin_price_usd'] = round(price_row.iloc[0]['price'], 2)
                news['price_change_24h'] = round(price_row.iloc[0]['price_change_24h'], 2)
            else:
                news['bitcoin_price_usd'] = None
                news['price_change_24h'] = 0
            
            enriched.append(news)
        
        print(f"  ✓ Enriched {len(enriched)} events")
        return enriched
    
    def classify_sentiment(self, row):
        """
        Sentiment basé sur votes CryptoPanic
        """
        positive = row.get('votes_positive', 0)
        negative = row.get('votes_negative', 0)
        
        net_vote = positive - negative
        
        # Normaliser entre -1 et 1
        if positive + negative == 0:
            return 0.0
        
        sentiment = net_vote / (positive + negative)
        return round(sentiment, 2)
    
    def categorize_news(self, text):
        """
        Catégoriser les news
        """
        text_lower = text.lower()
        
        categories = {
            'REGULATORY': ['regulation', 'sec', 'law', 'legal', 'government', 'ban', 'approve', 'etf'],
            'MACRO': ['inflation', 'fed', 'interest', 'economy', 'recession', 'market', 'dollar'],
            'TECH': ['upgrade', 'network', 'protocol', 'lightning', 'taproot', 'mining', 'halving'],
            'ADOPTION': ['adoption', 'accept', 'payment', 'tesla', 'company', 'institutional', 'microstrategy'],
            'SECURITY': ['hack', 'security', 'breach', 'exploit', 'vulnerability', 'attack', 'scam'],
            'MARKET': ['price', 'trading', 'volume', 'whale', 'exchange', 'pump', 'dump', 'rally', 'surge']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'MARKET'
    
    def compile_mega_dataset(self, target_events=5000):
        """
        Fonction principale pour compiler 5000+ événements
        """
        print("=" * 70)
        print(f"🚀 MEGA CRYPTO DATASET COLLECTOR - Target: {target_events} events")
        print("=" * 70)
        
        # Étape 1: Collecter news massivement
        news_list = self.get_crypto_news_cryptopanic_massive(target_events=target_events)
        
        if not news_list:
            print("\n⚠ No news collected!")
            return None
        
        # Étape 2: Récupérer toutes les dates uniques
        df_temp = pd.DataFrame(news_list)
        unique_dates = df_temp['date'].unique()
        
        if len(unique_dates) > 0:
            min_date = min(unique_dates)
            max_date = max(unique_dates)
            
            # Étape 3: Récupérer TOUS les prix Bitcoin en UNE requête
            prices_df = self.get_bitcoin_price_history_bulk(min_date, max_date)
            
            # Étape 4: Enrichir avec prix
            news_list = self.enrich_with_prices(news_list, prices_df)
        
        # Étape 5: Créer DataFrame et ajouter features
        df = pd.DataFrame(news_list)
        
        print("\n[PROCESSING] Adding sentiment and categories...")
        
        # Sentiment
        df['sentiment_score'] = df.apply(self.classify_sentiment, axis=1)
        
        # Catégorie
        df['category'] = df['summary'].apply(self.categorize_news)
        
        # Direction basée sur prix réel
        df['direction'] = df['price_change_24h'].apply(
            lambda x: 'UP' if x > 2 else ('DOWN' if x < -2 else 'NEUTRAL')
        )
        
        # Severity
        df['severity'] = df['price_change_24h'].abs().apply(
            lambda x: min(10, max(1, int(x)))
        )
        
        # Event ID
        df['event_id'] = df.apply(
            lambda row: f"real_{row['date'].replace('-', '')}_{hash(row['url']) % 100000:05d}",
            axis=1
        )
        
        # Réorganiser colonnes
        columns_order = [
            'event_id', 'timestamp', 'date', 'summary', 'source', 
            'category', 'sentiment_score', 'price_change_24h', 'direction', 
            'severity', 'bitcoin_price_usd', 'url'
        ]
        
        df = df[columns_order]
        
        # Stats finales
        print("\n" + "=" * 70)
        print("✅ DATASET COMPILED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nTotal Events: {len(df)}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"\n--- Direction Distribution ---")
        print(df['direction'].value_counts())
        print(f"\n--- Category Distribution ---")
        print(df['category'].value_counts())
        
        return df


def main():
    """
    Script principal pour collecter 5000+ événements
    """
    collector = MegaCryptoDataCollector()
    
    # COLLECTE MASSIVE
    print("\n⚠️ IMPORTANT: Cela peut prendre 30-60 minutes selon le quota API")
    print("   CryptoPanic DEVELOPER: 2 req/sec, 1000 req/month\n")
    
    # Target: 5000 événements minimum
    df = collector.compile_mega_dataset(target_events=5000)
    
    if df is not None:
        # Créer dossiers
        os.makedirs('data/raw', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarder
        output_main = f'data/raw/news_2018_2026.csv'
        output_backup = f'data/raw/mega_crypto_news_{timestamp}.csv'
        output_json = f'data/raw/mega_crypto_news_{timestamp}.json'
        
        df.to_csv(output_main, index=False)
        df.to_csv(output_backup, index=False)
        df.to_json(output_json, orient='records', indent=2)
        
        print("\n" + "=" * 70)
        print("💾 FILES SAVED:")
        print("=" * 70)
        print(f"1. {os.path.abspath(output_main)} (Kaggle-ready)")
        print(f"2. {os.path.abspath(output_backup)} (Backup)")
        print(f"3. {os.path.abspath(output_json)} (JSON)")
        print("=" * 70)
        
        # Sample
        print("\n--- First 5 Events ---")
        print(df.head())


if __name__ == "__main__":
    main()