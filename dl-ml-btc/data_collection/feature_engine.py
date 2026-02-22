# api/feature_engine.py - Optimized feature engineering engine

import pandas as pd
import os
import sys
import numpy as np
import warnings

# Try to import talib, but make it optional
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib not available - technical indicators will be limited")

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_DIR, FEATURES_OUTPUT_FILE


class FeatureEngine:
    """
    Feature creation engine merging all data sources.
    Optimized for 95%+ OHLCV data retention.
    """
    
    def __init__(self):
        self.data_paths = {
            'ohlcv': os.path.join(DATA_RAW_DIR, 'binance_BTCUSDT_1h.csv'),
            'blockchain': os.path.join(DATA_RAW_DIR, 'blockchain_metrics_daily.csv'),
            'macro': os.path.join(DATA_RAW_DIR, 'macro_indicators.csv'),
            'sentiment': os.path.join(DATA_RAW_DIR, 'sentiment_metrics.csv')
        }
        self.critical_features = ['Close', 'Volume', 'returns']
    
    def load_data(self):
        """Loads all available data sources."""
        print("=" * 70)
        print("  LOADING DATA")
        print("=" * 70)
        
        data = {}
        
        # OHLCV (mandatory)
        if not os.path.exists(self.data_paths['ohlcv']):
            print(f"❌ ERROR: OHLCV data not found in {self.data_paths['ohlcv']}")
            print("   Run first: python api/fetch_historical.py")
            return None
        
        data['ohlcv'] = pd.read_csv(self.data_paths['ohlcv'], index_col=0, parse_dates=True)
        print(f"✅ OHLCV loaded: {data['ohlcv'].shape}")
        print(f"   📅 {data['ohlcv'].index.min()} → {data['ohlcv'].index.max()}")
        
        # Optional sources
        for source in ['blockchain', 'macro', 'sentiment']:
            path = self.data_paths[source]
            if os.path.exists(path):
                try:
                    data[source] = pd.read_csv(path, index_col=0, parse_dates=True)
                    print(f"✅ {source.capitalize()} loaded: {data[source].shape}")
                    print(f"   📅 {data[source].index.min()} → {data[source].index.max()}")
                except Exception as e:
                    print(f"⚠️  Error loading {source}: {e}")
            else:
                print(f"⚠️  {source.capitalize()} not found - will be ignored")
        
        return data
    
    def add_price_features(self, df):
        """Generates technical indicators from OHLCV data."""
        print("\n[1/5] 📈 Generating price features...")
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['future_return_1h'] = df['Close'].shift(-1) / df['Close'] - 1
        df['future_return_6h'] = df['Close'].shift(-6) / df['Close'] - 1
        df['future_return_24h'] = df['Close'].shift(-24) / df['Close'] - 1
        
        # Momentum
        df['momentum_24h'] = df['Close'] / df['Close'].shift(24) - 1
        df['momentum_7d'] = df['Close'] / df['Close'].shift(168) - 1
        
        # Volatility
        df['volatility_24h'] = df['returns'].rolling(24).std()
        df['volatility_7d'] = df['returns'].rolling(168).std()
        
        # Volume
        df['volume_ma_24h'] = df['Volume'].rolling(24).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_24h']
        
        # Technical indicators (TA-Lib or manual)
        if TALIB_AVAILABLE:
            df['SMA_20'] = ta.SMA(df['Close'], 20)
            df['SMA_50'] = ta.SMA(df['Close'], 50)
            df['SMA_200'] = ta.SMA(df['Close'], 200)
            df['EMA_12'] = ta.EMA(df['Close'], 12)
            df['EMA_26'] = ta.EMA(df['Close'], 26)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Close'])
            df['RSI_14'] = ta.RSI(df['Close'], 14)
            df['RSI_21'] = ta.RSI(df['Close'], 21)
            df['STOCH_K'], df['STOCH_D'] = ta.STOCH(df['High'], df['Low'], df['Close'])
            df['ATR_14'] = ta.ATR(df['High'], df['Low'], df['Close'], 14)
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['Close'], 20)
            df['ADX_14'] = ta.ADX(df['High'], df['Low'], df['Close'], 14)
            df['CCI_20'] = ta.CCI(df['High'], df['Low'], df['Close'], 20)
            df['WILLR_14'] = ta.WILLR(df['High'], df['Low'], df['Close'], 14)
            df['MFI_14'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], 14)
            df['OBV'] = ta.OBV(df['Close'], df['Volume'])
        else:
            # Manual implementations using pandas
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            # Simple RSI implementation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            df['RSI_21'] = df['Close'].diff().rolling(21).apply(lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))), raw=False)
            # Bollinger Bands
            df['BB_middle'] = df['SMA_20']
            bb_std = df['Close'].rolling(20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR_14'] = true_range.rolling(14).mean()
        
        # Ratios MA
        df['close_to_sma20'] = df['Close'] / df['SMA_20'] - 1
        df['close_to_sma50'] = df['Close'] / df['SMA_50'] - 1
        df['close_to_sma200'] = df['Close'] / df['SMA_200'] - 1
        
        print(f"   ✅ {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features created")
        return df
    
    def merge_external(self, df, source_name, source_df):
        """Intelligently merges external data (daily) with OHLCV (hourly)."""
        if source_df is None or source_df.empty:
            return df
        
        print(f"\n[{source_name.upper()}] 🔗 Merging data...")
        
        # Hourly resample + forward fill
        source_hourly = source_df.resample('1h').ffill()
        
        # Left join to keep all OHLCV rows
        merged = df.join(source_hourly, how='left')
        
        # Intelligent NaN filling
        new_cols = [c for c in source_hourly.columns]
        merged[new_cols] = merged[new_cols].ffill().bfill()
        
        # Final filling with median if necessary
        for col in new_cols:
            if merged[col].isnull().any():
                merged[col].fillna(merged[col].median(), inplace=True)
        
        print(f"   ✅ {len(new_cols)} columns merged")
        return merged
    
    def add_derived_features(self, df):
        """Creates derived features from merged data."""
        print("\n[Derived] 🧮 Creating derived features...")
        
        # Blockchain
        if 'hash_rate_th_s' in df.columns:
            df['hash_rate_pct_7d'] = df['hash_rate_th_s'].pct_change(168)
        
        if 'difficulty' in df.columns:
            df['difficulty_pct_7d'] = df['difficulty'].pct_change(168)
        
        # NVT Ratio
        if all(c in df.columns for c in ['market_price_usd', 'tx_count_daily', 'total_btc_supply']):
            mcap = df['market_price_usd'] * df['total_btc_supply']
            df['nvt_ratio'] = mcap / (df['tx_count_daily'] * 1e6)
            df['nvt_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['nvt_ratio'].ffill(inplace=True)
        
        # Sentiment
        if 'fear_greed_value' in df.columns:
            df['fg_change_7d'] = df['fear_greed_value'] - df['fear_greed_value'].shift(168)
            df['extreme_fear'] = (df['fear_greed_value'] < 25).astype(int)
            df['extreme_greed'] = (df['fear_greed_value'] > 75).astype(int)
        
        # Macro
        if 'sp500' in df.columns:
            df['sp500_returns'] = df['sp500'].pct_change()
            df['btc_sp500_corr'] = df['returns'].rolling(720).corr(df['sp500_returns'])
        
        if 'dxy' in df.columns:
            df['dxy_returns'] = df['dxy'].pct_change()
        
        print(f"   ✅ Derived features added")
        return df
    
    def create_targets(self, df):
        """Creates labels for ML models."""
        print("\n[Targets] 🎯 Creating labels...")
        
        # Binary classification
        df['target_direction'] = (df['future_return_1h'] > 0).astype(int)
        
        # Multi-class classification
        df['target_class'] = pd.cut(
            df['future_return_1h'],
            bins=[-np.inf, -0.01, -0.002, 0.002, 0.01, np.inf],
            labels=[0, 1, 2, 3, 4]
        )
        
        # Regression
        df['target_regression'] = df['future_return_1h']
        
        print("   ✅ Labels created (binary, multi-class, regression)")
        return df
    
    def clean_data(self, df):
        """Final cleanup optimized to keep 95%+ of data."""
        print("\n[Cleaning] 🧹 Final cleanup...")
        
        initial = len(df)
        
        # Replace inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Keep only rows with critical features
        df.dropna(subset=self.critical_features, inplace=True)
        
        # Fill technical indicators
        tech_cols = [c for c in df.columns if any(x in c.lower() for x in 
                     ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'stoch', 'adx', 'cci', 'mfi', 'obv'])]
        if tech_cols:
            df[tech_cols] = df[tech_cols].ffill().bfill()
        
        # Remove rows without targets
        target_cols = [c for c in df.columns if 'target' in c.lower()]
        if target_cols:
            df.dropna(subset=target_cols, inplace=True)
        
        # Final cleanup
        df.dropna(inplace=True)
        
        final = len(df)
        retention = (final / initial) * 100
        
        print(f"   📊 Initial: {initial:,} → Final: {final:,}")
        print(f"   ✅ Retention: {retention:.1f}%")
        
        return df
    
    def run(self):
        """Main pipeline."""
        print("\n" + "=" * 70)
        print("  FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        
        # 1. Load
        data = self.load_data()
        if data is None:
            return None
        
        # 2. Start from OHLCV
        df = data['ohlcv'].copy()
        initial_rows = len(df)
        
        # 3. Price features
        df = self.add_price_features(df)
        
        # 4. Merge external sources
        for source in ['blockchain', 'macro', 'sentiment']:
            if source in data:
                df = self.merge_external(df, source, data[source])
        
        # 5. Derived features
        df = self.add_derived_features(df)
        
        # 6. Targets
        df = self.create_targets(df)
        
        # 7. Cleanup
        df = self.clean_data(df)
        
        # Summary
        print("\n" + "=" * 70)
        print("  ✅ FEATURE ENGINEERING COMPLETED")
        print("=" * 70)
        print(f"  Initial OHLCV rows: {initial_rows:,}")
        print(f"  Final rows: {len(df):,}")
        print(f"  Total Features: {df.shape[1]}")
        print(f"  Global Retention: {len(df)/initial_rows*100:.1f}%")
        
        return df
    
    def save(self, df):
        """Saves the final dataset."""
        if df is None:
            return
        
        os.makedirs(os.path.dirname(FEATURES_OUTPUT_FILE), exist_ok=True)
        df.to_csv(FEATURES_OUTPUT_FILE)
        
        print(f"\n💾 Saved: {FEATURES_OUTPUT_FILE}")
        print(f"📅 Period: {df.index.min()} → {df.index.max()}")
        print(f"⏱️  Duration: {(df.index.max() - df.index.min()).days} days")
        
        # Category summary
        tech = sum(1 for c in df.columns if any(x in c.lower() for x in ['sma', 'ema', 'rsi', 'macd']))
        blockchain = sum(1 for c in df.columns if any(x in c.lower() for x in ['hash', 'difficulty', 'tx']))
        sentiment = sum(1 for c in df.columns if 'fear' in c.lower() or 'greed' in c.lower())
        macro = sum(1 for c in df.columns if any(x in c.lower() for x in ['sp500', 'dxy', 'gold']))
        targets = sum(1 for c in df.columns if 'target' in c.lower())
        
        print(f"\n📊 Feature Categories:")
        print(f"   • Technical: {tech}")
        print(f"   • Blockchain: {blockchain}")
        print(f"   • Sentiment: {sentiment}")
        print(f"   • Macro: {macro}")
        print(f"   • Targets: {targets}")


def main():
    """Main execution."""
    engine = FeatureEngine()
    features = engine.run()
    
    if features is not None:
        engine.save(features)
        
        # Preview
        print("\n" + "=" * 70)
        print("  DATA PREVIEW (last 5 rows)")
        print("=" * 70)
        cols = ['Close', 'RSI_14', 'MACD', 'fear_greed_value', 'sp500', 'target_direction']
        available = [c for c in cols if c in features.columns]
        if available:
            print(features[available].tail())


if __name__ == '__main__':
    main()
