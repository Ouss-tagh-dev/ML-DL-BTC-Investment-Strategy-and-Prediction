# api/fetch_macro.py - Fetching macro indicators

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MACRO_OUTPUT_FILE, FRED_API_KEY

class MacroDataFetcher:
    """Fetches macro indicators from Yahoo Finance (FREE)."""
    
    def __init__(self):
        self.start_date = '2018-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
    def fetch_yahoo_finance_indices(self):
        """
        Fetches major market indices via Yahoo Finance.
        FREE - NO API KEY REQUIRED
        """
        print("📈 Fetching Yahoo Finance indices...")
        
        tickers = {
            'sp500': '^GSPC',      # S&P 500
            'nasdaq': '^IXIC',      # NASDAQ
            'dxy': 'DX-Y.NYB',      # Dollar Index
            'gold': 'GC=F',         # Gold Futures
            'oil': 'CL=F'           # Crude Oil
        }
        
        data = {}
        
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if not df.empty:
                    # Robust selection of 'Close' to handle MultiIndex
                    if isinstance(df.columns, pd.MultiIndex):
                        if 'Close' in df.columns.get_level_values(0):
                            s = df['Close']
                            # If it's still a DataFrame (multiple symbols), take the first column
                            if isinstance(s, pd.DataFrame):
                                s = s.iloc[:, 0]
                            data[name] = s
                    else:
                        if 'Close' in df.columns:
                            data[name] = df['Close']
                        else:
                            # Fallback if Close is missing but data exists
                            data[name] = df.iloc[:, 0]
                    
                    if name in data:
                        print(f"  ✅ {name}: {len(data[name])} points")
                else:
                    print(f"  ⚠️ {name}: No data")
            except Exception as e:
                print(f"  ❌ Error for {name}: {e}")
        
        # Only create DataFrame if we have valid data
        if not data:
            print(f"  ⚠️ No data fetched for indices")
            return pd.DataFrame()
        
        # Concat handles alignment across different dates automatically
        result_df = pd.concat(data, axis=1)
        return result_df
    
    def fetch_forex_rates(self):
        """Fetches Forex rates via Yahoo Finance."""
        print("💱 Fetching Forex rates...")
        
        pairs = {
            'eur_usd': 'EURUSD=X',
            'gbp_usd': 'GBPUSD=X',
            'jpy_usd': 'JPY=X'
        }
        
        data = {}
        
        for name, ticker in pairs.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if not df.empty:
                    # Robust selection to handle MultiIndex
                    if isinstance(df.columns, pd.MultiIndex):
                        s = df['Close']
                        if isinstance(s, pd.DataFrame):
                            s = s.iloc[:, 0]
                        data[name] = s
                    else:
                        data[name] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                    
                    print(f"  ✅ {name}: {len(data[name])} points")
            except Exception as e:
                print(f"  ❌ Error for {name}: {e}")
        
        # Use pd.concat to properly handle Series with different indices
        if not data:
            return pd.DataFrame()
        return pd.concat(data, axis=1)
    
    def fetch_all_macro_data(self):
        """Combines all macro data sources."""
        # Yahoo Finance
        indices_df = self.fetch_yahoo_finance_indices()
        forex_df = self.fetch_forex_rates()
        
        # Merge
        macro_df = pd.concat([indices_df, forex_df], axis=1)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df.sort_index(inplace=True)
        
        # Forward fill for weekends/holidays
        macro_df.ffill(inplace=True)
        
        return macro_df
    
    def save_data(self, df):
        """Saves data to CSV."""
        os.makedirs(os.path.dirname(MACRO_OUTPUT_FILE), exist_ok=True)
        df.to_csv(MACRO_OUTPUT_FILE, index_label='date')
        print(f"\n✅ Macro data saved: {MACRO_OUTPUT_FILE}")
        print(f"📊 {len(df)} days of data")
        print(f"📋 Columns: {list(df.columns)}")


def main():
    """Main function."""
    print("=" * 60)
    print("  FETCHING MACROECONOMIC INDICATORS")
    print("=" * 60)
    
    fetcher = MacroDataFetcher()
    
    # Fetching
    macro_df = fetcher.fetch_all_macro_data()
    
    if not macro_df.empty:
        fetcher.save_data(macro_df)
        print(f"📅 Period: {macro_df.index.min()} → {macro_df.index.max()}")
    else:
        print("❌ No data fetched")


if __name__ == '__main__':
    main()
