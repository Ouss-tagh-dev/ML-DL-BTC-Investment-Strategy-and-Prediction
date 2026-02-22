# api/fetch_historical.py - Fetching OHLCV data from Binance

import ccxt
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EXCHANGE_ID, SYMBOL, TIMEFRAME, SINCE_DATE, DATA_RAW_DIR, MAX_RETRIES

OUTPUT_FILE = os.path.join(DATA_RAW_DIR, f'{EXCHANGE_ID}_{SYMBOL.replace("/", "")}_{TIMEFRAME}.csv')
LIMIT = 1000  # Candle limit per request (Binance limit)


def fetch_all_ohlcv(exchange, symbol, timeframe, since, limit):
    """
    Fetches all OHLCV data from a start date with pagination.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (e.g. 'BTC/USDT')
        timeframe: Timeframe (e.g. '1h')
        since: Start date in ISO8601 format
        limit: Max candles per request
        
    Returns:
        List of lists [timestamp, open, high, low, close, volume]
    """
    since_ms = exchange.parse8601(since)
    all_ohlcv = []
    
    print(f"🚀 Starting fetch for {symbol} ({timeframe}) since {since}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since_ms = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            
            print(f"  📊 {len(all_ohlcv)} candles fetched. Last date: {exchange.iso8601(ohlcv[-1][0])}")
            
            # Respecting rate limit
            time.sleep(exchange.rateLimit / 1000)
            
        except ccxt.NetworkError as e:
            print(f"❌ Network error: {e}. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ General error: {e}")
            break
    
    return all_ohlcv


def save_data(ohlcv_data, filename):
    """
    Converts OHLCV data to DataFrame and saves to CSV.
    
    Args:
        ohlcv_data: List of OHLCV lists
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Datetime', inplace=True)
    df.drop('Timestamp', axis=1, inplace=True)
    
    df.to_csv(filename)
    print(f"\n✅ Data saved in {filename}")
    print(f"📈 Total rows: {len(df)}")
    print(f"📅 Period: {df.index.min()} → {df.index.max()}")


def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("  FETCHING HISTORICAL DATA BTC/USDT")
        print("=" * 60)
        
        exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
        print(f"✅ Exchange {EXCHANGE_ID} initialized")
        
        ohlcv_data = fetch_all_ohlcv(exchange, SYMBOL, TIMEFRAME, SINCE_DATE, LIMIT)
        
        if ohlcv_data:
            save_data(ohlcv_data, OUTPUT_FILE)
        else:
            print("❌ No data fetched")
            
    except AttributeError:
        print(f"❌ Exchange {EXCHANGE_ID} not found in CCXT")
    except Exception as e:
        print(f"❌ Fatal error: {e}")


if __name__ == '__main__':
    main()
