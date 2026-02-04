# config.py - Centralized configuration for the data pipeline

from datetime import datetime
import os
from pathlib import Path

# === FILE PATHS ===
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_FEATURES_DIR = BASE_DIR / 'data' / 'features'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_FEATURES_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === OHLCV DATA (BINANCE) ===
EXCHANGE_ID = 'binance'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'  # Hourly data
SINCE_DATE = '2018-01-01T00:00:00Z'  # Since 2018
OHLCV_OUTPUT_FILE = DATA_RAW_DIR / f'{EXCHANGE_ID}_{SYMBOL.replace("/", "")}_{TIMEFRAME}.csv'

# === BLOCKCHAIN METRICS ===
BLOCKCHAIN_OUTPUT_FILE = DATA_RAW_DIR / 'blockchain_metrics_daily.csv'
BLOCKCHAIN_DAYS = (datetime.now() - datetime(2018, 1, 1)).days  # From 2018 to now

# === MACRO INDICATORS ===
MACRO_OUTPUT_FILE = DATA_RAW_DIR / 'macro_indicators.csv'
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)  # Optional

# === SENTIMENT ===
SENTIMENT_OUTPUT_FILE = DATA_RAW_DIR / 'sentiment_metrics.csv'

# === FEATURES ===
FEATURES_OUTPUT_FILE = DATA_FEATURES_DIR / 'btc_features_complete.csv'

# === TECHNICAL PARAMETERS ===
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds between requests

# === FASTAPI SETTINGS ===
class Settings:
    """FastAPI Settings"""
    BASE_DIR = BASE_DIR
    MODELS_DIR = MODELS_DIR
    DATA_RAW_DIR = DATA_RAW_DIR
    DATA_FEATURES_DIR = DATA_FEATURES_DIR
    DATA_PROCESSED_DIR = DATA_PROCESSED_DIR
    OHLCV_FILE = OHLCV_OUTPUT_FILE
    FEATURES_FILE = FEATURES_OUTPUT_FILE
    BLOCKCHAIN_FILE = BLOCKCHAIN_OUTPUT_FILE
    MACRO_FILE = MACRO_OUTPUT_FILE
    SENTIMENT_FILE = SENTIMENT_OUTPUT_FILE
    
    # Available Models
    AVAILABLE_MODELS = [
        'logistic_regression',
        'naive_bayes',
        'random_forest',
        'svm',
        'xgboost',
        'mlp',
        'lstm',
        'gru',
        'lstm_cnn'
    ]
    
    # API Parameters
    API_PREFIX = "/api"
    DEBUG = True

settings = Settings()
