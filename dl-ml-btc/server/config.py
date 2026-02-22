"""
Configuration for FastAPI server
"""
from pathlib import Path
from pydantic_settings import BaseSettings

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FEATURES_FILE = DATA_DIR / "features" / "btc_features_complete.csv"
NEWS_DATA_FILE = DATA_DIR / "news" / "news_rss_processed.csv"

# Global list of models to ensure consistency
TOTAL_AVAILABLE_MODELS = [
    "logistic_regression",
    "naive_bayes",
    "random_forest",
    "svm",
    "xgboost",
    "mlp",
    "lstm",
    "gru",
    "lstm_cnn",
    "hybrid_cnn_bilstm"
]

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Bitcoin ML/DL Dashboard API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API for Bitcoin price prediction dashboard with 9 ML/DL models"
    
    # CORS Configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Data Configuration
    DATA_DIR: Path = DATA_DIR
    MODELS_DIR: Path = MODELS_DIR
    FEATURES_FILE: Path = FEATURES_FILE
    NEWS_DATA_FILE: Path = NEWS_DATA_FILE
    
    # Model Configuration
    AVAILABLE_MODELS: list = TOTAL_AVAILABLE_MODELS
    
    # Cache Configuration
    CACHE_MODELS: bool = True  # Keep models in memory
    CACHE_DATA: bool = True    # Keep data in memory
    
    class Config:
        case_sensitive = True

settings = Settings()
