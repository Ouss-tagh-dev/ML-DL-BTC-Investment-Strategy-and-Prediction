"""
Data service for loading and serving Bitcoin data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from config import settings

logger = logging.getLogger(__name__)

class DataService:
    """Service for managing Bitcoin data"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.data_loaded = False
        
    def load_data(self) -> bool:
        """Load Bitcoin features data from CSV"""
        try:
            logger.info(f"Loading data from {settings.FEATURES_FILE}")
            self.df = pd.read_csv(settings.FEATURES_FILE, index_col=0, parse_dates=True)
            self.data_loaded = True
            logger.info(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get historical OHLCV data with optional filtering"""
        if not self.data_loaded:
            self.load_data()
        
        df = self.df.copy()
        
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Select specific columns
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            df = df[available_cols]
        else:
            # Default: OHLCV + key indicators
            default_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'returns', 'RSI_14', 'MACD', 'MACD_signal']
            available_cols = [col for col in default_cols if col in df.columns]
            df = df[available_cols]
        
        # Apply limit
        if limit:
            df = df.tail(limit)
        
        # Convert to records format
        df_reset = df.reset_index()
        df_reset.columns = ['timestamp'] + list(df.columns)
        
        return {
            "data": df_reset.to_dict(orient='records'),
            "count": len(df),
            "columns": list(df.columns),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max())
        }
    
    def get_features(
        self,
        feature_names: Optional[List[str]] = None,
        limit: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """Get specific features"""
        if not self.data_loaded:
            self.load_data()
        
        df = self.df.copy()
        
        # Select features
        if feature_names:
            available_features = [f for f in feature_names if f in df.columns]
            df = df[available_features]
        
        # Apply limit
        if limit:
            df = df.tail(limit)
        
        df_reset = df.reset_index()
        df_reset.columns = ['timestamp'] + list(df.columns)
        
        return {
            "data": df_reset.to_dict(orient='records'),
            "count": len(df),
            "features": list(df.columns)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data_loaded:
            self.load_data()
        
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "date_range": {
                "start": str(self.df.index.min()),
                "end": str(self.df.index.max())
            },
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_stats": {}
        }
        
        # Get statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:20]:  # Limit to first 20 for performance
            stats["numeric_stats"][col] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "median": float(self.df[col].median())
            }
        
        return stats
    
    def get_latest_data(self, n: int = 1) -> Dict[str, Any]:
        """Get the latest n rows of data with additional metrics"""
        if not self.data_loaded:
            self.load_data()
        
        # Get latest n rows for the 'data' field
        latest_n = self.df.tail(n).copy()
        
        # Calculate global metrics from the full dataframe
        current_price = float(self.df['Close'].iloc[-1])
        
        # 24h change (assuming 1h data, so 24 rows ago)
        change_24h = 0.0
        if len(self.df) > 24:
            price_24h_ago = float(self.df['Close'].iloc[-25])
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
        # Prepare records
        df_reset = latest_n.reset_index()
        df_reset.columns = ['timestamp'] + list(latest_n.columns)
        records = df_reset.to_dict(orient='records')
        
        # Add price and change to each record for convenience, 
        # but also provide them at top level
        for record in records:
            record['price'] = current_price
            record['change24h'] = change_24h
            # Modernize timestamp format if it's a datetime object
            if isinstance(record['timestamp'], (pd.Timestamp, datetime)):
                record['timestamp'] = record['timestamp'].isoformat()
        
        return {
            "data": records,
            "count": len(latest_n),
            "price": current_price,
            "change24h": change_24h,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_feature_names(self) -> List[str]:
        """Get all available feature names"""
        if not self.data_loaded:
            self.load_data()
        
        return list(self.df.columns)

# Global instance
data_service = DataService()
