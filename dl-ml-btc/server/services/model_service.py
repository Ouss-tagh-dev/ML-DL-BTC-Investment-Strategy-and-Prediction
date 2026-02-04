"""
Model service for loading and managing ML/DL models
"""
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing ML/DL models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.models_loaded = False
        
    def load_all_models(self) -> bool:
        """Load all available models"""
        try:
            for model_id in settings.AVAILABLE_MODELS:
                self.load_model(model_id)
            self.models_loaded = True
            logger.info(f"Loaded {len(self.models)} models successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model"""
        try:
            model_dir = settings.MODELS_DIR / model_id
            
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False
            
            # Load metadata
            metadata_files = list(model_dir.glob("*_metadata.json"))
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    self.metadata[model_id] = json.load(f)
                logger.info(f"Loaded metadata for {model_id}")
            
            # Load scaler
            scaler_files = list(model_dir.glob("*_scaler.pkl"))
            if scaler_files:
                self.scalers[model_id] = joblib.load(scaler_files[0])
                logger.info(f"Loaded scaler for {model_id}")
            
            # Load model
            # For ML models (.pkl)
            pkl_files = list(model_dir.glob("*_model.pkl"))
            if pkl_files:
                self.models[model_id] = joblib.load(pkl_files[0])
                logger.info(f"Loaded ML model: {model_id}")
                return True
            
            # For XGBoost (.json)
            json_files = list(model_dir.glob("*_model.json"))
            if json_files:
                import xgboost as xgb
                self.models[model_id] = xgb.Booster()
                self.models[model_id].load_model(str(json_files[0]))
                logger.info(f"Loaded XGBoost model: {model_id}")
                return True
            
            # For DL models (.h5)
            h5_files = list(model_dir.glob("*_model.h5"))
            if h5_files:
                from tensorflow import keras
                self.models[model_id] = keras.models.load_model(str(h5_files[0]))
                logger.info(f"Loaded DL model: {model_id}")
                return True
            
            logger.warning(f"No model file found for {model_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information with normalized metrics"""
        if model_id not in self.metadata:
            return None
        
        metadata = self.metadata[model_id].copy()
        
        # Normalize nested metrics to top level for easier frontend consumption
        # 1. From performance_metrics block
        if 'performance_metrics' in metadata:
            perf = metadata['performance_metrics']
            if 'test_accuracy' in perf: metadata['accuracy'] = perf['test_accuracy']
            if 'test_auc' in perf: metadata['auc'] = perf['test_auc']
            if 'test_auc_roc' in perf: metadata['auc'] = perf['test_auc_roc']
            if 'test_loss' in perf: metadata['loss'] = perf['test_loss']
            
        # 2. From backtesting block
        if 'backtesting' in metadata:
            bt = metadata['backtesting']
            for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'maximum_drawdown', 
                        'annual_volatility', 'win_rate', 'total_trades']:
                if key in bt:
                    # Map maximum_drawdown to max_drawdown
                    target_key = 'max_drawdown' if key == 'maximum_drawdown' else key
                    metadata[target_key] = bt[key]
        
        # 3. Direct mappings for inconsistent formats
        if 'total_return_backtest' in metadata:
            metadata['total_return'] = metadata['total_return_backtest']
            
        metadata['id'] = model_id
        metadata['loaded'] = model_id in self.models
        metadata['has_scaler'] = model_id in self.scalers
        metadata['status'] = 'ready' if metadata['loaded'] else 'unavailable'
        
        # Ensure common keys exist
        if 'name' not in metadata and 'model_name' in metadata:
            metadata['name'] = metadata['model_name']
        if 'type' not in metadata and 'model_type' in metadata:
            metadata['type'] = metadata['model_type']
            
        return metadata
    
    def get_all_models_info(self) -> List[Dict[str, Any]]:
        """Get information for all models"""
        models_info = []
        
        for model_id in settings.AVAILABLE_MODELS:
            # Try to load if not already loaded
            if model_id not in self.models:
                self.load_model(model_id)
            
            info = self.get_model_info(model_id)
            if info:
                models_info.append(info)
            else:
                # Create basic info if metadata not available
                models_info.append({
                    'id': model_id,
                    'name': model_id.replace('_', ' ').title(),
                    'loaded': model_id in self.models,
                    'status': 'ready' if model_id in self.models else 'not_found'
                })
        
        return models_info
    
    def predict(
        self,
        model_id: str,
        features: Dict[str, float],
        use_scaler: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Make prediction with a specific model"""
        try:
            # Load model if not already loaded
            if model_id not in self.models:
                if not self.load_model(model_id):
                    return None
            
            model = self.models[model_id]
            
            # Convert features dict to DataFrame
            df = pd.DataFrame([features])
            
            # Apply scaler if available
            if use_scaler and model_id in self.scalers:
                scaler = self.scalers[model_id]
                X = scaler.transform(df)
            else:
                X = df.values
            
            # Make prediction
            if model_id == 'xgboost':
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                prob = model.predict(dmatrix)[0]
                prediction = int(prob > 0.5)
            elif model_id in ['lstm', 'gru', 'lstm_cnn', 'mlp']:
                # DL models need sequence reshaping for RNNs
                if model_id in ['lstm', 'gru', 'lstm_cnn']:
                    # Reshape for sequence models (assuming sequence_length from metadata)
                    sequence_length = self.metadata.get(model_id, {}).get('sequence_length', 24)
                    # For single prediction, repeat the features
                    X = np.repeat(X, sequence_length, axis=0).reshape(1, sequence_length, -1)
                
                prob = float(model.predict(X, verbose=0)[0][0])
                prediction = int(prob > 0.5)
            else:
                # Standard sklearn models
                prediction = int(model.predict(X)[0])
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(X)[0][1])
                else:
                    prob = float(prediction)
            
            return {
                'model_id': model_id,
                'prediction': prediction,
                'probability': prob,
                'confidence': abs(prob - 0.5) * 2,  # 0 to 1 scale
                'direction': 'UP' if prediction == 1 else 'DOWN'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_id}: {e}")
            return None
    
    def predict_all(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Make predictions with all models"""
        predictions = []
        
        for model_id in settings.AVAILABLE_MODELS:
            result = self.predict(model_id, features)
            if result:
                predictions.append(result)
        
        return predictions

# Global instance
model_service = ModelService()
