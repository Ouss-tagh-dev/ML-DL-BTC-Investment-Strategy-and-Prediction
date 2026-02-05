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
            
            # Load metadata - Merge all found metadata files
            metadata_files = list(model_dir.glob("*_metadata.json")) + list(model_dir.glob("*_model_metadata.json"))
            if metadata_files:
                combined_metadata = {}
                for meta_file in metadata_files:
                    try:
                        with open(meta_file, 'r') as f:
                            data = json.load(f)
                            combined_metadata.update(data)
                    except Exception as e:
                        logger.error(f"Error reading metadata file {meta_file}: {e}")
                
                self.metadata[model_id] = combined_metadata
                logger.info(f"Loaded merged metadata for {model_id} from {len(metadata_files)} files")
            
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
            # Standardize accuracy keys
            if 'test_accuracy' in perf: metadata['accuracy'] = float(perf['test_accuracy'])
            elif 'accuracy' in perf: metadata['accuracy'] = float(perf['accuracy'])
            
            # Standardize AUC keys
            if 'test_auc' in perf: metadata['auc'] = float(perf['test_auc'])
            elif 'test_auc_roc' in perf: metadata['auc'] = float(perf['test_auc_roc'])
            elif 'auc_roc' in perf: metadata['auc'] = float(perf['auc_roc'])
            elif 'auc' in perf: metadata['auc'] = float(perf['auc'])
            
            # Standardize loss
            if 'test_loss' in perf: metadata['loss'] = float(perf['test_loss'])
            
        # 2. From backtesting block
        if 'backtesting' in metadata:
            bt = metadata['backtesting']
            for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'maximum_drawdown', 
                        'annual_volatility', 'win_rate', 'total_trades']:
                if key in bt:
                    # Map maximum_drawdown to max_drawdown
                    target_key = 'max_drawdown' if key == 'maximum_drawdown' else key
                    metadata[target_key] = float(bt[key])
        
        # 3. Direct mappings for top-level keys (common in ML models)
        if 'accuracy' in metadata: 
            metadata['accuracy'] = float(metadata['accuracy'])
            
        if 'total_return_backtest' in metadata:
            metadata['total_return'] = float(metadata['total_return_backtest'])
        
        # 4. Fallback for missing metrics to ensure consistency in comparison tables
        for key in ['accuracy', 'auc', 'sharpe_ratio', 'total_return', 'max_drawdown']:
            if key not in metadata:
                metadata[key] = 0.0 # Default to 0 instead of missing
            else:
                # Ensure float
                try:
                    metadata[key] = float(metadata[key])
                except:
                    metadata[key] = 0.0
            
        metadata['id'] = model_id
        metadata['name'] = metadata.get('model_name', model_id.replace('_', ' ').title())
        metadata['type'] = metadata.get('model_type', 'ML')
        metadata['loaded'] = model_id in self.models
        metadata['has_scaler'] = model_id in self.scalers
        metadata['status'] = 'ready' if metadata['loaded'] else 'unavailable'
            
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
            
            # FEATURE FILTERING LOGIC
            # Determine expected features from scaler or metadata
            expected_features = []
            scaler = self.scalers.get(model_id)
            
            # 1. Try to get from scaler (feature_names_in_)
            if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                expected_features = list(scaler.feature_names_in_)
                logger.info(f"Using features from scaler for {model_id} ({len(expected_features)})")
            
            # 2. Try to get from metadata
            elif model_id in self.metadata:
                meta = self.metadata[model_id]
                if 'features' in meta:
                    expected_features = meta['features']
                    logger.info(f"Using 'features' from metadata for {model_id} ({len(expected_features)})")
                elif 'top_features' in meta:
                    expected_features = meta['top_features']
                    logger.info(f"Using 'top_features' from metadata for {model_id} ({len(expected_features)})")
            
            # Apply filtering if we found expected features
            if expected_features:
                # Check for missing features and fill with 0
                missing = [f for f in expected_features if f not in df.columns]
                if missing:
                    logger.warning(f"Model {model_id} missing features: {missing}. Filling with 0.")
                    for f in missing:
                        df[f] = 0.0
                
                # Filter df to exactly the expected features in correct order
                df = df[expected_features]
            else:
                logger.warning(f"No feature info found for {model_id}, using all {len(df.columns)} columns")

            # Apply scaler if available
            if use_scaler and scaler is not None:
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
                'prediction': int(prediction),
                'probability': float(prob),
                'confidence': float(abs(prob - 0.5) * 2),  # 0 to 1 scale
                'direction': 'UP' if prediction == 1 else 'DOWN'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_id}: {e}")
            return None
    
    def predict_all(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Make predictions with all models"""
        predictions = []
        
        for model_id in settings.AVAILABLE_MODELS:
            try:
                result = self.predict(model_id, features)
                if result:
                    predictions.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction for {model_id}: {e}")
                # Continue with other models
                continue
        
        return predictions

# Global instance
model_service = ModelService()
