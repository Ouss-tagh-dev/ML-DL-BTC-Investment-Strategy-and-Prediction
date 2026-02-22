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

from server.config import settings
from server.services.ml_utils import FocalLoss
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

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
        logger.info(f"--- STARTING LOAD_ALL_MODELS with {len(settings.AVAILABLE_MODELS)} models: {settings.AVAILABLE_MODELS} ---")
        try:
            for i, model_id in enumerate(settings.AVAILABLE_MODELS):
                logger.info(f"Loop index {i}: target model_id={model_id}")
                self.load_model(model_id)
            self.models_loaded = True
            logger.info(f"Loaded {len(self.models)} models successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model"""
        logger.info(f"--- Attempting to load model: {model_id} ---")
        try:
            model_dir = settings.MODELS_DIR / model_id
            
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False
            
            # Load metadata - Merge all found metadata files
            metadata_files = (
                list(model_dir.glob("*_metadata.json")) + 
                list(model_dir.glob("*_model_metadata.json")) +
                list(model_dir.glob("config.json"))
            )
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
            scaler_files = list(model_dir.glob("*_scaler.pkl")) + list(model_dir.glob("best_scaler_*.pkl"))
            if scaler_files:
                self.scalers[model_id] = joblib.load(scaler_files[0])
                logger.info(f"Loaded scaler for {model_id}")
            
            # Load PCA if present
            pca_files = list(model_dir.glob("*_pca.pkl"))
            if pca_files:
                self.scalers[f"{model_id}_pca"] = joblib.load(pca_files[0])
                logger.info(f"Loaded PCA for {model_id}")
            
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
            
            # For DL models (.h5 or .keras)
            h5_files = list(model_dir.glob("*_model.h5")) or list(model_dir.glob("*.h5"))
            keras_files = list(model_dir.glob("*_model.keras")) or list(model_dir.glob("model.keras")) or list(model_dir.glob("best_model.keras"))
            
            if h5_files or keras_files:
                from tensorflow import keras
                model_path = str(h5_files[0] if h5_files else keras_files[0])
                logger.info(f"Loading Keras model for {model_id} from {model_path}")
                self.models[model_id] = keras.models.load_model(
                    model_path,
                    custom_objects={'FocalLoss': FocalLoss}
                )
                
                if model_id == 'hybrid_cnn_bilstm':
                    try:
                        self.scalers[f"{model_id}_tfidf"] = joblib.load(model_dir / "tfidf.pkl")
                        self.scalers[f"{model_id}_svd"] = joblib.load(model_dir / "svd.pkl")
                        self.scalers[f"{model_id}_tfidf_scaler"] = joblib.load(model_dir / "tfidf_scaler.pkl")
                        with open(model_dir / "tokenizer.json", 'r') as f:
                            from tensorflow.keras.preprocessing.text import tokenizer_from_json
                            self.scalers[f"{model_id}_tokenizer"] = tokenizer_from_json(json.load(f))
                        
                        # Ensure hybrid model is correctly categorized in its own metadata
                        if model_id in self.metadata:
                            self.metadata[model_id]['model_type'] = 'DL'
                            
                        logger.info(f"Loaded additional hybrid artifacts for {model_id}")
                    except Exception as e:
                        logger.error(f"Error loading hybrid artifacts: {e}")
                        
                logger.info(f"Loaded {'Keras' if keras_files else 'DL'} model: {model_id}")
                return True
            
            logger.warning(f"No model file found for {model_id}")
            return False
            
        except Exception as e:
            logger.error(f"FATAL ERROR loading model {model_id}: {e}", exc_info=True)
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
        # More robust type detection
        m_type = str(metadata.get('model_type', 'ML')).upper()
        if model_id == 'hybrid_cnn_bilstm' or any(x in m_type for x in ['DEEP LEARNING', 'LSTM', 'GRU', 'CNN', 'MLP', 'HYBRID']):
            metadata['type'] = 'DL'
        else:
            metadata['type'] = 'ML'
            
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
        history: Optional[List[Dict[str, float]]] = None,
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
            
            # 2. Try to get from metadata (various possible paths)
            elif model_id in self.metadata:
                meta = self.metadata[model_id]
                if 'features' in meta:
                    expected_features = meta['features']
                    logger.info(f"Using 'features' from metadata for {model_id} ({len(expected_features)})")
                elif 'top_features' in meta:
                    expected_features = meta['top_features']
                    logger.info(f"Using 'top_features' from metadata for {model_id} ({len(expected_features)})")
                elif 'data_config' in meta and 'feature_names' in meta['data_config']:
                    expected_features = meta['data_config']['feature_names']
                    logger.info(f"Using 'data_config.feature_names' from metadata for {model_id} ({len(expected_features)})")
                elif 'data_info' in meta and 'feature_names' in meta['data_info']:
                    expected_features = meta['data_info']['feature_names']
                    logger.info(f"Using 'data_info.feature_names' from metadata for {model_id} ({len(expected_features)})")
            
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
                # Filter to numeric columns only to avoid scaling strings (e.g. for LSTM)
                df = df.select_dtypes(include=[np.number])
                logger.warning(f"No feature info found for {model_id}, using all {len(df.columns)} numeric columns")

            # Apply scaler logic
            X = None
            if model_id == 'hybrid_cnn_bilstm':
                # Skip standard scaling, hybrid uses its own complex pipeline
                X = df.values # Will be ignored anyway
            elif use_scaler and scaler is not None:
                # For GaussianNB or cases where scaler is actually a model
                if not hasattr(scaler, 'transform'):
                    logger.warning(f"Scaler for {model_id} has no 'transform' method. Skipping scaling.")
                    X = df.values
                # For sequence models, we need to transform EACH row in the sequence
                elif history and model_id in ['lstm', 'gru', 'lstm_cnn']:
                    df_history = pd.DataFrame(history)
                    if expected_features:
                        for f in expected_features:
                            if f not in df_history.columns:
                                df_history[f] = 0.0
                        df_history = df_history[expected_features]
                    else:
                        df_history = df_history.select_dtypes(include=[np.number])
                    
                    # Ensure feature count matches scaler
                    if hasattr(scaler, 'n_features_in_') and df_history.shape[1] != scaler.n_features_in_:
                        logger.warning(f"{model_id} scaler expects {scaler.n_features_in_} features, but got {df_history.shape[1]}. Padding with 0.")
                        while df_history.shape[1] < scaler.n_features_in_:
                            df_history[f'pad_{df_history.shape[1]}'] = 0.0
                        if df_history.shape[1] > scaler.n_features_in_:
                            df_history = df_history.iloc[:, :scaler.n_features_in_]
                    
                    logger.info(f"Transforming history for {model_id}: shape={df_history.shape}")
                    X = scaler.transform(df_history)
                else:
                    # Basic 2D scaling
                    # Ensure feature count matches scaler
                    if hasattr(scaler, 'n_features_in_') and df.shape[1] != scaler.n_features_in_:
                        logger.warning(f"{model_id} scaler expects {scaler.n_features_in_} features, but got {df.shape[1]}. Padding with 0.")
                        while df.shape[1] < scaler.n_features_in_:
                            df[f'pad_{df.shape[1]}'] = 0.0
                        if df.shape[1] > scaler.n_features_in_:
                            df = df.iloc[:, :scaler.n_features_in_]
                    
                    logger.info(f"Transforming single row for {model_id}: shape={df.shape}")
                    X = scaler.transform(df)
            else:
                X = df.values
                
            # Apply PCA if available
            pca = self.scalers.get(f"{model_id}_pca")
            if pca is not None:
                X = pca.transform(X)
                logger.info(f"Applied PCA for {model_id}. New shape: {X.shape}")
            
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
                    seq_len = self.metadata[model_id].get('sequence_length', 24)
                    
                    # Ensure we have enough data (pad if necessary)
                    if X.shape[0] < seq_len:
                        padding = np.zeros((seq_len - X.shape[0], X.shape[1]))
                        X = np.vstack([padding, X])
                    elif X.shape[0] > seq_len:
                        X = X[-seq_len:]
                        
                    X = X.reshape((1, seq_len, X.shape[1]))
                
                logger.info(f"Model {model_id} predict: input_shape={X.shape}, model_type={type(model)}")
                prob = model.predict(X)[0]
                # If binary classification (sigmoid), prob is single value
                if hasattr(prob, '__len__') and len(prob) > 1:
                    prediction = int(np.argmax(prob))
                    prob = float(np.max(prob))
                else:
                    prob = float(prob[0] if hasattr(prob, '__len__') else prob)
                    prediction = int(prob > 0.5)
            
            elif model_id == 'hybrid_cnn_bilstm':
                # Specialized logic for 3-branch hybrid model
                cfg = self.metadata[model_id]
                text = features.get('text', features.get('headline', ''))
                sentiment_score = features.get('sentiment_score', 0.0)
                category = features.get('category', 'OTHER')
                source = features.get('source', 'Other')
                price_change_24h = features.get('price_change_24h', 0.0)
                severity = features.get('severity', 1)
                
                # 1. Text Preprocessing
                clean_text = re.sub(r"http\S+|www\S+", "", text)
                clean_text = re.sub(r"<[^>]+>", " ", clean_text)
                clean_text = re.sub(r"[^a-zA-Z0-9\s'\-\$\%\.\,]", " ", clean_text)
                clean_text = re.sub(r"\s+", " ", clean_text).strip().lower()
                
                # Enriched text matching training logic
                def sentiment_label(score):
                    if score > 0.3: return "very_positive"
                    elif score > 0.1: return "positive"
                    elif score < -0.3: return "very_negative"
                    elif score < -0.1: return "negative"
                    else: return "neutral"
                
                def severity_label(s):
                    if s >= 8: return "critical_event"
                    elif s >= 5: return "major_event"
                    else: return "minor_event"
                
                enriched = f"{clean_text} {category.lower()} {sentiment_label(sentiment_score)} {severity_label(severity)}"
                
                # Tokenization
                tokenizer = self.scalers.get(f"{model_id}_tokenizer")
                X_text = pad_sequences(
                    tokenizer.texts_to_sequences([enriched]),
                    maxlen=cfg.get("MAX_LENGTH", 200), padding="post", truncating="post"
                )
                
                # 2. TF-IDF LSA Preprocessing
                tfidf = self.scalers.get(f"{model_id}_tfidf")
                svd = self.scalers.get(f"{model_id}_svd")
                tfidf_scaler = self.scalers.get(f"{model_id}_tfidf_scaler")
                
                tfidf_raw = tfidf.transform([clean_text])
                tfidf_lsa = tfidf_scaler.transform(svd.transform(tfidf_raw)).astype(np.float32)
                
                # 3. Numeric Preprocessing
                daily_defaults = {
                    "volume_news": 5.0, "avg_sentiment": 0.0, "std_sentiment": 0.1,
                    "avg_price_change": 0.0, "max_severity": 5.0, "avg_severity": 3.0,
                    "news_momentum": 5.0, "sent_momentum": 0.0, "sent_volatility": 0.0,
                    "volume_accel": 0.0, "log_price": np.log1p(95_000), "neg_news_ratio": 0.3,
                    "pos_news_ratio": 0.4, "sentiment_range": 0.2, "max_sentiment": 0.3,
                    "min_sentiment": -0.1, "sent_trend_5d": 0.0, "price_sent_x": 0.0,
                    "is_weekend": 0.0, "is_monday": 0.0, "is_friday": 0.0
                }
                
                cont_cols = cfg.get("CONTINUOUS_COLS", [])
                per_article = {
                    "sentiment_score": sentiment_score,
                    "price_change_24h": price_change_24h,
                    "severity": float(severity)
                }
                
                cont_row = np.array(
                    [[per_article.get(c, daily_defaults.get(c, 0.0)) for c in cont_cols]],
                    dtype=np.float32
                )
                scaler = self.scalers.get(model_id)
                # Robust padding for hybrid numeric scaler
                if hasattr(scaler, 'n_features_in_') and cont_row.shape[1] != scaler.n_features_in_:
                    logger.info(f"Padding hybrid numeric features from {cont_row.shape[1]} to {scaler.n_features_in_}")
                    if cont_row.shape[1] < scaler.n_features_in_:
                        pad = np.zeros((1, scaler.n_features_in_ - cont_row.shape[1]), dtype=np.float32)
                        cont_row = np.hstack([cont_row, pad])
                    else:
                        cont_row = cont_row[:, :scaler.n_features_in_]
                
                cont_scaled = scaler.transform(cont_row)
                
                # OHE for Category and Source
                cat_list = cfg.get("CATEGORIES", [])
                src_list = cfg.get("SOURCES", [])
                
                def get_ohe(val, val_list):
                    idx = val_list.index(val) if val in val_list else val_list.index("OTHER") if "OTHER" in val_list else len(val_list)-1
                    ohe = np.zeros(len(val_list))
                    ohe[idx] = 1
                    return ohe
                
                cat_ohe = get_ohe(category, cat_list)
                src_ohe = get_ohe(source, src_list)
                
                X_num = np.hstack([cont_scaled[0], cat_ohe, src_ohe]).astype(np.float32).reshape(1, -1)
                
                # 4. Hybrid Inference
                logger.info(f"Hybrid {model_id} shapes: text={X_text.shape}, tfidf={tfidf_lsa.shape}, num={X_num.shape}")
                logger.info(f"Hybrid {model_id} model_type={type(model)}")
                prob = float(model.predict([X_text, tfidf_lsa, X_num], verbose=0)[0][0])
                thresh = cfg.get("threshold", 0.5)
                prediction = int(prob >= thresh)
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
    
    def predict_all(self, features: Dict[str, float], history: Optional[List[Dict[str, float]]] = None) -> List[Dict[str, Any]]:
        """Make predictions with all models"""
        predictions = []
        
        for model_id in settings.AVAILABLE_MODELS:
            try:
                result = self.predict(model_id, features, history=history)
                if result:
                    predictions.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction for {model_id}: {e}")
                # Continue with other models
                continue
        
        return predictions

# Global instance
model_service = ModelService()
