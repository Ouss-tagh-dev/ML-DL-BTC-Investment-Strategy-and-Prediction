"""
Models router - Endpoints for model predictions and information
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, List, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services import model_service, data_service
from config import settings

router = APIRouter()

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_id: str
    features: Optional[Dict[str, float]] = None
    use_latest: bool = False

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_id: str
    prediction: int
    probability: float
    confidence: float
    direction: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features: Optional[Dict[str, float]] = None
    use_latest: bool = False

class ModelInfo(BaseModel):
    """Model information"""
    model_config = ConfigDict(extra='allow')
    id: str
    name: Optional[str] = None
    loaded: bool
    status: Optional[str] = None

@router.get("/list", response_model=List[ModelInfo])
async def list_models():
    """
    Get list of all available models with their status
    
    Returns information about all 9 models (5 ML + 4 DL)
    """
    try:
        models_info = model_service.get_all_models_info()
        # Add status field for each model
        for info in models_info:
            if 'status' not in info:
                info['status'] = 'ready' if info.get('loaded') else 'unavailable'
        return models_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction with a specific model
    
    - **model_id**: ID of the model to use
    - **features**: Feature values for prediction (dict)
    - **use_latest**: Use latest data from dataset if True
    """
    try:
        features = request.features
        
        # If use_latest, get latest data
        history = None
        if request.use_latest:
            # Fetch last 24 rows to support sequence models
            hist_data = data_service.get_latest_data(n=24)
            if hist_data and hist_data.get('data'):
                history = hist_data['data']
                features = history[-1]
                # Remove timestamp if present
                features.pop('timestamp', None)
        
        if not features:
            raise HTTPException(
                status_code=400,
                detail="Features required. Provide 'features' dict or set 'use_latest' to true"
            )
        
        result = model_service.predict(request.model_id, features)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found or prediction failed"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/info")
async def get_model_info(model_id: str):
    """
    Get detailed information about a specific model
    
    - **model_id**: ID of the model
    
    Returns metadata including architecture, hyperparameters, and performance metrics
    """
    try:
        info = model_service.get_model_info(model_id)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict")
async def batch_predict(data: Dict[str, Any] = Body(...)):
    """
    Make predictions with all available models
    
    - **features**: Feature values for prediction (dict)
    - **use_latest**: Use latest data from dataset if True
    
    Returns predictions from all models for comparison
    """
    try:
        features = data.get("features")
        use_latest = data.get("use_latest", False)
        
        # If use_latest, get latest data
        if use_latest:
            latest = data_service.get_latest_data(n=1)
            if latest and latest.get('data'):
                features = latest['data'][0].copy()
                features.pop('timestamp', None)
        
        if not features:
            raise HTTPException(
                status_code=400,
                detail="Features required. Provide 'features' dict or set 'use_latest' to true"
            )
        
        # Ensure models are loaded (fallback if startup failed)
        if not model_service.models_loaded:
             model_service.load_all_models()

        # Get history if use_latest
        history = None
        if use_latest:
            # Fetch last 24 rows to support sequence models
            hist_data = data_service.get_latest_data(n=24)
            if hist_data and hist_data.get('data'):
                history = hist_data['data']

        predictions = model_service.predict_all(features, history=history)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "features_used": list(features.keys())[:10]  # Show first 10 features
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load/{model_id}")
async def load_model(model_id: str):
    """
    Manually load a specific model into memory
    
    - **model_id**: ID of the model to load
    """
    try:
        success = model_service.load_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to load model {model_id}"
            )
        
        return {
            "model_id": model_id,
            "status": "loaded",
            "message": f"Model {model_id} loaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
