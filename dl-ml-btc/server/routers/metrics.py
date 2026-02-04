"""
Metrics router - Endpoints for model performance metrics
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services import model_service

router = APIRouter()

class PerformanceMetrics(BaseModel):
    """Performance metrics for a model"""
    model_id: str
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None

@router.get("/performance")
async def get_performance_metrics():
    """
    Get performance metrics for all models
    
    Returns accuracy, AUC, Sharpe Ratio, Total Return, etc.
    """
    try:
        models_info = model_service.get_all_models_info()
        
        metrics = []
        for model in models_info:
            metric = {
                "model_id": model.get('id'),
                "model_name": model.get('name', model.get('id')),
                "accuracy": model.get('accuracy'),
                "auc": model.get('auc'),
                "sharpe_ratio": model.get('sharpe_ratio'),
                "total_return": model.get('total_return'),
                "max_drawdown": model.get('max_drawdown'),
                "annual_volatility": model.get('annual_volatility')
            }
            metrics.append(metric)
        
        return {
            "metrics": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/backtesting")
async def get_backtesting_results(model_id: str):
    """
    Get backtesting results for a specific model
    
    - **model_id**: ID of the model
    
    Returns detailed backtesting metrics and equity curve
    """
    try:
        info = model_service.get_model_info(model_id)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        # Extract backtesting info from metadata
        backtesting = {
            "model_id": model_id,
            "model_name": info.get('name', model_id),
            "total_return": info.get('total_return'),
            "benchmark_return": info.get('benchmark_return'),
            "sharpe_ratio": info.get('sharpe_ratio'),
            "max_drawdown": info.get('max_drawdown'),
            "annual_volatility": info.get('annual_volatility'),
            "win_rate": info.get('win_rate'),
            "total_trades": info.get('total_trades')
        }
        
        return backtesting
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comparison")
async def get_comparison_metrics():
    """
    Get comprehensive comparison metrics for all models
    
    Returns data formatted for comparison tables and charts
    """
    try:
        models_info = model_service.get_all_models_info()
        
        comparison = {
            "models": [],
            "ml_models": [],
            "dl_models": []
        }
        
        for model in models_info:
            model_data = {
                "id": model.get('id'),
                "name": model.get('name', model.get('id')),
                "type": model.get('type', 'ML'),
                "accuracy": model.get('accuracy'),
                "auc": model.get('auc'),
                "sharpe_ratio": model.get('sharpe_ratio'),
                "total_return": model.get('total_return'),
                "max_drawdown": model.get('max_drawdown')
            }
            
            comparison["models"].append(model_data)
            
            # Categorize by type
            if model.get('type') == 'DL' or model.get('id') in ['mlp', 'lstm', 'gru', 'lstm_cnn']:
                comparison["dl_models"].append(model_data)
            else:
                comparison["ml_models"].append(model_data)
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
