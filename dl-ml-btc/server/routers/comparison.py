"""
Comparison router - Endpoints for model comparisons and charts
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services import model_service

router = APIRouter()

@router.get("/accuracy")
async def compare_accuracy():
    """
    Compare accuracy across all models
    
    Returns data formatted for bar chart visualization
    """
    try:
        models_info = model_service.get_all_models_info()
        
        data = []
        for model in models_info:
            if model.get('accuracy') is not None:
                data.append({
                    "model": model.get('name', model.get('id')),
                    "name": model.get('name', model.get('id')),
                    "model_id": model.get('id'),
                    "accuracy": model.get('accuracy'),
                    "type": model.get('type', 'ML')
                })
        
        # Sort by accuracy descending
        data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return {
            "data": data,
            "count": len(data),
            "best_model": data[0] if data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sharpe-ratio")
async def compare_sharpe_ratio():
    """
    Compare Sharpe Ratio across all models
    
    Returns data formatted for bar chart visualization
    """
    try:
        models_info = model_service.get_all_models_info()
        
        data = []
        for model in models_info:
            if model.get('sharpe_ratio') is not None:
                data.append({
                    "model": model.get('name', model.get('id')),
                    "name": model.get('name', model.get('id')),
                    "model_id": model.get('id'),
                    "sharpe_ratio": model.get('sharpe_ratio'),
                    "type": model.get('type', 'ML')
                })
        
        # Sort by Sharpe Ratio descending
        data.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return {
            "data": data,
            "count": len(data),
            "best_model": data[0] if data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/returns")
async def compare_returns():
    """
    Compare total returns across all models
    
    Returns data formatted for bar chart visualization
    """
    try:
        models_info = model_service.get_all_models_info()
        
        data = []
        for model in models_info:
            if model.get('total_return') is not None:
                data.append({
                    "model": model.get('name', model.get('id')),
                    "name": model.get('name', model.get('id')),
                    "model_id": model.get('id'),
                    "total_return": model.get('total_return'),
                    "benchmark_return": model.get('benchmark_return', 0.0),
                    "type": model.get('type', 'ML')
                })
        
        # Sort by total return descending
        data.sort(key=lambda x: x['total_return'], reverse=True)
        
        return {
            "data": data,
            "count": len(data),
            "best_model": data[0] if data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all-metrics")
async def compare_all_metrics():
    """
    Compare all key metrics across models
    
    Returns comprehensive comparison data for multiple chart types
    """
    try:
        models_info = model_service.get_all_models_info()
        
        comparison = {
            "accuracy": [],
            "sharpe_ratio": [],
            "total_return": [],
            "max_drawdown": [],
            "auc": []
        }
        
        for model in models_info:
            model_name = model.get('name', model.get('id'))
            model_id = model.get('id')
            model_type = model.get('type', 'ML')
            
            if model.get('accuracy') is not None:
                comparison["accuracy"].append({
                    "model": model_name,
                    "model_id": model_id,
                    "value": model.get('accuracy'),
                    "type": model_type
                })
            
            if model.get('sharpe_ratio') is not None:
                comparison["sharpe_ratio"].append({
                    "model": model_name,
                    "model_id": model_id,
                    "value": model.get('sharpe_ratio'),
                    "type": model_type
                })
            
            if model.get('total_return') is not None:
                comparison["total_return"].append({
                    "model": model_name,
                    "model_id": model_id,
                    "value": model.get('total_return'),
                    "type": model_type
                })
            
            if model.get('max_drawdown') is not None:
                comparison["max_drawdown"].append({
                    "model": model_name,
                    "model_id": model_id,
                    "value": model.get('max_drawdown'),
                    "type": model_type
                })
            
            if model.get('auc') is not None:
                comparison["auc"].append({
                    "model": model_name,
                    "model_id": model_id,
                    "value": model.get('auc'),
                    "type": model_type
                })
        
        # Sort each metric
        for metric in comparison:
            comparison[metric].sort(key=lambda x: x['value'], reverse=True)
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml-vs-dl")
async def compare_ml_vs_dl():
    """
    Compare ML models vs DL models
    
    Returns aggregated statistics for ML and DL categories
    """
    try:
        models_info = model_service.get_all_models_info()
        
        ml_models = []
        dl_models = []
        
        for model in models_info:
            # Check model type from metadata
            model_type = model.get('type', 'ML').upper()
            if model_type == 'DL':
                dl_models.append(model)
            else:
                ml_models.append(model)
        
        def calculate_avg(models, metric):
            values = [m.get(metric) for m in models if m.get(metric)]
            return sum(values) / len(values) if values else 0
        
        comparison = {
            "ml": {
                "count": len(ml_models),
                "avg_accuracy": calculate_avg(ml_models, 'accuracy'),
                "avg_sharpe_ratio": calculate_avg(ml_models, 'sharpe_ratio'),
                "avg_total_return": calculate_avg(ml_models, 'total_return'),
                "models": [m.get('id') for m in ml_models]
            },
            "dl": {
                "count": len(dl_models),
                "avg_accuracy": calculate_avg(dl_models, 'accuracy'),
                "avg_sharpe_ratio": calculate_avg(dl_models, 'sharpe_ratio'),
                "avg_total_return": calculate_avg(dl_models, 'total_return'),
                "models": [m.get('id') for m in dl_models]
            }
        }
        
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
