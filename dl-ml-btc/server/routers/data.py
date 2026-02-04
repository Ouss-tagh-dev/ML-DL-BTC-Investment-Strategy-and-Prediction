"""
Data router - Endpoints for Bitcoin historical data
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services import data_service

router = APIRouter()

class DataResponse(BaseModel):
    """Response model for data endpoints"""
    data: List[dict]
    count: int
    columns: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class StatsResponse(BaseModel):
    """Response model for statistics"""
    total_rows: int
    total_columns: int
    date_range: dict
    missing_values: dict
    numeric_stats: dict

@router.get("/historical", response_model=DataResponse)
async def get_historical_data(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(1000, description="Maximum number of rows", ge=1, le=10000),
    columns: Optional[str] = Query(None, description="Comma-separated column names")
):
    """
    Get historical Bitcoin OHLCV data with optional filtering
    
    - **start_date**: Filter data from this date
    - **end_date**: Filter data until this date
    - **limit**: Maximum number of rows to return
    - **columns**: Specific columns to return (comma-separated)
    """
    try:
        cols = columns.split(',') if columns else None
        result = data_service.get_historical_data(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            columns=cols
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features", response_model=DataResponse)
async def get_features(
    feature_names: Optional[str] = Query(None, description="Comma-separated feature names"),
    limit: Optional[int] = Query(1000, description="Maximum number of rows", ge=1, le=10000)
):
    """
    Get specific features from the dataset
    
    - **feature_names**: Specific features to return (comma-separated)
    - **limit**: Maximum number of rows to return
    """
    try:
        features = feature_names.split(',') if feature_names else None
        result = data_service.get_features(
            feature_names=features,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=StatsResponse)
async def get_statistics():
    """
    Get dataset statistics including:
    - Total rows and columns
    - Date range
    - Missing values
    - Numeric statistics (mean, std, min, max, median)
    """
    try:
        return data_service.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_data(
    n: int = Query(1, description="Number of latest rows", ge=1, le=100)
):
    """
    Get the latest n rows of data
    
    - **n**: Number of latest rows to return
    """
    try:
        return data_service.get_latest_data(n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-names")
async def get_feature_names():
    """
    Get all available feature names in the dataset
    """
    try:
        features = data_service.get_feature_names()
        return {
            "features": features,
            "count": len(features)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
