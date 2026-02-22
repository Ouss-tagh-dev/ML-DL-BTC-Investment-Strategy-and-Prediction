"""
Settings router - Endpoints for application configuration and system info
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services import settings_service

router = APIRouter()


class SettingsUpdate(BaseModel):
    """Payload for updating mutable settings."""
    cache_models: Optional[bool] = None
    cache_data: Optional[bool] = None
    log_level: Optional[str] = None
    cors_origins: Optional[List[str]] = None


@router.get("/")
async def get_settings():
    """
    Get current application settings.
    Returns API config, CORS, model config, and data paths.
    """
    try:
        return settings_service.get_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/")
async def update_settings(payload: SettingsUpdate):
    """
    Update mutable runtime settings.

    - **cache_models**: Enable/disable model caching
    - **cache_data**: Enable/disable data caching
    - **log_level**: Set logging level (DEBUG, INFO, WARNING, ERROR)
    - **cors_origins**: Update allowed CORS origins
    """
    try:
        data = payload.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No settings to update")
        result = settings_service.update_settings(data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-info")
async def get_system_info():
    """
    Get system information: Python version, uptime, OS, memory, CPU.
    """
    try:
        return settings_service.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
