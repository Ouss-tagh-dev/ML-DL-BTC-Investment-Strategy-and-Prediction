"""
Settings service - Reads / updates runtime configuration and system info
"""
import logging
import platform
import sys
import time
from datetime import datetime

_start_time = time.time()

logger = logging.getLogger(__name__)


def get_settings():
    """Return current runtime settings (read from config.py)."""
    from config import settings as cfg

    return {
        "api": {
            "project_name": cfg.PROJECT_NAME,
            "version": cfg.VERSION,
            "description": cfg.DESCRIPTION,
            "prefix": cfg.API_V1_PREFIX,
        },
        "cors": {
            "origins": cfg.CORS_ORIGINS,
        },
        "models": {
            "available": cfg.AVAILABLE_MODELS,
            "cache_enabled": cfg.CACHE_MODELS,
        },
        "data": {
            "data_dir": str(cfg.DATA_DIR),
            "models_dir": str(cfg.MODELS_DIR),
            "features_file": str(cfg.FEATURES_FILE),
            "cache_enabled": cfg.CACHE_DATA,
        },
    }


def update_settings(payload: dict):
    """
    Update mutable runtime settings.
    Only a subset of settings are writable at runtime.
    """
    from config import settings as cfg

    updated = []

    if "cache_models" in payload:
        cfg.CACHE_MODELS = bool(payload["cache_models"])
        updated.append("cache_models")

    if "cache_data" in payload:
        cfg.CACHE_DATA = bool(payload["cache_data"])
        updated.append("cache_data")

    if "log_level" in payload:
        level = payload["log_level"].upper()
        numeric = getattr(logging, level, None)
        if numeric is not None:
            logging.getLogger().setLevel(numeric)
            updated.append("log_level")

    if "cors_origins" in payload and isinstance(payload["cors_origins"], list):
        cfg.CORS_ORIGINS = payload["cors_origins"]
        updated.append("cors_origins")

    logger.info("Settings updated: %s", updated)
    return {"updated_fields": updated}


def get_system_info():
    """Return system-level information."""
    uptime_sec = time.time() - _start_time
    hours, remainder = divmod(int(uptime_sec), 3600)
    minutes, seconds = divmod(remainder, 60)

    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "os": platform.system(),
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "uptime_seconds": int(uptime_sec),
        "started_at": datetime.fromtimestamp(_start_time).isoformat(),
        "log_level": logging.getLevelName(logging.getLogger().level),
    }

    # Optional: memory info via psutil (graceful fallback)
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent,
        }
        info["cpu_percent"] = psutil.cpu_percent(interval=0)
    except ImportError:
        info["memory"] = None
        info["cpu_percent"] = None

    return info
