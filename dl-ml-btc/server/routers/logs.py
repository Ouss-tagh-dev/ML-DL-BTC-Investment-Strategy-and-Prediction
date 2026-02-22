"""
Logs router - Endpoints for system log retrieval and management
"""
from fastapi import APIRouter, Query
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from server.services.log_service import log_handler

router = APIRouter()


@router.get("/")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by level: DEBUG, INFO, WARNING, ERROR"),
    search: Optional[str] = Query(None, description="Search term to filter messages"),
    limit: int = Query(200, description="Max number of log entries", ge=1, le=500),
):
    """
    Retrieve application logs from the in-memory buffer.

    - **level**: Filter by log level
    - **search**: Full-text search in log messages
    - **limit**: Maximum entries to return
    """
    logs = log_handler.get_logs(level=level, search=search, limit=limit)
    return {
        "logs": logs,
        "count": len(logs),
        "total_buffered": log_handler.count,
    }


@router.delete("/")
async def clear_logs():
    """Clear the in-memory log buffer."""
    log_handler.clear()
    return {"message": "Log buffer cleared", "count": 0}
