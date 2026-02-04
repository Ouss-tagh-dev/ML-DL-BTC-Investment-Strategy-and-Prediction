"""
FastAPI main application for Bitcoin ML/DL Dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from server.routers import data, models, metrics, comparison
from server.services import model_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Pre-loads all models on startup to ensure instant inference.
    """
    # Startup: Load all models
    logger.info("Starting up: Loading all models...")
    try:
        success = model_service.load_all_models()
        if success:
            logger.info("All models loaded successfully.")
        else:
            logger.warning("Some models failed to load.")
    except Exception as e:
        logger.error(f"Error loading models on startup: {e}")
    
    yield
    
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Bitcoin ML/DL Dashboard API",
    version="1.0.0",
    description="API for Bitcoin price prediction using multiple ML/DL models",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(comparison.router, prefix="/api/comparison", tags=["Comparison"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bitcoin ML/DL Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": model_service.models_loaded
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
