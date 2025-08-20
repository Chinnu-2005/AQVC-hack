from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from router import  health,model, data
from middleware.error_handler import (
    quantum_ml_exception_handler,
    validation_exception_handler,
    http_exception_handler
)
from util.exceptions import QuantumMLException
from util.config import settings
from util.logger import setup_logger

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI application
    """
    # Setup logging
    logger = setup_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title="Quantum ML FTSE 100 Forecasting API",
        description="A Quantum Machine Learning API for FTSE 100 stock market prediction using Variational Quantum Classifier",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    app.add_exception_handler(QuantumMLException, quantum_ml_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(data.router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Quantum ML FTSE 100 Forecasting API",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "model_status": "/model/status",
                "train": "/model/train",
                "predict": "/model/predict",
                "predict_latest": "/model/predict/latest",
                "latest_data": "/data/latest",
                "historical_data": "/data/historical"
            }
        }
    
    logger.info("FastAPI application created successfully")
    return app
