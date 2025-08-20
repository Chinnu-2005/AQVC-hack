from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from util.exceptions import (
    QuantumMLException,
    ModelNotTrainedException,
    DataFetchException,
    InvalidFeatureException,
    TrainingException
)

logger = logging.getLogger(__name__)

async def quantum_ml_exception_handler(request: Request, exc: QuantumMLException):
    """Handle custom Quantum ML exceptions"""
    logger.error(f"QuantumMLException: {str(exc)}")
    
    status_code = 500
    if isinstance(exc, ModelNotTrainedException):
        status_code = 400
    elif isinstance(exc, (DataFetchException, InvalidFeatureException)):
        status_code = 400
    elif isinstance(exc, TrainingException):
        status_code = 500
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "timestamp": request.url.path
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP exception: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )
