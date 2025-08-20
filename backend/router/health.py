from fastapi import APIRouter
from datetime import datetime

from model.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # Add any specific readiness checks here
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "database": "ok",
            "quantum_backend": "ok"
        }
    }

