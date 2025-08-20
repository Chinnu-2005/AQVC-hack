from fastapi import APIRouter, BackgroundTasks, HTTPException

from model.schemas import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    ModelStatus
)
from service.model_service import model_service
from util.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/model", tags=["model"])

@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model training status"""
    return ModelStatus(**model_service.get_status())

@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the quantum classifier with FTSE 100 data"""
    try:
        logger.info("Received training request")
        
        # Train the model
        results = model_service.train_model(
            start_date=request.start_date,
            end_date=request.end_date,
            lookback_period=request.lookback_period,
            test_size=request.test_size
        )
        
        return TrainingResponse(**results)
        
    except Exception as e:
        logger.error(f"Training request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using trained quantum classifier"""
    try:
        result = model_service.predict(request.features)
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/latest", response_model=PredictionResponse)
async def predict_latest():
    """Predict next day movement using latest FTSE 100 data"""
    try:
        result = model_service.predict_latest()
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Latest prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
