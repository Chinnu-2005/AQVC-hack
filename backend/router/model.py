from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

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
    """Train the quantum classifier with 3 years of FTSE 100 data ending yesterday (dynamic sliding window)"""
    try:
        logger.info("Received training request")
        
        # Train the model (dates are optional, will use 3 years ending yesterday)
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

@router.post("/predict/latest")
async def predict_latest():
    """Predict today's intraday movements at 2-hour intervals (8:00, 10:00, 12:00, 14:00, 16:00). 
    Automatically retrains model daily with 3 years of data ending yesterday."""
    try:
        result = model_service.predict_intraday_latest()
        return result
        
    except Exception as e:
        logger.error(f"Latest prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/date")
async def predict_for_date(target_date: str = Query(..., description="Target date in YYYY-MM-DD format")):
    """Predict intraday movements for a specific date using 3 years of data ending the day before.
    Returns predictions and actual data for comparison if it's a historical date."""
    try:
        result = model_service.predict_for_date(target_date)
        return result
        
    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Date prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
