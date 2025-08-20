from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float] = Field(..., description="Feature vector for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.012, 0.05, 0.02, 0.65, 0.001, 0.15, 0.1, 0.008]
            }
        }

class TrainingRequest(BaseModel):
    """Request model for training"""
    start_date: str = Field(default="2020-01-01", description="Start date for training data (YYYY-MM-DD)")
    end_date: str = Field(default="2023-12-31", description="End date for training data (YYYY-MM-DD)")
    lookback_period: int = Field(default=10, description="Number of days to look back for features")
    test_size: float = Field(default=0.2, description="Test set size ratio")
    
    class Config:
        schema_extra = {
            "example": {
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
                "lookback_period": 10,
                "test_size": 0.2
            }
        }

class ModelStatus(BaseModel):
    """Model status response"""
    is_trained: bool
    accuracy: Optional[float] = None
    last_trained: Optional[str] = None
    feature_dim: Optional[int] = None
    training_samples: Optional[int] = None

class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: int = Field(..., description="0 for down, 1 for up")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    
class TrainingResponse(BaseModel):
    """Training response model"""
    message: str
    accuracy: float
    train_size: int
    test_size: int
    training_time: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"

class DataResponse(BaseModel):
    """Data response model"""
    data: List[dict]
    count: int
    period: str
