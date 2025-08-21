from typing import List, Optional, Dict, Any
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
    start_date: Optional[str] = Field(default=None, description="Start date for training data (YYYY-MM-DD), defaults to 3 years ago")
    end_date: Optional[str] = Field(default=None, description="End date for training data (YYYY-MM-DD), defaults to yesterday")
    lookback_period: int = Field(default=10, description="Number of days to look back for features")
    test_size: float = Field(default=0.2, description="Test set size ratio")
    
    class Config:
        schema_extra = {
            "example": {
                "start_date": "2022-01-01",
                "end_date": "2024-12-31",
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

class IntradayPredictionResponse(BaseModel):
    """Intraday prediction response model"""
    predictions: Dict[str, Dict[str, Any]] = Field(..., description="Predictions for each time interval")
    timestamp: str = Field(..., description="Prediction timestamp")
    feature_vector: List[float] = Field(..., description="Feature vector used for prediction")

class TrainingResponse(BaseModel):
    """Training response model"""
    message: str
    accuracy: float
    train_size: int
    test_size: int
    training_time: Optional[str] = None
    training_period: Optional[str] = None

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

class DatePredictionRequest(BaseModel):
    """Request model for date-specific predictions"""
    target_date: str = Field(..., description="Target date in YYYY-MM-DD format")
    
    class Config:
        schema_extra = {
            "example": {
                "target_date": "2024-01-15"
            }
        }

class ActualDataPoint(BaseModel):
    """Actual data point model"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    returns: float

class ActualDataResponse(BaseModel):
    """Actual data response model"""
    target_date: str
    target_data: Optional[ActualDataPoint] = None
    context_data: List[ActualDataPoint]
    data_available: bool

class DatePredictionResponse(BaseModel):
    """Date prediction response model"""
    target_date: str
    is_historical_date: bool
    training_period: str
    training_accuracy: float
    training_samples: int
    predictions: Dict[str, Dict[str, Any]]
    base_feature_vector: List[float]
    actual_data: ActualDataResponse
    timestamp: str
