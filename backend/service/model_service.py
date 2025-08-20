import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from model.quantum_predictor import QuantumStockPredictor
from service.data_service import DataService
from util.config import settings
from util.logger import setup_logger
from util.exceptions import ModelNotTrainedException, TrainingException

logger = setup_logger(__name__)

class ModelService:
    """Service for quantum model operations"""
    
    def __init__(self):
        self.predictor: Optional[QuantumStockPredictor] = None
        self.model_metadata = {
            "is_trained": False,
            "accuracy": None,
            "last_trained": None,
            "feature_dim": settings.FEATURE_DIM,
            "training_samples": None
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return self.model_metadata.copy()
    
    def train_model(self, start_date: str, end_date: str, 
                   lookback_period: int = 10, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the quantum model
        
        Args:
            start_date: Training data start date
            end_date: Training data end date
            lookback_period: Days to look back for features
            test_size: Test set ratio
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting model training process...")
            
            # Fetch data
            data = DataService.fetch_ftse_data(start_date, end_date)
            
            # Initialize predictor
            self.predictor = QuantumStockPredictor(
                feature_dim=settings.FEATURE_DIM,
                reps=settings.QUANTUM_REPS
            )
            
            # Prepare features
            X, y = self.predictor.prepare_features(data, lookback_period)
            
            if len(X) == 0:
                raise TrainingException("No valid features generated from the data")
            
            logger.info(f"Prepared {len(X)} training samples")
            
            # Train the model
            results = self.predictor.train(X, y, test_size)
            
            # Update metadata
            self.model_metadata.update({
                "is_trained": True,
                "accuracy": results["accuracy"],
                "last_trained": datetime.now().isoformat(),
                "training_samples": results["train_size"] + results["test_size"]
            })
            
            logger.info("Model training completed successfully")
            
            return {
                "message": "Model trained successfully",
                "accuracy": results["accuracy"],
                "train_size": results["train_size"],
                "test_size": results["test_size"],
                "training_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise TrainingException(f"Model training failed: {str(e)}")
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction with feature vector
        
        Args:
            features: Feature vector
            
        Returns:
            Prediction results
        """
        if not self.model_metadata["is_trained"] or self.predictor is None:
            raise ModelNotTrainedException("Model not trained yet")
        
        try:
            features_array = np.array(features)
            prediction, probability, confidence = self.predictor.predict(features_array)
            
            return {
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_latest(self) -> Dict[str, Any]:
        """
        Predict using latest market data
        
        Returns:
            Prediction results
        """
        if not self.model_metadata["is_trained"] or self.predictor is None:
            raise ModelNotTrainedException("Model not trained yet")
        
        try:
            # Fetch latest data
            data = DataService.get_latest_data(days=30)
            
            # Prepare features
            X, _ = self.predictor.prepare_features(data, lookback_period=10)
            
            if len(X) == 0:
                raise Exception("Could not generate features from latest data")
            
            # Use the most recent feature vector
            latest_features = X[-1]
            prediction, probability, confidence = self.predictor.predict(latest_features)
            
            return {
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Latest prediction failed: {str(e)}")
            raise

# Global model service instance
model_service = ModelService()
