import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

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
        self.model_dir: Path = Path(settings.MODEL_DIR)
        self.model_path: Path = self.model_dir / settings.MODEL_FILE
        self.metadata_path: Path = self.model_dir / settings.MODEL_METADATA_FILE
        self.model_metadata = {
            "is_trained": False,
            "accuracy": None,
            "last_trained": None,
            "feature_dim": settings.FEATURE_DIM,
            "training_samples": None
        }
        # Try to load a persisted model at startup of the service
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load a saved model and metadata from disk."""
        try:
            if self.model_path.exists():
                predictor = QuantumStockPredictor()
                predictor.load(str(self.model_path))
                self.predictor = predictor
                # Load metadata if available
                if self.metadata_path.exists():
                    with self.metadata_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.model_metadata.update(meta)
                else:
                    # Minimal metadata if none saved
                    self.model_metadata.update({
                        "is_trained": True,
                        "last_trained": None,
                        "feature_dim": predictor.feature_dim,
                    })
                logger.info("Model loaded from disk and ready for predictions")
        except Exception as e:
            logger.error(f"Failed to load model from disk: {str(e)}")

    def should_retrain(self) -> bool:
        """
        Check if model should be retrained based on last training date
        
        Returns:
            bool: True if model should be retrained
        """
        if not self.model_metadata.get("is_trained", False):
            return True
        
        last_trained = self.model_metadata.get("last_trained")
        if not last_trained:
            return True
        
        try:
            last_trained_date = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
            current_date = datetime.now()
            
            # Retrain if last training was more than 1 day ago
            days_since_training = (current_date - last_trained_date).days
            return days_since_training >= 1
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {str(e)}")
            return True

    def auto_train_if_needed(self) -> bool:
        """
        Automatically train model if needed
        
        Returns:
            bool: True if model was trained, False if already up to date
        """
        if self.should_retrain():
            logger.info("Model needs retraining - starting automatic training")
            try:
                self.train_model()  # Uses default 3-year period
                return True
            except Exception as e:
                logger.error(f"Automatic training failed: {str(e)}")
                return False
        else:
            logger.info("Model is up to date - no retraining needed")
            return False

    def _persist_after_training(self, results: Dict[str, Any]) -> None:
        """Persist model and metadata to disk after successful training."""
        try:
            self.model_dir.mkdir(exist_ok=True, parents=True)
            if self.predictor is not None:
                self.predictor.save(str(self.model_path))
            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info("Persisted model and metadata to disk")
        except Exception as e:
            logger.error(f"Failed to persist model/metadata: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return self.model_metadata.copy()
    
    def train_model(self, start_date: str = None, end_date: str = None, 
                   lookback_period: int = 10, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the quantum model with 3 years of data ending yesterday
        
        Args:
            start_date: Training data start date (optional, defaults to 3 years ago)
            end_date: Training data end date (optional, defaults to yesterday)
            lookback_period: Days to look back for features
            test_size: Test set ratio
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting model training process...")
            
            # Use 3 years of data ending yesterday if dates not provided
            if start_date is None or end_date is None:
                start_date, end_date = DataService.get_training_date_range()
                logger.info(f"Using default training period: {start_date} to {end_date}")
            
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
            
            # Persist model and metadata
            self._persist_after_training(results)

            return {
                "message": "Model trained successfully",
                "accuracy": results["accuracy"],
                "train_size": results["train_size"],
                "test_size": results["test_size"],
                "training_time": datetime.now().isoformat(),
                "training_period": f"{start_date} to {end_date}"
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

    def predict_intraday_latest(self) -> Dict[str, Any]:
        """
        Predict today's intraday movements at 2-hour intervals using 3 years of historical data ending yesterday
        
        Returns:
            Intraday prediction results
        """
        # Check if model needs retraining and auto-train if needed
        self.auto_train_if_needed()
        
        if not self.model_metadata["is_trained"] or self.predictor is None:
            raise ModelNotTrainedException("Model not trained yet")
        
        try:
            # Get 3 years of historical data ending yesterday
            start_date, end_date = DataService.get_training_date_range()
            historical_data = DataService.fetch_ftse_data(start_date, end_date)
            
            # Make intraday predictions using historical data only
            result = self.predictor.predict_intraday(historical_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Intraday prediction failed: {str(e)}")
            raise

    def predict_for_date(self, target_date: str) -> Dict[str, Any]:
        """
        Predict intraday movements for a specific date using 3 years of data ending the day before
        
        Args:
            target_date: Target date string (YYYY-MM-DD)
            
        Returns:
            dict: Predictions and actual data for the target date
        """
        try:
            logger.info(f"Predicting for date: {target_date}")
            
            # Validate date format
            if not DataService.validate_date_format(target_date):
                raise ValueError(f"Invalid date format: {target_date}. Use YYYY-MM-DD format")
            
            # Get training date range for the target date
            start_date, end_date = DataService.get_training_date_range_for_date(target_date)
            logger.info(f"Training period for {target_date}: {start_date} to {end_date}")
            
            # Fetch historical data for training
            historical_data = DataService.fetch_ftse_data(start_date, end_date)
            
            # Initialize predictor for this specific date
            predictor = QuantumStockPredictor(
                feature_dim=settings.FEATURE_DIM,
                reps=settings.QUANTUM_REPS
            )
            
            # Prepare features and train model
            X, y = predictor.prepare_features(historical_data, lookback_period=10)
            
            if len(X) == 0:
                raise TrainingException("No valid features generated from the data")
            
            logger.info(f"Prepared {len(X)} training samples for {target_date}")
            
            # Train the model
            training_results = predictor.train(X, y, test_size=0.2)
            
            # Make intraday predictions
            prediction_result = predictor.predict_intraday(historical_data)
            
            # Fetch actual data for comparison
            actual_data = DataService.fetch_actual_data_for_date(target_date)
            
            # Determine if this is a historical date or future date
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            current_dt = datetime.now()
            is_historical = target_dt.date() < current_dt.date()
            
            # Add metadata to the result
            result = {
                "target_date": target_date,
                "is_historical_date": is_historical,
                "training_period": f"{start_date} to {end_date}",
                "training_accuracy": training_results["accuracy"],
                "training_samples": training_results["train_size"] + training_results["test_size"],
                "predictions": prediction_result["predictions"],
                "base_feature_vector": prediction_result["base_feature_vector"],
                "actual_data": actual_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Date prediction failed for {target_date}: {str(e)}")
            raise

# Global model service instance
model_service = ModelService()
