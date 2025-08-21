"""
Configuration settings
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Model Configuration
    RANDOM_SEED: int = 42
    FEATURE_DIM: int = 8
    QUANTUM_REPS: int = 2
    MODEL_DIR: str = "models"
    MODEL_FILE: str = "quantum_vqc.joblib"
    MODEL_METADATA_FILE: str = "model_metadata.json"
    
    # Cache Configuration
    CACHE_DIR: str = "cache"
    CACHE_FILE: str = "predictions_cache.csv"
    
    # Data Configuration
    DEFAULT_START_DATE: str = "2020-01-01"
    DEFAULT_END_DATE: str = "2023-12-31"
    DEFAULT_LOOKBACK_PERIOD: int = 10
    DEFAULT_TEST_SIZE: float = 0.2
    
    # Database Configuration (optional)
    MONGODB_URI: Optional[str] = None
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra fields from .env
    }

settings = Settings()