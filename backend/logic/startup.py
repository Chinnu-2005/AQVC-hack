import os
from pathlib import Path

from util.config import settings
from util.logger import setup_logger

logger = setup_logger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_environment():
    """Validate environment configuration"""
    required_settings = [
        "API_HOST",
        "API_PORT",
        "FEATURE_DIM",
        "QUANTUM_REPS"
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not hasattr(settings, setting):
            missing_settings.append(setting)
    
    if missing_settings:
        raise ValueError(f"Missing required settings: {missing_settings}")
    
    logger.info("Environment validation completed")

def startup():
    """Run startup tasks"""
    logger.info("Running startup tasks...")
    
    create_directories()
    validate_environment()
    
    logger.info("Startup completed successfully")
