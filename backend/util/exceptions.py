class QuantumMLException(Exception):
    """Base exception for Quantum ML application"""
    pass

class ModelNotTrainedException(QuantumMLException):
    """Raised when trying to use an untrained model"""
    pass

class DataFetchException(QuantumMLException):
    """Raised when data fetching fails"""
    pass

class InvalidFeatureException(QuantumMLException):
    """Raised when invalid features are provided"""
    pass

class TrainingException(QuantumMLException):
    """Raised when model training fails"""
    pass