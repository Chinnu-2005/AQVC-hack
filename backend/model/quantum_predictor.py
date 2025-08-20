import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from util.config import settings
from util.logger import setup_logger
from util.exceptions import TrainingException, InvalidFeatureException

logger = setup_logger(__name__)

class QuantumStockPredictor:
    """
    Variational Quantum Classifier for stock market prediction
    """
    
    def __init__(self, feature_dim: int = None, reps: int = None):
        """
        Initialize Quantum Stock Predictor with VQC
        
        Args:
            feature_dim: Dimension of feature vector
            reps: Number of repetitions in ansatz
        """
        self.feature_dim = feature_dim or settings.FEATURE_DIM
        self.reps = reps or settings.QUANTUM_REPS
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Set random seed
        algorithm_globals.random_seed = settings.RANDOM_SEED
        np.random.seed(settings.RANDOM_SEED)
        
        logger.info(f"Initialized QuantumStockPredictor with feature_dim={self.feature_dim}, reps={self.reps}")
        
    def create_quantum_circuit(self) -> QuantumCircuit:
        """Create the quantum circuit for VQC"""
        try:
            # Feature map - encodes classical data into quantum states
            feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dim,
                reps=1,
                entanglement='linear'
            )
            
            # Ansatz - parameterized quantum circuit
            ansatz = RealAmplitudes(
                num_qubits=self.feature_dim,
                reps=self.reps,
                entanglement='linear'
            )
            
            # Combine feature map and ansatz
            circuit = QuantumCircuit(self.feature_dim)
            circuit.compose(feature_map, inplace=True)
            circuit.compose(ansatz, inplace=True)
            
            logger.info("Created quantum circuit successfully")
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating quantum circuit: {str(e)}")
            raise TrainingException(f"Failed to create quantum circuit: {str(e)}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        try:
            # Returns
            data['Returns'] = data['Close'].pct_change()
            
            # Simple Moving Averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            # RSI
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            # MACD
            data['MACD'] = self.calculate_macd(data['Close'])
            
            # Volatility
            data['Volatility'] = data['Returns'].rolling(window=10).std()
            
            # Volume Moving Average
            data['Volume_MA'] = data['Volume'].rolling(window=5).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def prepare_features(self, data: pd.DataFrame, lookback_period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from FTSE 100 data
        
        Args:
            data: DataFrame with OHLCV data
            lookback_period: Number of days to look back
            
        Returns:
            tuple: (features, labels)
        """
        try:
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            features = []
            labels = []
            
            # Create feature vectors
            for i in range(lookback_period, len(data)):
                # Current day features (8 features total)
                current_features = [
                    data['Returns'].iloc[i-1] if not pd.isna(data['Returns'].iloc[i-1]) else 0.0,
                    ((data['Close'].iloc[i-1] - data['SMA_5'].iloc[i-1]) / data['SMA_5'].iloc[i-1]) 
                    if not pd.isna(data['SMA_5'].iloc[i-1]) and data['SMA_5'].iloc[i-1] != 0 else 0.0,
                    ((data['SMA_5'].iloc[i-1] - data['SMA_20'].iloc[i-1]) / data['SMA_20'].iloc[i-1])
                    if not pd.isna(data['SMA_20'].iloc[i-1]) and data['SMA_20'].iloc[i-1] != 0 else 0.0,
                    (data['RSI'].iloc[i-1] / 100.0) if not pd.isna(data['RSI'].iloc[i-1]) else 0.5,
                    data['MACD'].iloc[i-1] if not pd.isna(data['MACD'].iloc[i-1]) else 0.0,
                    data['Volatility'].iloc[i-1] if not pd.isna(data['Volatility'].iloc[i-1]) else 0.0,
                    ((data['Volume'].iloc[i-1] - data['Volume_MA'].iloc[i-1]) / data['Volume_MA'].iloc[i-1])
                    if not pd.isna(data['Volume_MA'].iloc[i-1]) and data['Volume_MA'].iloc[i-1] != 0 else 0.0,
                    data['Returns'].iloc[i-5:i-1].mean() if not pd.isna(data['Returns'].iloc[i-5:i-1].mean()) else 0.0
                ]
                
                # Ensure all features are valid numbers
                current_features = [float(f) if not (pd.isna(f) or np.isinf(f)) else 0.0 for f in current_features]
                
                features.append(current_features)
                
                # Label: 1 if next day's close > current close, 0 otherwise
                label = 1 if data['Close'].iloc[i] > data['Close'].iloc[i-1] else 0
                labels.append(label)
            
            features_array = np.array(features)
            labels_array = np.array(labels)
            
            logger.info(f"Prepared {len(features_array)} training samples with {features_array.shape[1]} features")
            return features_array, labels_array
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise TrainingException(f"Failed to prepare features: {str(e)}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the Variational Quantum Classifier
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set ratio
            
        Returns:
            dict: Training results
        """
        try:
            logger.info("Starting VQC training...")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=settings.RANDOM_SEED, stratify=y
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create quantum circuit
            qc = self.create_quantum_circuit()
            
            # Create SamplerQNN
            sampler = Sampler()
            qnn = SamplerQNN(
                circuit=qc,
                input_params=qc.parameters[:self.feature_dim],
                weight_params=qc.parameters[self.feature_dim:],
                sampler=sampler
            )
            
            # Create and configure VQC
            optimizer = SPSA(maxiter=100)
            
            # Create feature map and ansatz separately for VQC
            feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dim,
                reps=1,
                entanglement='linear'
            )
            
            ansatz = RealAmplitudes(
                num_qubits=self.feature_dim,
                reps=self.reps,
                entanglement='linear'
            )
            
            self.classifier = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer
            )
            
            logger.info("Training quantum classifier...")
            
            # Train the classifier
            self.classifier.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = self.classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            logger.info(f"Training completed. Test accuracy: {accuracy:.4f}")
            
            return {
                "accuracy": accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "classification_report": class_report
            }
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise TrainingException(f"Model training failed: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, float]:
        """
        Make prediction using trained VQC
        
        Args:
            features: Feature vector
            
        Returns:
            tuple: (prediction, probability, confidence)
        """
        if self.classifier is None:
            raise InvalidFeatureException("Model not trained yet")
        
        try:
            # Ensure features is the right shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Validate feature dimensions
            if features.shape[1] != self.feature_dim:
                raise InvalidFeatureException(
                    f"Expected {self.feature_dim} features, got {features.shape[1]}"
                )
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.classifier.predict(features_scaled)[0]
            
            # Get prediction probabilities
            try:
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                probability = max(probabilities)
                confidence = abs(probabilities[1] - probabilities[0]) if len(probabilities) > 1 else 0.5
            except Exception:
                probability = 0.6 if prediction == 1 else 0.4
                confidence = 0.2
            
            return int(prediction), float(probability), float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise InvalidFeatureException(f"Prediction failed: {str(e)}")
