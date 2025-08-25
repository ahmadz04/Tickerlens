from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import joblib
import os
from datetime import datetime

class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameters: Dict[str, Any] = None) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, path)
        self.is_trained = True
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data.get('training_metrics', {})
        self.validation_metrics = model_data.get('validation_metrics', {})
        self.is_trained = True
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape)
        }