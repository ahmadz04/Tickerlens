import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from app.models.base import BaseModel

class LinearModel(BaseModel):
    """Linear regression model implementation"""
    
    def __init__(self, model_type: str = "linear"):
        super().__init__(model_type)
        self.scaler = RobustScaler()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameters: Dict[str, Any] = None) -> Dict[str, float]:
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize model based on type
        if self.model_type == "ridge":
            if hyperparameters and 'alpha' in hyperparameters:
                self.model = Ridge(alpha=hyperparameters['alpha'])
            else:
                # Grid search for best alpha
                param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
                ridge = Ridge()
                grid_search = GridSearchCV(
                    ridge, param_grid, cv=3, 
                    scoring='neg_mean_absolute_error', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
        else:
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        self.training_metrics = self.calculate_metrics(y_train, train_pred)
        self.validation_metrics = self.calculate_metrics(y_val, val_pred)
        
        self.is_trained = True
        return self.validation_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        
        # For linear models, use absolute coefficients as importance
        importances = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, importances))