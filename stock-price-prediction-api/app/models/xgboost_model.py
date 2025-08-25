import pandas as pd
import numpy as np
from typing import Dict, Any
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from app.models.base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model implementation"""
    
    def __init__(self):
        super().__init__("xgboost")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              hyperparameters: Dict[str, Any] = None) -> Dict[str, float]:
        
        self.feature_names = X_train.columns.tolist()
        
        if hyperparameters:
            self.model = XGBRegressor(**hyperparameters, n_jobs=-1)
        else:
            # Default hyperparameters with grid search
            param_grid = {
                'n_estimators': [200, 400],
                'max_depth': [3, 5],
                'learning_rate': [0.06, 0.1],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0]
            }
            
            xgb = XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb, param_grid, cv=3,
                scoring='neg_mean_absolute_error', n_jobs=2
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        
        if not hasattr(self, 'model') or self.model is None:
            self.model = XGBRegressor(random_state=42, n_jobs=-1)
            
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        self.training_metrics = self.calculate_metrics(y_train, train_pred)
        self.validation_metrics = self.calculate_metrics(y_val, val_pred)
        
        self.is_trained = True
        return self.validation_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))