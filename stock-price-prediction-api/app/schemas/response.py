from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionResult(BaseModel):
    ticker: str
    predicted_price: float
    confidence_interval: Optional[Dict[str, float]] = None
    prediction_date: datetime
    model_used: str
    features_used: List[str]

class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2_score: float
    mape: Optional[float] = None

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model_performance: Optional[ModelMetrics] = None
    processing_time_ms: float
    timestamp: datetime

class ModelTrainingResponse(BaseModel):
    model_id: str
    model_type: str
    training_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    feature_importance: Optional[Dict[str, float]] = None
    training_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    models_loaded: List[str]