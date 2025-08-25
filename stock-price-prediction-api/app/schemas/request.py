from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date

class PredictionRequest(BaseModel):
    ticker: str
    model_type: str = "linear"  # linear, ridge, xgboost
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError("Invalid ticker symbol")
        return v.upper()
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ['linear', 'ridge', 'xgboost']
        if v not in allowed_models:
            raise ValueError(f"Model type must be one of: {allowed_models}")
        return v

class BatchPredictionRequest(BaseModel):
    tickers: List[str]
    model_type: str = "linear"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or len(v) > 20:
            raise ValueError("Must provide 1-20 ticker symbols")
        return [ticker.upper() for ticker in v]

class ModelTrainingRequest(BaseModel):
    tickers: List[str]
    model_type: str = "linear"
    train_start: date
    train_end: date
    hyperparameters: Optional[Dict[str, Any]] = None