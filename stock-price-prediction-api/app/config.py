from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Stock Price Prediction API"
    api_version: str = "1.0.0"
    api_description: str = "ML API for predicting stock prices"
    
    # Financial Data API
    financial_data_api_key: str = Field(..., env="FINANCIAL_DATA_API_KEY")
    financial_data_base_url: str = "https://api.financialdatasets.ai"
    
    # Model Configuration
    default_tickers: List[str] = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    feature_window_days: int = 10
    prediction_horizon_days: int = 1
    
    # Training Configuration
    train_start_date: str = "2015-01-01"
    train_end_date: str = "2022-12-31"
    val_start_date: str = "2023-01-01"
    val_end_date: str = "2023-12-31"
    test_start_date: str = "2024-01-01"
    test_end_date: str = "2024-12-31"
    
    # Model Storage
    model_storage_path: str = "data/models"
    cache_storage_path: str = "data/cache"
    
    # Performance
    max_workers: int = 4
    api_rate_limit_per_minute: int = 60
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()