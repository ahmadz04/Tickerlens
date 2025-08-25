#!/usr/bin/env python3
"""
Test script for the first part of the stock prediction API
"""
import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-price-prediction-api'))

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from app.config import settings
        print(f"✅ Config loaded successfully")
        print(f"   API Title: {settings.api_title}")
        print(f"   Default tickers: {settings.default_tickers}")
        print(f"   Model storage path: {settings.model_storage_path}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_exceptions():
    """Test custom exceptions"""
    print("\nTesting custom exceptions...")
    try:
        from app.utils.exceptions import StockPredictionError, DataFetchError
        
        # Test base exception
        try:
            raise StockPredictionError("Test error")
        except StockPredictionError as e:
            print(f"✅ Base exception works: {e}")
        
        # Test specific exception
        try:
            raise DataFetchError("Test data fetch error")
        except DataFetchError as e:
            print(f"✅ DataFetchError works: {e}")
            
        return True
    except Exception as e:
        print(f"❌ Exceptions test failed: {e}")
        return False

def test_logger():
    """Test logging setup"""
    print("\nTesting logger...")
    try:
        from app.utils.logger import logger
        logger.info("Test log message")
        print("✅ Logger setup successful")
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False

def test_schemas():
    """Test Pydantic schemas"""
    print("\nTesting schemas...")
    try:
        from app.schemas.request import PredictionRequest, BatchPredictionRequest
        from app.schemas.response import PredictionResult, ModelMetrics
        
        # Test prediction request
        req = PredictionRequest(ticker="AAPL", model_type="linear")
        print(f"✅ PredictionRequest created: {req.ticker}, {req.model_type}")
        
        # Test batch request
        batch_req = BatchPredictionRequest(tickers=["AAPL", "GOOGL"], model_type="ridge")
        print(f"✅ BatchPredictionRequest created: {batch_req.tickers}")
        
        # Test response models
        result = PredictionResult(
            ticker="AAPL",
            predicted_price=150.0,
            prediction_date=datetime.now(),
            model_used="linear",
            features_used=["close", "volume"]
        )
        print(f"✅ PredictionResult created: {result.ticker} - ${result.predicted_price}")
        
        return True
    except Exception as e:
        print(f"❌ Schemas test failed: {e}")
        return False

def test_base_model():
    """Test base model interface"""
    print("\nTesting base model...")
    try:
        from app.models.base import BaseModel
        
        # Test that it's abstract (should raise TypeError)
        try:
            BaseModel("test")
            print("❌ BaseModel should be abstract")
            return False
        except TypeError:
            print("✅ BaseModel is properly abstract")
            return True
            
    except Exception as e:
        print(f"❌ Base model test failed: {e}")
        return False

def test_linear_model():
    """Test linear model with dummy data"""
    print("\nTesting linear model...")
    try:
        from app.models.linear_model import LinearModel
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y_train = pd.Series(2 * X_train['feature1'] + 3 * X_train['feature2'] + np.random.randn(100) * 0.1)
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20)
        })
        y_val = pd.Series(2 * X_val['feature1'] + 3 * X_val['feature2'] + np.random.randn(20) * 0.1)
        
        # Test linear regression
        model = LinearModel("linear")
        metrics = model.train(X_train, y_train, X_val, y_val)
        print(f"✅ Linear model trained - MAE: {metrics['mae']:.4f}")
        
        # Test prediction
        predictions = model.predict(X_val)
        print(f"✅ Predictions made - shape: {predictions.shape}")
        
        # Test feature importance
        importance = model.get_feature_importance()
        print(f"✅ Feature importance: {importance}")
        
        return True
    except Exception as e:
        print(f"❌ Linear model test failed: {e}")
        return False

def test_xgboost_model():
    """Test XGBoost model with dummy data"""
    print("\nTesting XGBoost model...")
    try:
        from app.models.xgboost_model import XGBoostModel
        
        # Create dummy data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y_train = pd.Series(2 * X_train['feature1'] + 3 * X_train['feature2'] + np.random.randn(100) * 0.1)
        
        X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20)
        })
        y_val = pd.Series(2 * X_val['feature1'] + 3 * X_val['feature2'] + np.random.randn(20) * 0.1)
        
        # Test XGBoost with simple hyperparameters
        model = XGBoostModel()
        hyperparameters = {
            'n_estimators': 10,  # Small for fast testing
            'max_depth': 3,
            'learning_rate': 0.1
        }
        metrics = model.train(X_train, y_train, X_val, y_val, hyperparameters)
        print(f"✅ XGBoost model trained - MAE: {metrics['mae']:.4f}")
        
        # Test prediction
        predictions = model.predict(X_val)
        print(f"✅ Predictions made - shape: {predictions.shape}")
        
        # Test feature importance
        importance = model.get_feature_importance()
        print(f"✅ Feature importance: {importance}")
        
        return True
    except Exception as e:
        print(f"❌ XGBoost model test failed: {e}")
        return False

def test_data_fetcher():
    """Test data fetcher (requires API key)"""
    print("\nTesting data fetcher...")
    try:
        from app.data.fetcher import FinancialDataFetcher
        from app.config import settings
        
        # Check if API key is available
        if not settings.financial_data_api_key or settings.financial_data_api_key == "your_api_key_here":
            print("⚠️  Skipping data fetcher test - no API key provided")
            return True
        
        fetcher = FinancialDataFetcher()
        print("✅ Data fetcher initialized")
        
        # Note: Uncomment below to test actual API call (uses your API quota)
        # df = fetcher.fetch_daily_prices("AAPL", "2024-01-01", "2024-01-05")
        # print(f"✅ Data fetched - shape: {df.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("TESTING STOCK PREDICTION API COMPONENTS")
    print("=" * 50)
    
    tests = [
        test_config,
        test_exceptions,
        test_logger,
        test_schemas,
        test_base_model,
        test_linear_model,
        test_xgboost_model,
        test_data_fetcher
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed)*100:.1f}%")

if __name__ == "__main__":
    main()