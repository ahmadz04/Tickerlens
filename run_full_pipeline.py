#!/usr/bin/env python3
"""
Complete stock prediction pipeline - mimics the Google Colab notebook
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stock-price-prediction-api'))

from app.config import settings
from app.data.fetcher import FinancialDataFetcher
from app.data.preprocessor import StockDataPreprocessor
from app.models.linear_model import LinearModel
from app.models.xgboost_model import XGBoostModel
from app.utils.logger import logger
from app.utils.exceptions import DataFetchError

def main():
    print("=" * 60)
    print("STOCK PRICE PREDICTION - FULL PIPELINE")
    print("=" * 60)
    
    # Check API key
    if not settings.financial_data_api_key or settings.financial_data_api_key == "your_api_key_here":
        print("❌ Please set your FINANCIAL_DATA_API_KEY in the .env file")
        print("You can get an API key from: https://financialdatasets.ai/")
        return
    
    # Configuration
    tickers = ["AAPL", "GOOGL", "MSFT"]  # Start with 3 tickers to save API quota
    start_date = "2020-01-01"  # Shorter period to save API calls
    end_date = "2024-12-31"
    
    train_start, train_end = "2020-01-01", "2022-12-31"
    val_start, val_end = "2023-01-01", "2023-12-31"
    test_start, test_end = "2024-01-01", "2024-12-31"
    
    try:
        # Step 1: Fetch Data
        print(f"\n📊 STEP 1: Fetching data for {len(tickers)} tickers...")
        fetcher = FinancialDataFetcher()
        
        all_data = []
        for ticker in tickers:
            print(f"Fetching {ticker}...")
            try:
                df = fetcher.fetch_daily_prices(ticker, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
                    print(f"✅ {ticker}: {len(df)} records")
                else:
                    print(f"⚠️  {ticker}: No data returned")
            except DataFetchError as e:
                print(f"❌ {ticker}: {e}")
                continue
        
        if not all_data:
            print("❌ No data fetched for any ticker. Check your API key and connection.")
            return
        
        # Combine all data
        raw_df = pd.concat(all_data, ignore_index=True)
        print(f"📈 Combined data shape: {raw_df.shape}")
        print(f"📅 Date range: {raw_df['time'].min()} to {raw_df['time'].max()}")
        
        # Step 2: Feature Engineering
        print(f"\n🔧 STEP 2: Creating features...")
        preprocessor = StockDataPreprocessor()
        feature_df = preprocessor.create_price_features(raw_df)
        
        # Clean data
        feature_df = preprocessor.clean_data(feature_df)
        
        # Get feature columns
        feature_cols = preprocessor.get_feature_columns(feature_df)
        print(f"📊 Created {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        
        # Step 3: Split Data
        print(f"\n📊 STEP 3: Splitting data...")
        train_df, val_df, test_df = preprocessor.split_by_date(
            feature_df, train_start, train_end, val_start, val_end, test_start, test_end
        )
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("❌ One or more data splits is empty. Check your date ranges.")
            return
        
        # Prepare features and targets
        X_train, y_train = train_df[feature_cols], train_df['target_close']
        X_val, y_val = val_df[feature_cols], val_df['target_close']
        X_test, y_test = test_df[feature_cols], test_df['target_close']
        
        print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Step 4: Train Models
        print(f"\n🤖 STEP 4: Training models...")
        
        models = {}
        results = {}
        
        # Baseline model (yesterday's close predicts today's close)
        baseline_pred = val_df['yesterday_close'].values
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        baseline_mae = mean_absolute_error(y_val, baseline_pred)
        print(f"📊 Baseline MAE: ${baseline_mae:.4f}")
        
        # Train Linear Regression
        print("Training Linear Regression...")
        linear_model = LinearModel("linear")
        linear_metrics = linear_model.train(X_train, y_train, X_val, y_val)
        models['Linear'] = linear_model
        results['Linear'] = linear_metrics
        print(f"✅ Linear Regression - MAE: ${linear_metrics['mae']:.4f}")
        
        # Train Ridge Regression
        print("Training Ridge Regression...")
        ridge_model = LinearModel("ridge")
        ridge_metrics = ridge_model.train(X_train, y_train, X_val, y_val)
        models['Ridge'] = ridge_model
        results['Ridge'] = ridge_metrics
        print(f"✅ Ridge Regression - MAE: ${ridge_metrics['mae']:.4f}")
        
        # Train XGBoost (with reduced parameters for speed)
        print("Training XGBoost...")
        xgb_model = XGBoostModel()
        xgb_hyperparameters = {
            'n_estimators': 100,  # Reduced for faster training
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val, xgb_hyperparameters)
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_metrics
        print(f"✅ XGBoost - MAE: ${xgb_metrics['mae']:.4f}")
        
        # Step 5: Select Best Model and Final Evaluation
        print(f"\n🏆 STEP 5: Model evaluation...")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        best_model = models[best_model_name]
        print(f"🥇 Best model: {best_model_name} (MAE: ${results[best_model_name]['mae']:.4f})")
        
        # Retrain on train + validation data
        print("Retraining best model on combined train+validation data...")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        if best_model_name == 'XGBoost':
            final_model = XGBoostModel()
            final_model.train(X_train_val, y_train_val, X_test, y_test, xgb_hyperparameters)
        else:
            final_model = LinearModel(best_model_name.lower())
            final_model.train(X_train_val, y_train_val, X_test, y_test)
        
        # Final test predictions
        test_predictions = final_model.predict(X_test)
        
        # Calculate test metrics
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        
        # Baseline on test set
        baseline_test_pred = test_df['yesterday_close'].values
        baseline_test_mae = mean_absolute_error(y_test, baseline_test_pred)
        mae_improvement = (baseline_test_mae - test_mae) / baseline_test_mae * 100
        
        # Step 6: Results Summary
        print(f"\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        
        print(f"📊 DATASET SUMMARY:")
        print(f"   Tickers: {', '.join(tickers)}")
        print(f"   Training period: {train_start} to {train_end}")
        print(f"   Test period: {test_start} to {test_end}")
        print(f"   Features used: {len(feature_cols)}")
        print(f"   Total predictions: {len(test_predictions)}")
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   Best model: {best_model_name}")
        print(f"   Test MAE: ${test_mae:.4f}")
        print(f"   Test RMSE: ${test_rmse:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Baseline MAE: ${baseline_test_mae:.4f}")
        print(f"   Improvement: {mae_improvement:.1f}%")
        
        # Per-ticker performance
        print(f"\n📈 PER-TICKER PERFORMANCE:")
        for ticker in tickers:
            ticker_mask = test_df['ticker'] == ticker
            if ticker_mask.sum() > 0:
                ticker_mae = mean_absolute_error(
                    y_test[ticker_mask], 
                    test_predictions[ticker_mask]
                )
                ticker_baseline = mean_absolute_error(
                    y_test[ticker_mask], 
                    baseline_test_pred[ticker_mask]
                )
                ticker_improvement = (ticker_baseline - ticker_mae) / ticker_baseline * 100
                print(f"   {ticker}: MAE ${ticker_mae:.4f} (improvement: {ticker_improvement:.1f}%)")
        
        # Feature importance (if available)
        if hasattr(final_model, 'get_feature_importance'):
            importance = final_model.get_feature_importance()
            if importance:
                print(f"\n🔍 TOP FEATURES:")
                sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for feature, score in sorted_features:
                    print(f"   {feature}: {score:.4f}")
        
        # Save model
        model_path = f"data/models/{best_model_name.lower()}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        final_model.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        print(f"\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()