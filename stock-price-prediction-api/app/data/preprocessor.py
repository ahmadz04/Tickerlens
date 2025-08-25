import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app.utils.logger import logger

class StockDataPreprocessor:
    """Handles feature engineering for stock price prediction"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from price data (leak-free)"""
        logger.info("Creating price-based features...")
        df = df.sort_values(['ticker', 'time']).copy()
        
        # Lag features (t-1) - yesterday's data
        df['yesterday_close'] = df.groupby('ticker')['close'].shift(1)
        df['yesterday_vol'] = df.groupby('ticker')['volume'].shift(1)
        df['yesterday_high'] = df.groupby('ticker')['high'].shift(1)
        df['yesterday_low'] = df.groupby('ticker')['low'].shift(1)
        df['yesterday_open'] = df.groupby('ticker')['open'].shift(1)
        
        # Rolling averages (using past data only)
        df['sma_5'] = df.groupby('ticker')['close'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['sma_10'] = df.groupby('ticker')['close'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['sma_20'] = df.groupby('ticker')['close'].rolling(20, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        
        # Volume moving averages
        df['vol_sma_5'] = df.groupby('ticker')['volume'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        df['vol_sma_10'] = df.groupby('ticker')['volume'].rolling(10, min_periods=1).mean().shift(1).reset_index(0, drop=True)
        
        # Price ranges and ratios (using yesterday's data)
        df['high_low_ratio'] = df['yesterday_high'] / df['yesterday_low']
        df['close_open_ratio'] = df['yesterday_close'] / df['yesterday_open']
        
        # Returns (using historical data only)
        df['prev_close_2'] = df.groupby('ticker')['close'].shift(2)
        df['prev_close_5'] = df.groupby('ticker')['close'].shift(5)
        df['ret_1d'] = (df['yesterday_close'] / df['prev_close_2'] - 1).fillna(0)
        df['ret_5d'] = (df['yesterday_close'] / df['prev_close_5'] - 1).fillna(0)
        
        # Volatility (using historical data)
        df['volatility_5d'] = df.groupby('ticker')['ret_1d'].rolling(5, min_periods=1).std().shift(1).reset_index(0, drop=True)
        df['volatility_10d'] = df.groupby('ticker')['ret_1d'].rolling(10, min_periods=1).std().shift(1).reset_index(0, drop=True)
        
        # Target variable (what we want to predict)
        df['target_close'] = df['close']
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['ticker', 'time', 'open', 'high', 'low', 'close', 'volume']])} features")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude metadata and target)"""
        exclude_cols = [
            'ticker', 'time', 'time_milliseconds', 'close', 'target_close', 
            'prev_close_2', 'prev_close_5'  # These are intermediate calculations
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def split_by_date(self, df: pd.DataFrame, train_start: str, train_end: str,
                     val_start: str, val_end: str, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by date ranges"""
        
        def mask_by_date(data, start_date, end_date):
            mask = (data['time'] >= start_date) & (data['time'] <= end_date)
            return data[mask].copy()
        
        train_df = mask_by_date(df, train_start, train_end)
        val_df = mask_by_date(df, val_start, val_end)
        test_df = mask_by_date(df, test_start, test_end)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and remove rows with insufficient lookback data"""
        initial_count = len(df)
        
        # Get feature columns to check for NaN values
        feature_cols = self.get_feature_columns(df)
        
        # Remove rows where we don't have enough historical data for features
        # Also remove any rows with NaN values in feature columns
        df_clean = df.dropna(subset=feature_cols + ['target_close']).copy()
        
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        
        logger.info(f"Removed {removed_count} rows due to insufficient historical data or NaN values")
        logger.info(f"Final dataset shape: {df_clean.shape}")
        
        return df_clean