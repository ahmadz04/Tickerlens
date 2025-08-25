import pandas as pd
import requests
import time
from typing import List, Optional
from datetime import datetime, timedelta
from app.config import settings
from app.utils.logger import logger
from app.utils.exceptions import DataFetchError

class FinancialDataFetcher:
    """Handles fetching data from Financial Datasets API"""
    
    def __init__(self):
        self.base_url = settings.financial_data_base_url
        self.headers = {"X-API-KEY": settings.financial_data_api_key}
        self.rate_limit_delay = 0.5  # seconds between requests
    
    def fetch_daily_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV data for a ticker"""
        try:
            all_data = []
            params = {
                "ticker": ticker,
                "interval": "day",
                "interval_multiplier": 1,
                "start_date": start_date,
                "end_date": end_date
            }
            
            url = f"{self.base_url}/prices"
            
            while url:
                logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
                
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'prices' in data and data['prices']:
                    all_data.extend(data['prices'])
                elif 'results' in data and data['results']:
                    all_data.extend(data['results'])
                else:
                    logger.warning(f"No price data found for {ticker}")
                    break
                
                # Handle pagination
                url = data.get('next_page_url')
                if url:
                    params = {}  # Next page URL is complete
                
                time.sleep(self.rate_limit_delay)
            
            if not all_data:
                raise DataFetchError(f"No data retrieved for {ticker}")
            
            df = pd.DataFrame(all_data)
            df['ticker'] = ticker
            
            # Standardize timestamp column
            timestamp_field = self._find_timestamp_field(df.columns)
            if timestamp_field:
                df['time'] = pd.to_datetime(df[timestamp_field])
                df = df.sort_values(['ticker', 'time']).reset_index(drop=True)
            else:
                raise DataFetchError(f"No timestamp field found for {ticker}")
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except requests.RequestException as e:
            raise DataFetchError(f"Failed to fetch data for {ticker}: {str(e)}")
        except Exception as e:
            raise DataFetchError(f"Unexpected error fetching data for {ticker}: {str(e)}")
    
    def fetch_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for multiple tickers"""
        all_dataframes = []
        
        for ticker in tickers:
            try:
                df = self.fetch_daily_prices(ticker, start_date, end_date)
                if not df.empty:
                    all_dataframes.append(df)
            except DataFetchError as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                continue
        
        if not all_dataframes:
            raise DataFetchError("No data could be fetched for any ticker")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    
    def _find_timestamp_field(self, columns: List[str]) -> Optional[str]:
        """Find the timestamp field in the data"""
        timestamp_fields = ['time', 'timestamp', 'date', 'datetime']
        for field in timestamp_fields:
            if field in columns:
                return field
        return None