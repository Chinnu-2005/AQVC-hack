import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from util.logger import setup_logger
from util.exceptions import DataFetchException

logger = setup_logger(__name__)

class DataService:
    """Service for fetching and processing market data"""
    
    @staticmethod
    def fetch_ftse_data(start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch FTSE 100 data from Yahoo Finance
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching FTSE 100 data from {start_date} to {end_date}")
            
            # FTSE 100 symbol
            ftse = yf.Ticker("^FTSE")
            data = ftse.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                raise DataFetchException("No data found for the specified date range")
            
            # Clean the data
            data = data.dropna()
            
            logger.info(f"Successfully fetched {len(data)} days of FTSE 100 data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching FTSE data: {str(e)}")
            raise DataFetchException(f"Failed to fetch FTSE data: {str(e)}")
    
    @staticmethod
    def get_latest_data(days: int = 30) -> pd.DataFrame:
        """
        Get latest FTSE 100 data
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame with latest data
        """
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            return DataService.fetch_ftse_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")
            raise DataFetchException(f"Failed to fetch latest data: {str(e)}")

    @staticmethod
    def get_training_date_range() -> tuple:
        """
        Get 3 years of training data ending yesterday (dynamic sliding window)
        
        Returns:
            tuple: (start_date, end_date) as strings
        """
        # End date is yesterday
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        # Start date is 3 years before yesterday
        start_date = (datetime.now() - timedelta(days=3*365 + 1)).strftime("%Y-%m-%d")
        return start_date, end_date

    @staticmethod
    def get_training_date_range_for_date(target_date: str) -> tuple:
        """
        Get 3 years of training data ending the day before the target date
        
        Args:
            target_date: Target date string (YYYY-MM-DD)
            
        Returns:
            tuple: (start_date, end_date) as strings
        """
        try:
            # Parse target date
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            
            # End date is the day before target date
            end_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            # Start date is 3 years before the end date
            start_date = (target_dt - timedelta(days=3*365 + 1)).strftime("%Y-%m-%d")
            
            return start_date, end_date
            
        except ValueError as e:
            raise DataFetchException(f"Invalid date format: {target_date}. Use YYYY-MM-DD format")

    @staticmethod
    def fetch_actual_data_for_date(target_date: str) -> Dict[str, Any]:
        """
        Fetch actual FTSE 100 data for a specific date and its surrounding days
        
        Args:
            target_date: Target date string (YYYY-MM-DD)
            
        Returns:
            dict: Actual data for the target date and surrounding context
        """
        try:
            # Parse target date
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            
            # Get data for 5 days around the target date (2 days before, target day, 2 days after)
            start_date = (target_dt - timedelta(days=2)).strftime("%Y-%m-%d")
            end_date = (target_dt + timedelta(days=2)).strftime("%Y-%m-%d")
            
            # Fetch the data
            data = DataService.fetch_ftse_data(start_date, end_date)
            
            # Find the target date in the data
            target_date_str = target_dt.strftime("%Y-%m-%d")
            target_data = None
            
            for date_str, row in data.iterrows():
                if date_str.strftime("%Y-%m-%d") == target_date_str:
                    target_data = {
                        "date": target_date_str,
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume']),
                        "returns": float((row['Close'] - row['Open']) / row['Open']) if row['Open'] != 0 else 0.0
                    }
                    break
            
            # Get surrounding context
            context_data = []
            for date_str, row in data.iterrows():
                context_data.append({
                    "date": date_str.strftime("%Y-%m-%d"),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "returns": float((row['Close'] - row['Open']) / row['Open']) if row['Open'] != 0 else 0.0
                })
            
            return {
                "target_date": target_date,
                "target_data": target_data,
                "context_data": context_data,
                "data_available": target_data is not None
            }
            
        except Exception as e:
            logger.error(f"Error fetching actual data for date {target_date}: {str(e)}")
            raise DataFetchException(f"Failed to fetch actual data for date {target_date}: {str(e)}")
    
    @staticmethod
    def validate_date_format(date_string: str) -> bool:
        """Validate date format YYYY-MM-DD"""
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False