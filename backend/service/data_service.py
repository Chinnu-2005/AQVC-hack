import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

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
            data = ftse.history(start=start_date, end=end_date)
            
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
    def validate_date_format(date_string: str) -> bool:
        """Validate date format YYYY-MM-DD"""
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False