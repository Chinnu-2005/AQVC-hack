from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from model.schemas import DataResponse
from service.data_service import DataService
from util.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/latest", response_model=DataResponse)
async def get_latest_data(days: int = Query(default=30, ge=1, le=365)):
    """Get latest FTSE 100 data"""
    try:
        data = DataService.get_latest_data(days=days)
        
        # Convert to dictionary for JSON response
        data_dict = data.reset_index().to_dict('records')
        
        return DataResponse(
            data=data_dict,
            count=len(data_dict),
            period=f"Last {days} days"
        )
        
    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical", response_model=DataResponse)
async def get_historical_data(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """Get historical FTSE 100 data"""
    try:
        # Validate date formats
        if not (DataService.validate_date_format(start_date) and 
                DataService.validate_date_format(end_date)):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        data = DataService.fetch_ftse_data(start_date, end_date)
        
        # Convert to dictionary for JSON response
        data_dict = data.reset_index().to_dict('records')
        
        return DataResponse(
            data=data_dict,
            count=len(data_dict),
            period=f"{start_date} to {end_date}"
        )
        
    except Exception as e:
        logger.error(f"Historical data fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
