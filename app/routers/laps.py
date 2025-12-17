from fastapi import APIRouter, HTTPException, Query
from app.services import TelemetryService
from typing import List, Optional

router = APIRouter(prefix="/laps", tags=["laps"])
service = TelemetryService(db_path="data/")

@router.get("/{filename}")
def get_laps(filename: str):
    """
    List all available laps in a specific DuckDB file.
    """
    data = service.get_lap_list(filename=filename)
    if not data:
         raise HTTPException(status_code=404, detail=f"No laps found or file '{filename}' does not exist.")
    return data

@router.get("/{filename}/{lap_number}")
def get_lap_telemetry(
    filename: str,
    lap_number: int, 
    channels: Optional[str] = Query("GPS Longitude, GPS Latitude, Ground Speed,Throttle Pos,Brake Pos,Gear,Engine RPM", description="Comma-separated list of channels")
):
    """
    Get full telemetry for a lap from a specific DuckDB file.
    """
    channel_list = [c.strip() for c in channels.split(",")]
    
    # We pass the filename to the service method
    data = service.get_lap_telemetry(filename, lap_number, channel_list)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found in {filename}")
        
    return data