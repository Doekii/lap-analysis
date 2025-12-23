from fastapi import APIRouter, HTTPException, Query, Response
from app.services import TelemetryService
from typing import List, Optional

router = APIRouter(prefix="/laps", tags=["laps"])
service = TelemetryService()

@router.get("/compare")
def compare_laps(
    file1: str,
    lap1: int,
    file2: str,
    lap2: int,
    channels: Optional[str] = Query("Ground Speed,Throttle Pos,Brake Pos,Gear,Engine RPM", description="Comma-separated list of channels")
):
    """
    Compares two laps. Returns a raw JSON string to avoid float errors.
    """
    channel_list = [c.strip() for c in channels.split(",")]
    
    # Get JSON string from service
    json_data = service.get_lap_comparison(file1, lap1, file2, lap2, channel_list)
    
    if json_data == "[]":
        raise HTTPException(status_code=404, detail="Could not compare laps.")
    
    # Return raw Response so FastAPI doesn't double-encode it
    return Response(content=json_data, media_type="application/json")

@router.get("/{filename}")
def get_laps(filename: str):
    """
    List all available laps. Returns standard dict (no float issues here).
    """
    data = service.get_lap_list(filename=filename)
    if not data:
         raise HTTPException(status_code=404, detail=f"No laps found or file '{filename}' does not exist.")
    return data

@router.get("/{filename}/{lap_number}")
def get_lap_telemetry(
    filename: str,
    lap_number: int, 
    channels: Optional[str] = Query("Ground Speed,Throttle Pos,Brake Pos,Gear,Engine RPM", description="Comma-separated list of channels")
):
    """
    Get full telemetry. Returns a raw JSON string to avoid float errors.
    """
    channel_list = [c.strip() for c in channels.split(",")]
    
    # Get JSON string from service
    json_data = service.get_lap_telemetry(lap_number, channel_list, filename=filename)
    
    if json_data == "[]":
        raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found in {filename}")
        
    # Return raw Response
    return Response(content=json_data, media_type="application/json")