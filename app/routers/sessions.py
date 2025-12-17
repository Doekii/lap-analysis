from fastapi import APIRouter
from app.services import TelemetryService

router = APIRouter(prefix="/sessions", tags=["sessions"])
service = TelemetryService() 

@router.get("/")
def get_available_sessions():
    """
    Returns a list of all available DuckDB database files 
    in the data directory.
    """
    return {
        "files": service.list_database_files()
    }