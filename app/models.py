from pydantic import BaseModel
from typing import List, Optional, Any

# Placeholder for Pydantic models if strict validation is needed later.
# Currently, the service returns dictionaries directly from Pandas.

class LapInfo(BaseModel):
    lap_number: int

class TelemetryPoint(BaseModel):
    Time: float
    # Other fields are dynamic based on requested channelss