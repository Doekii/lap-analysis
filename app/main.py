from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Explicitly import the router variable to avoid module attribute errors
from app.routers.laps import router as laps_router
from app.routers.sessions import router as sessions_router

app = FastAPI(
    title="Race Telemetry API",
    description="Backend for serving DuckDB race telemetry to React Dashboard",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173", # Vite default
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(laps_router)
app.include_router(sessions_router)

@app.get("/")
def root():
    return {"message": "Telemetry API is running. Go to /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    # Run with: python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)