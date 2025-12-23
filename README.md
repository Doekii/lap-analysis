# Race Telemetry Analysis Backend

A high-performance FastAPI backend for analyzing racing telemetry data stored in DuckDB. This service handles data extraction, resampling, channel alignment, and lap comparison logic.

## 🚀 Features

* **DuckDB Integration:** Directly queries efficient `.duckdb` files.
* **Automatic Alignment:** Resamples sensors with different frequencies (e.g., 10Hz GPS vs 50Hz Speed) onto a unified time/distance grid.
* **Lap Comparison:** Calculates real-time deltas (Time Delta, Speed Delta) between any two laps, even across different sessions.
* **JSON Sanitization:** Robustly handles `NaN` and `Infinity` values common in telemetry math.

## 🛠️ Setup & Installation

1.  **Prerequisites**
    * Python 3.9+
    * Telemetry data files (`.duckdb`)

2.  **Installation**
    Navigate to the backend directory and install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    
3.  **Data Placement**
    Place your `.duckdb` telemetry files in the `backend/data/` directory.

## ⚡ Running the Server

Start the development server using Uvicorn:

```bash
# Run from the 'backend' folder
uvicorn app.main:app --reload
```
The API will be available at: **http://127.0.0.1:8000**

## 📚 API Documentation

Once the server is running, visit **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for the interactive Swagger UI.

### Key Endpoints

#### 1. List Sessions
Get a list of available `.duckdb` files in the data folder.
* `GET /sessions/`

#### 2. List Laps
Get all valid laps for a specific session file.
* `GET /laps/{filename}`
* **Example:** `/laps/session_fp1.duckdb`

#### 3. Get Lap Telemetry
Fetch full telemetry for a specific lap. Returns aligned data points including calculated Relative Time.
* `GET /laps/{filename}/{lap_number}`
* **Query Params:** `channels` (comma-separated list, e.g., `Ground Speed,Throttle Pos,Brake Pos`)

#### 4. Compare Laps
Calculate the delta between two laps (Reference vs. Comparison).
* `GET /laps/compare`
* **Query Params:**
    * `file1`, `lap1` (Reference Lap)
    * `file2`, `lap2` (Comparison Lap)
    * `channels` (Channels to compare)

## 📂 Project Structure

```text
backend/
├── data/                 # Place .duckdb files here
├── app/
│   ├── main.py           # Application entry point & CORS config
│   ├── services.py       # Core logic: DuckDB queries, interpolation, math
│   ├── models.py         # Pydantic data models
│   └── routers/          # API route definitions
│       ├── laps.py       # Lap & Comparison endpoints
│       └── sessions.py   # File listing endpoints
└── requirements.txt      # Python dependencies
```
