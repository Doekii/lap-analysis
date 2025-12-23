import duckdb
import pandas as pd
import numpy as np
import os
from typing import List, Optional
import time
import json

class TelemetryService:
    def __init__(self, db_path: str = "data/"):
        # Allow overriding via env var, default to relative path
        self.db_path = os.getenv("DB_PATH", db_path)

    def _get_connection(self, filename: Optional[str] = None):
        """Creates a read-only connection to DuckDB."""
        # Determine which file to open: the specific filename (if provided) or the default
        if filename:
            target_path = os.path.join("data", filename)
        else:
            target_path = self.db_path

        # Robust path handling: check current dir, then backend dir
        if not os.path.exists(target_path):
             # If running from project root, try looking inside backend/data
             alt_path = os.path.join("backend", target_path)
             if os.path.exists(alt_path):
                 return duckdb.connect(alt_path, read_only=True)
             
             # Fallback: try absolute path based on this file's location
             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             abs_path = os.path.join(base_dir, "..", target_path)
             if os.path.exists(abs_path):
                 return duckdb.connect(abs_path, read_only=True)
             
             # Attempt to find it if it was passed without "data/" prefix
             if filename and not filename.startswith("data"):
                 retry_path = os.path.join("data", filename)
                 if os.path.exists(retry_path):
                     return duckdb.connect(retry_path, read_only=True)

        return duckdb.connect(target_path, read_only=True)

    def list_database_files(self) -> List[str]:
        """Scans the data directory for .duckdb files."""
        target_dir = os.path.dirname(self.db_path)
        if not target_dir: target_dir = "data"
            
        if not os.path.exists(target_dir):
             if os.path.exists(os.path.join("backend", target_dir)):
                 target_dir = os.path.join("backend", target_dir)
             elif os.path.exists(os.path.join("..", target_dir)):
                 target_dir = os.path.join("..", target_dir)
        
        if not os.path.exists(target_dir):
            return []

        return [f for f in os.listdir(target_dir) if f.endswith(".duckdb")]

    def get_lap_list(self, filename: Optional[str] = None) -> List[dict]:
        """Returns a list of all available laps in the database."""
        con = self._get_connection(filename)
        try:
            query = "SELECT DISTINCT value as lap_number FROM Lap ORDER BY value"
            df = con.execute(query).df()
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error fetching lap list: {e}")
            return []
        finally:
            con.close()

    def get_lap_telemetry(self, lap_number: int, channels: List[str], filename: Optional[str] = None) -> List[dict]:
        """
        Fetches telemetry for a specific lap. Optimized for speed and low overhead.
        """
        con = self._get_connection(filename)
        try:
            # 1. Quick Validation: Get all table names once
            # This avoids querying information_schema inside the loop for every channel
            existing_tables_df = con.execute("SHOW TABLES").df()
            existing_tables = set(existing_tables_df['name'].tolist())

            # Prepare channel list
            # We enforce 'Lap Dist' as the first channel
            target_channels = ['Lap Dist'] + [c for c in channels if c != 'Lap Dist']
            # Filter out requested channels that don't actually exist in the DB
            valid_channels = [c for c in target_channels if c in existing_tables]

            # 2. Fetch Session Globals (Start, Duration)
            # Fetch as scalar values (faster than df overhead)
            stats_query = """
                SELECT MIN(value), MAX(value)
                FROM "GPS Time"
            """
            stats = con.execute(stats_query).fetchone()
            sess_start, sess_end = stats[0], stats[1]
            sess_duration = sess_end - sess_start

            # 3. Get Lap Time Boundaries
            lap_bounds = con.execute(f"SELECT ts FROM Lap WHERE value IN ({lap_number}, {lap_number+1}) ORDER BY value").fetchall()
            
            if not lap_bounds:
                return [] # Lap not found

            lap_start = lap_bounds[0][0]
            # Handle last lap case
            lap_end = lap_bounds[1][0] if len(lap_bounds) > 1 else sess_end

            # 4. Create Master Skeleton (GPS Time)
            # OPTIMIZATION: Calculate 'Time' (relative) directly in SQL
            master_query = f"""
                SELECT 
                    (value - {lap_start}) as Time,
                    value as ts
                FROM "GPS Time"
                WHERE value >= {lap_start} AND value <= {lap_end}
                ORDER BY value
            """
            final_df = con.execute(master_query).df()

            # 5. Loop Through Channels & Merge
            for channel in valid_channels:
                try:
                    # Get Row Count
                    row_count = con.execute(f'SELECT COUNT(*) FROM "{channel}"').fetchone()[0]
                    if row_count == 0: continue

                    # Calculate Frequency
                    freq = row_count / sess_duration
                    
                    # Calculate offsets based on time
                    start_row_est = int((lap_start - sess_start) * freq)
                    end_row_est   = int((lap_end - sess_start) * freq)
                    
                    # Buffer to ensure we cover the edges
                    buffer = 50
                    offset = max(0, start_row_est - buffer)
                    limit = (end_row_est - start_row_est) + (2 * buffer)
                    
                    # Fetch Data (Only the column we need)
                    data_query = f'SELECT value as "{channel}" FROM "{channel}" LIMIT {limit} OFFSET {offset}'
                    chan_df = con.execute(data_query).df()
                    
                    if chan_df.empty: continue

                    # Reconstruct Time (Absolute) for alignment
                    # Time = Start + (Index / Freq)
                    indices = np.arange(offset, offset + len(chan_df))
                    chan_df['ts'] = sess_start + (indices / freq)
                    
                    # Merge 'nearest' matches sensor data to GPS time
                    final_df = pd.merge_asof(final_df, chan_df, on='ts', direction='nearest', tolerance=0.1)
                    
                except Exception as e:
                    print(f"Skipping channel '{channel}': {e}")
                    final_df[channel] = None

            # 6. Final Cleanup
            if 'Lap Dist' in final_df.columns:
                # OPTIMIZATION: Vectorized check for distance reset
                # Find where distance drops by > 100m
                dist_diff = final_df['Lap Dist'].diff()
                reset_mask = dist_diff < -100
                
                if reset_mask.any():
                    # Get the index of the first reset
                    first_reset_idx = reset_mask.idxmax()
                    final_df = final_df.loc[:first_reset_idx-1]

                # Filter negative distance
                final_df = final_df[final_df['Lap Dist'] >= 0]

            # 7. JSON Compliance (Critical)
            # Replace Infinity with NaN, then NaN with None
            final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            final_df = final_df.where(pd.notnull(final_df), None)
            
            # Return list of dicts (FastAPI handles the JSON serialization)
            return final_df.to_json(orient="records",  double_precision=4)

        finally:
            con.close()
            
    def get_lap_comparison(self, file1: str, lap1: int, file2: str, lap2: int, channels: List[str]) -> str:
        """
        Calculates Time Delta. Returns a JSON string.
        """
        t0 = time.time()
        print(f"[DEBUG] Starting comparison request: File1={file1} L{lap1} vs File2={file2} L{lap2}")

        req_channels = list(set(channels + ['Lap Dist']))
        
        # 1. Fetch JSON strings
        data1_json = self.get_lap_telemetry(lap1, req_channels, filename=file1)
        data2_json = self.get_lap_telemetry(lap2, req_channels, filename=file2)
        
        # 2. Parse JSON strings into Python objects (List of Dicts)
        # This fixes the "DataFrame constructor not properly called" error
        data1 = json.loads(data1_json)
        data2 = json.loads(data2_json)
        
        if not data1 or not data2:
            return "[]"

        # 3. Create DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        cols_to_float = ['Lap Dist', 'Time'] + [c for c in req_channels if c in df1.columns]
        for col in cols_to_float:
            if col in df1.columns: df1[col] = pd.to_numeric(df1[col], errors='coerce')
            if col in df2.columns: df2[col] = pd.to_numeric(df2[col], errors='coerce')

        df1.dropna(subset=['Lap Dist', 'Time'], inplace=True)
        df2.dropna(subset=['Lap Dist', 'Time'], inplace=True)
        
        df1.sort_values('Lap Dist', inplace=True)
        df2.sort_values('Lap Dist', inplace=True)

        # 4. Math Logic
        base_dist = df1['Lap Dist'].values
        result = pd.DataFrame({
            'Lap Dist': base_dist,
            'Time_Ref': df1['Time'].values,
        })

        time2_interp = np.interp(base_dist, df2['Lap Dist'].values, df2['Time'].values, left=np.nan, right=np.nan)
        result['Time_Delta'] = time2_interp - df1['Time'].values

        ignore_cols = ['Lap Dist', 'Time', 'ts', 'GPS Latitude', 'GPS Longitude']
        for col in req_channels:
            if col in ignore_cols: continue
            
            if col in df1.columns and col in df2.columns:
                y_comp = df2[col].ffill().bfill().values
                y_comp_interp = np.interp(base_dist, df2['Lap Dist'].values, y_comp, left=np.nan, right=np.nan)
                
                result[f'{col}_Ref'] = df1[col].values
                result[f'{col}_Comp'] = y_comp_interp
                result[f'{col}_Delta'] = y_comp_interp - df1[col].fillna(0).values

        print(f"[DEBUG] Comparison finished in {time.time() - t0:.4f}s")
        
        # 5. Return JSON string
        return result.to_json(orient="records", double_precision=4)