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
        if filename:
            target_path = os.path.join("data", filename)
        else:
            target_path = self.db_path

        if not os.path.exists(target_path):
             alt_path = os.path.join("backend", target_path)
             if os.path.exists(alt_path):
                 return duckdb.connect(alt_path, read_only=True)
             
             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             abs_path = os.path.join(base_dir, "..", target_path)
             if os.path.exists(abs_path):
                 return duckdb.connect(abs_path, read_only=True)
             
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
        Fetches telemetry for a specific lap. Optimized for speed and handles multi-column tables.
        """
        con = self._get_connection(filename)
        try:
            # 1. Quick Validation: Get all table names once
            existing_tables_df = con.execute("SHOW TABLES").df()
            existing_tables = set(existing_tables_df['name'].tolist())

            target_channels = ['Lap Dist', 'GPS Latitude', 'GPS Longitude'] + [c for c in channels if c not in ['Lap Dist', 'GPS Latitude', 'GPS Longitude']]
            valid_channels = [c for c in target_channels if c in existing_tables]

            # 2. Fetch Session Globals
            stats_query = 'SELECT MIN(value), MAX(value) FROM "GPS Time"'
            stats = con.execute(stats_query).fetchone()
            sess_start, sess_end = stats[0], stats[1]
            sess_duration = sess_end - sess_start

            # 3. Get Lap Time Boundaries
            lap_bounds = con.execute(f"SELECT ts FROM Lap WHERE value IN ({lap_number}, {lap_number+1}) ORDER BY value").fetchall()
            
            if not lap_bounds:
                return "[]"

            lap_start = lap_bounds[0][0]
            lap_end = lap_bounds[1][0] if len(lap_bounds) > 1 else sess_end

            # 4. Create Master Skeleton
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
                    # Get columns for this table
                    table_info = con.execute(f"PRAGMA table_info('{channel}')").df()
                    all_cols = set(table_info['name'].tolist())
                    data_cols = [c for c in table_info['name'].tolist() if c != 'ts']
                    
                    if not data_cols: continue

                    row_count = con.execute(f'SELECT COUNT(*) FROM "{channel}"').fetchone()[0]
                    if row_count == 0: continue

                    freq = row_count / sess_duration
                    start_row_est = int((lap_start - sess_start) * freq)
                    end_row_est   = int((lap_end - sess_start) * freq)
                    
                    buffer = 50
                    offset = max(0, start_row_est - buffer)
                    limit = (end_row_est - start_row_est) + (2 * buffer)
                    
                    # Construct query: Only select 'ts' if it actually exists in this table
                    select_parts = []
                    if 'ts' in all_cols:
                        select_parts.append('ts')
                    select_parts.extend([f'"{c}"' for c in data_cols])
                    
                    col_select = ", ".join(select_parts)
                    data_query = f'SELECT {col_select} FROM "{channel}" LIMIT {limit} OFFSET {offset}'
                    chan_df = con.execute(data_query).df()
                    
                    if chan_df.empty: continue

                    # Reconstruct Time (Absolute) for alignment if 'ts' was missing from table
                    if 'ts' not in all_cols:
                        indices = np.arange(offset, offset + len(chan_df))
                        chan_df['ts'] = sess_start + (indices / freq)

                    # Handle multi-column structure (e.g., Tyres Wear)
                    if len(data_cols) > 1:
                        chan_df[channel] = chan_df[data_cols].to_dict(orient='records')
                        chan_df = chan_df[['ts', channel]]
                    else:
                        chan_df.rename(columns={data_cols[0]: channel}, inplace=True)
                    
                    # Merge 'nearest' matches sensor data to GPS time indices
                    final_df = pd.merge_asof(
                        final_df.sort_values('ts'), 
                        chan_df.sort_values('ts'), 
                        on='ts', 
                        direction='nearest', 
                        tolerance=0.1
                    )
                    
                except Exception as e:
                    print(f"Skipping channel '{channel}': {e}")
                    # Use NaN instead of None to prevent TypeError in cleanup math
                    final_df[channel] = np.nan

            # 6. Final Cleanup
            if 'Lap Dist' in final_df.columns:
                # Ensure it's numeric before diffing to avoid TypeError
                final_df['Lap Dist'] = pd.to_numeric(final_df['Lap Dist'], errors='coerce')
                
                # Check if we actually have data to work with
                if not final_df['Lap Dist'].isnull().all():
                    dist_diff = final_df['Lap Dist'].diff()
                    reset_mask = dist_diff < -100
                    if reset_mask.any():
                        first_reset_idx = reset_mask.idxmax()
                        final_df = final_df.loc[:first_reset_idx-1]
                    
                    final_df = final_df[final_df['Lap Dist'] >= 0]

            # 7. JSON Compliance
            final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            final_df = final_df.where(pd.notnull(final_df), None)
            
            return final_df.to_json(orient="records")

        finally:
            con.close()
            
    def get_lap_comparison(self, file1: str, lap1: int, file2: str, lap2: int, channels: List[str]) -> str:
        """
        Calculates Time Delta. Returns a JSON string. Handles complex data types (lists/dicts).
        """
        req_channels = list(set(channels + ['Lap Dist']))
        
        data1_json = self.get_lap_telemetry(lap1, req_channels, filename=file1)
        data2_json = self.get_lap_telemetry(lap2, req_channels, filename=file2)
        
        data1 = json.loads(data1_json)
        data2 = json.loads(data2_json)
        
        if not data1 or not data2:
            return "[]"

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # Standardize distance and time
        for df in [df1, df2]:
            if 'Lap Dist' not in df.columns or 'Time' not in df.columns:
                continue
            df['Lap Dist'] = pd.to_numeric(df['Lap Dist'], errors='coerce')
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df.dropna(subset=['Lap Dist', 'Time'], inplace=True)
            # CRITICAL: Sort by distance to ensure interpolation works linearly
            df.sort_values('Lap Dist', inplace=True) 

        if df1.empty or df2.empty:
            return "[]"

        base_dist = df1['Lap Dist'].values
        result = pd.DataFrame({
            'Lap Dist': base_dist,
            'Time_Ref': df1['Time'].values,
        })

        # Interpolate Time Delta
        time2_interp = np.interp(base_dist, df2['Lap Dist'].values, df2['Time'].values, left=np.nan, right=np.nan)
        
        # SMOOTHING STEP: Remove high-frequency math noise from Time Delta
        # A window of 5 points is roughly 0.1s - invisible to the eye but kills jitter
        raw_delta = time2_interp - df1['Time'].values
        result['Time_Delta'] = pd.Series(raw_delta).rolling(window=5, center=True, min_periods=1).mean().values

        ignore_cols = ['Lap Dist', 'Time', 'ts', 'GPS Latitude', 'GPS Longitude']
        for col in req_channels:
            if col in ignore_cols or col not in df1.columns or col not in df2.columns: 
                continue
            
            # Check if data is scalar or complex (dict/list)
            first_val = df1[col].iloc[0] if not df1[col].empty else None
            is_scalar = not isinstance(first_val, (dict, list))
            
            if is_scalar:
                # Standard interpolation for numeric values
                y_ref = pd.to_numeric(df1[col], errors='coerce').fillna(0).values
                y_comp = pd.to_numeric(df2[col], errors='coerce').ffill().bfill().values
                y_comp_interp = np.interp(base_dist, df2['Lap Dist'].values, y_comp, left=np.nan, right=np.nan)
                
                result[f'{col}_Ref'] = y_ref
                result[f'{col}_Comp'] = y_comp_interp
                result[f'{col}_Delta'] = y_comp_interp - y_ref
            else:
                # For complex data, use merge_asof logic to get the 'nearest' complex object
                result[f'{col}_Ref'] = df1[col].values
                temp_comp = pd.merge_asof(
                    result[['Lap Dist']], 
                    df2[['Lap Dist', col]].rename(columns={col: f'{col}_Comp'}), 
                    on='Lap Dist', 
                    direction='nearest'
                )
                result[f'{col}_Comp'] = temp_comp[f'{col}_Comp'].values

        # Remove 'double_precision' to preserve full float accuracy
        return result.to_json(orient="records")