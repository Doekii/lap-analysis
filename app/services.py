import duckdb
import pandas as pd
import numpy as np
import os
from typing import List, Optional

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

    def get_lap_telemetry(self, filename, lap_number, channels=['Ground Speed', 'Throttle Pos']):
        """
        Reconstructs lap data.
        Returns DataFrame with: [Time (relative), Lap Dist, ts (absolute), ...channels...]
        """
        con = self._get_connection(filename)

        # ---------------------------------------------------------
        # 1. Setup & Session Stats
        # ---------------------------------------------------------
        # Ensure 'Lap Dist' is in our fetch list, but not duplicated
        target_channels = [c for c in channels if c != 'Lap Dist']
        target_channels.insert(0, 'Lap Dist')

        stats_query = """
            SELECT 
                MIN(value) as start_time,
                MAX(value) as end_time,
                MAX(value) - MIN(value) as duration
            FROM "GPS Time"
        """
        session_stats = con.execute(stats_query).df().iloc[0]
        
        sess_start = session_stats['start_time']
        sess_duration = session_stats['duration']
        
        # ---------------------------------------------------------
        # 2. Get Lap Boundaries & Master Skeleton
        # ---------------------------------------------------------
        lap_times = con.execute(f"SELECT ts FROM Lap WHERE value IN ({lap_number}, {lap_number+1}) ORDER BY value").df()
        
        if lap_times.empty:
            return None

        lap_start = lap_times.iloc[0]['ts']
        lap_end = lap_times.iloc[1]['ts'] if len(lap_times) > 1 else session_stats['end_time']

        # Master Skeleton based on GPS Time
        master_query = f"""
            SELECT value as ts
            FROM "GPS Time"
            WHERE value >= {lap_start} AND value <= {lap_end}
            ORDER BY value
        """
        final_df = con.execute(master_query).df()
        
        # ---------------------------------------------------------
        # 3. Add Relative Time (NEW STEP)
        # ---------------------------------------------------------
        # Simple subtraction: Current Absolute Time - Lap Start Time
        final_df['Time'] = final_df['ts'] - lap_start
        
        # Reorder columns so 'Time' is first
        cols = ['Time', 'ts']
        final_df = final_df[cols]

        # ---------------------------------------------------------
        # 4. Loop Through Channels & Merge
        # ---------------------------------------------------------
        for channel in target_channels:
            try:
                # Get Row Count
                row_count_res = con.execute(f'SELECT COUNT(*) FROM "{channel}"').fetchone()
                if not row_count_res: continue
                row_count = row_count_res[0]
                
                # Calculate Frequency & Indices
                freq = row_count / sess_duration
                start_row_est = int((lap_start - sess_start) * freq)
                end_row_est   = int((lap_end - sess_start) * freq)
                
                buffer = 50
                offset = max(0, start_row_est - buffer)
                limit = (end_row_est - start_row_est) + (buffer * 2)
                
                # Fetch Data
                data_query = f'SELECT value as "{channel}" FROM "{channel}" LIMIT {limit} OFFSET {offset}'
                chan_df = con.execute(data_query).df()
                
                # Reconstruct Time (Absolute)
                global_indices = np.arange(offset, offset + len(chan_df))
                chan_df['ts'] = sess_start + (global_indices / freq)
                
                # Merge
                final_df = pd.merge_asof(final_df, chan_df, on='ts', direction='nearest', tolerance=0.1)
                
            except Exception as e:
                print(f"Error processing channel '{channel}': {e}")
                final_df[channel] = None

        # ---------------------------------------------------------
        # 5. Final Cleanup
        # ---------------------------------------------------------
        if 'Lap Dist' in final_df.columns:
            # 1. Drop the data where the distance sensor reset (Lap N -> Lap N+1 transition)
            dist_diff = final_df['Lap Dist'].diff()
            reset_indices = dist_diff[dist_diff < -100].index
            
            if not reset_indices.empty:
                cutoff_index = reset_indices[0]
                final_df = final_df.loc[:cutoff_index-1]

            # 2. Filter out negative distance (pre-start line jitter)
            final_df = final_df[final_df['Lap Dist'] >= 0]
            #final_df = final_df.iloc[:12000]
        return final_df.to_json(orient="records")