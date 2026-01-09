import duckdb
import pandas as pd
import numpy as np
import os
import time
import json
from typing import List, Optional

# Suppress FutureWarning for Pandas 2.1+ behavior regarding downcasting
pd.set_option('future.no_silent_downcasting', True)

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

    def _downsample_dataframe(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """
        Helper: Decimates dataframe to roughly match max_points.
        """
        if len(df) <= max_points or max_points <= 0:
            return df
        step = len(df) // max_points
        return df.iloc[::step].copy()

    def get_lap_telemetry(self, lap_number: int, channels: List[str], filename: Optional[str] = None, max_points: int = 0) -> str:
        """
        Fetches telemetry for a specific lap. Optimized for speed and handles multi-column tables.
        """
        con = self._get_connection(filename)
        try:
            print(f"[DEBUG] Fetching Lap {lap_number} from {filename}")
            # 1. Quick Validation
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

            # 4. Create Dynamic Master Skeleton (High Resolution)
            max_freq = 10.0 # Default
            
            for ch in valid_channels:
                try:
                    c_count = con.execute(f'SELECT COUNT(*) FROM "{ch}"').fetchone()[0]
                    c_freq = c_count / sess_duration
                    if c_freq > max_freq:
                        max_freq = c_freq
                except:
                    pass
            
            max_freq = min(max_freq, 500.0)
            
            step_size = 1.0 / max_freq
            timestamps = np.arange(lap_start, lap_end, step_size)
            
            final_df = pd.DataFrame({'ts': timestamps})
            final_df['Time'] = final_df['ts'] - lap_start

            # 5. Loop Through Channels & Merge
            for channel in valid_channels:
                try:
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
                    
                    select_parts = []
                    if 'ts' in all_cols: select_parts.append('ts')
                    select_parts.extend([f'"{c}"' for c in data_cols])
                    col_select = ", ".join(select_parts)
                    
                    data_query = f'SELECT {col_select} FROM "{channel}" LIMIT {limit} OFFSET {offset}'
                    chan_df = con.execute(data_query).df()
                    
                    if chan_df.empty: continue

                    if 'ts' not in all_cols:
                        indices = np.arange(offset, offset + len(chan_df))
                        chan_df['ts'] = sess_start + (indices / freq)

                    if len(data_cols) > 1:
                        chan_df[channel] = chan_df[data_cols].to_dict(orient='records')
                        chan_df = chan_df[['ts', channel]]
                    else:
                        chan_df.rename(columns={data_cols[0]: channel}, inplace=True)
                    
                    if channel == 'Lap Dist':
                        chan_df.sort_values('ts', inplace=True)
                        final_df[channel] = np.interp(final_df['ts'], chan_df['ts'], chan_df[channel], left=np.nan, right=np.nan)
                    else:
                        final_df = pd.merge_asof(
                            final_df.sort_values('ts'), 
                            chan_df.sort_values('ts'), 
                            on='ts', 
                            direction='nearest', 
                            tolerance=0.1
                        )
                    
                except Exception as e:
                    print(f"Skipping channel '{channel}': {e}")
                    final_df[channel] = np.nan

            # 6. Final Cleanup
            if 'Lap Dist' in final_df.columns:
                final_df['Lap Dist'] = pd.to_numeric(final_df['Lap Dist'], errors='coerce')
                
                if not final_df['Lap Dist'].isnull().all():
                    dist_diff = final_df['Lap Dist'].diff()
                    reset_mask = dist_diff < -100
                    if reset_mask.any():
                        reset_indices = dist_diff.index[reset_mask].tolist()
                        for idx in reset_indices:
                            if idx < len(final_df) * 0.1:
                                final_df = final_df.loc[idx:]
                            else:
                                final_df = final_df.loc[:idx-1]
                                break 
                    
                    final_df = final_df[final_df['Lap Dist'] >= 0]

            if max_points > 0 and len(final_df) > max_points:
                step = len(final_df) // max_points
                final_df = final_df.iloc[::step].copy()

            final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            final_df = final_df.where(pd.notnull(final_df), None)
            
            return final_df.to_json(orient="records")

        finally:
            con.close()
            
    def get_lap_comparison(self, file1: str, lap1: int, file2: str, lap2: int, channels: List[str], max_points: int = 0) -> str:
        """
        Calculates Time Delta by ALIGNING laps on a common integer distance grid.
        Uses OUTER merge to preserve data.
        """
        print(f"[DEBUG] Compare Request: {file1} Lap {lap1} vs {file2} Lap {lap2}")
        req_channels = list(set(channels + ['Lap Dist']))
        
        # 1. Fetch Full Data (Resolution handled by Grid later)
        data1_json = self.get_lap_telemetry(lap1, req_channels, filename=file1, max_points=0)
        data2_json = self.get_lap_telemetry(lap2, req_channels, filename=file2, max_points=0)
        
        data1 = json.loads(data1_json)
        data2 = json.loads(data2_json)
        
        if not data1 or not data2:
            return "[]"

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # 2. Pre-process: Standardize
        for df in [df1, df2]:
            if 'Lap Dist' not in df.columns or 'Time' not in df.columns: return "[]"
            df['Lap Dist'] = pd.to_numeric(df['Lap Dist'], errors='coerce')
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df.dropna(subset=['Lap Dist', 'Time'], inplace=True)
            
            # Normalize to 0 to align lap starts
            df['Lap Dist'] -= df['Lap Dist'].min()
            
            # Round to 1-meter resolution for binning
            df['Lap Dist'] = df['Lap Dist'].round(0).astype(int)
            
            # Deduplicate (Keep first occurrence per meter)
            df.sort_values(by=['Lap Dist', 'Time'], inplace=True)
            df.drop_duplicates(subset='Lap Dist', keep='first', inplace=True)
            
            # Fill gaps for discrete data to ensure continuity for comparison
            # Infer objects to handle potential future warnings
            df.ffill(inplace=True)
            df.bfill(inplace=True)

        if df1.empty or df2.empty: return "[]"

        # 3. Outer Merge (Safe Join)
        merged = pd.merge(
            df1.add_suffix('_Ref'), 
            df2.add_suffix('_Comp'), 
            left_on='Lap Dist_Ref', 
            right_on='Lap Dist_Comp', 
            how='outer' 
        )
        
        # Coalesce 'Lap Dist' (Combine the two columns)
        merged['Lap Dist'] = merged['Lap Dist_Ref'].combine_first(merged['Lap Dist_Comp'])
        
        # Clean up merge columns
        merged.drop(columns=['Lap Dist_Ref', 'Lap Dist_Comp'], inplace=True)
        merged.sort_values('Lap Dist', inplace=True)

        # 4. Calculate Deltas
        merged['Time_Delta'] = merged['Time_Comp'] - merged['Time_Ref']

        ignore_cols = ['Lap Dist', 'Time', 'ts', 'GPS Latitude', 'GPS Longitude']
        
        for col in req_channels:
            if col in ignore_cols: continue
            
            ref_col = f"{col}_Ref"
            comp_col = f"{col}_Comp"
            delta_col = f"{col}_Delta"
            
            # FIX: Robust check for numeric types before subtraction
            if ref_col in merged.columns and comp_col in merged.columns:
                # Force conversion to numeric, coercing errors to NaN
                s_ref = pd.to_numeric(merged[ref_col], errors='coerce').astype(float)
                s_comp = pd.to_numeric(merged[comp_col], errors='coerce').astype(float)
                
                if not s_ref.isna().all() and not s_comp.isna().all():
                     merged[delta_col] = s_comp - s_ref
                else:
                     merged[delta_col] = None

        # Downsampling
        if max_points > 0:
            merged = self._downsample_dataframe(merged, max_points)

        # JSON Compliance
        merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Build list of all channel columns to check for NaNs (Ref + Comp)
        subset_cols = ['Time_Ref', 'Time_Comp']
        for col in req_channels:
            if col in ignore_cols: continue
            if f"{col}_Ref" in merged.columns: subset_cols.append(f"{col}_Ref")
            if f"{col}_Comp" in merged.columns: subset_cols.append(f"{col}_Comp")
            
        # Drop rows where ANY of the requested channels are NaN
        merged.dropna(subset=subset_cols, how='any', inplace=True)
        
        return merged.to_json(orient="records")