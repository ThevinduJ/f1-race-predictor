import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from raw FastF1 results dataframe.
    Required features:
    - Grid position
    - Q3 pace deltas
    - Rolling team form
    """
    if df.empty:
        logger.warning("Empty dataframe provided to extract_features.")
        return pd.DataFrame()

    # Separate Quali and Race data
    df_q = df[df['SessionType'] == 'Q'].copy()
    df_r = df[df['SessionType'] == 'R'].copy()

    # 1. Grid Position and Race Outcome
    # FastF1 race results contain 'GridPosition' and 'Position' (finishing position), and 'Points'
    # We will build our base dataframe from the Race sessions.
    base_df = df_r[['Year', 'RoundNumber', 'EventName', 'Abbreviation', 'DriverNumber', 
                    'TeamName', 'GridPosition', 'Position', 'Points']].copy()

    # Convert GridPosition to numeric
    base_df['GridPosition'] = pd.to_numeric(base_df['GridPosition'], errors='coerce')
    
    # 2. Q3 Pace Deltas
    # We will look at df_q to find the fastest Q3 time per event, and calculate delta for each driver
    df_q['Q3_seconds'] = pd.to_timedelta(df_q['Q3']).dt.total_seconds()
    
    # Find the minimum Q3 time per event (the pole lap or fastest in Q3)
    q3_min = df_q.groupby(['Year', 'RoundNumber'])['Q3_seconds'].min().reset_index()
    q3_min.rename(columns={'Q3_seconds': 'Min_Q3_seconds'}, inplace=True)
    
    df_q = df_q.merge(q3_min, on=['Year', 'RoundNumber'], how='left')
    df_q['Q3_Pace_Delta'] = df_q['Q3_seconds'] - df_q['Min_Q3_seconds']
    
    # If a driver didn't make Q3, their Q3_Pace_Delta will be NaN. We could fill with a large penalty or leave as NaN for LightGBM.
    # For now, let's keep it as NaN, LightGBM handles missing values natively.
    
    quali_features = df_q[['Year', 'RoundNumber', 'Abbreviation', 'Q3_Pace_Delta']]
    
    # Merge Quali features into base
    dataset = base_df.merge(quali_features, on=['Year', 'RoundNumber', 'Abbreviation'], how='left')

    # 3. Rolling Team Form (Average points over last 3 races)
    # First, sort dataset chronologically
    dataset.sort_values(by=['Year', 'RoundNumber'], inplace=True)
    
    # Calculate team points per race
    team_points = dataset.groupby(['Year', 'RoundNumber', 'TeamName'])['Points'].sum().reset_index()
    
    # Sort team points and calculate rolling average
    team_points.sort_values(by=['Year', 'RoundNumber'], inplace=True)
    
    # Shift by 1 so the current race's points are not included in the historical rolling average
    def rolling_avg(series, window=3):
        return series.shift(1).rolling(window=window, min_periods=1).mean()
        
    team_points['Team_Rolling_Form'] = team_points.groupby('TeamName')['Points'].apply(rolling_avg).reset_index(level=0, drop=True)
    
    # Merge back to dataset
    dataset = dataset.merge(team_points[['Year', 'RoundNumber', 'TeamName', 'Team_Rolling_Form']], 
                            on=['Year', 'RoundNumber', 'TeamName'], how='left')
    
    # Handle NaN in rolling form (e.g. first race of the season/dataset)
    dataset['Team_Rolling_Form'] = dataset['Team_Rolling_Form'].fillna(0)

    # Sort and clean up
    dataset.sort_values(by=['Year', 'RoundNumber', 'Position'], inplace=True)
    
    # Drop drivers who didn't start/finish if their position is NaN (Optional)
    dataset['Position'] = pd.to_numeric(dataset['Position'], errors='coerce')
    
    logger.info(f"Feature extraction completed. Shape: {dataset.shape}")
    return dataset

def main():
    raw_path = ROOT_DIR / "data" / "processed" / "raw_results.csv"
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run ingestion first.")
        return

    raw_df = pd.read_csv(raw_path)
    features_df = extract_features(raw_df)
    
    out_path = ROOT_DIR / "data" / "processed" / "race_features.csv"
    features_df.to_csv(out_path, index=False)
    logger.info(f"Features saved to {out_path}")

if __name__ == "__main__":
    main()
