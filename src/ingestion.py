import logging
import os
from pathlib import Path
from typing import List, Optional

import fastf1
import fastf1.events
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure FastF1 cache
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def ingest_season(year: int) -> pd.DataFrame:
    """Ingest qualifying and race sessions for a given year."""
    logger.info(f"Starting ingestion for season {year}")
    
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logger.warning(f"Failed to fetch schedule for {year}: {e}")
        return pd.DataFrame()

    results: List[pd.DataFrame] = []

    for _, event in schedule.iterrows():
        # Ignore pre-season testing
        if event['EventFormat'] == 'testing':
            continue

        event_name = event['EventName']
        round_num = event['RoundNumber']
        
        logger.info(f"Processing Year {year} | Round {round_num} | {event_name}")

        # Attempt to load Qualifying
        try:
            session_q = fastf1.get_session(year, round_num, 'Q')
            session_q.load(telemetry=False, weather=False, messages=False)
            df_q = session_q.results
            df_q['SessionType'] = 'Q'
            df_q['Year'] = year
            df_q['RoundNumber'] = round_num
            df_q['EventName'] = event_name
            results.append(df_q)
        except Exception as e:
            logger.warning(f"Could not load Qualifying for {event_name} {year}: {e}")

        # Attempt to load Race
        try:
            session_r = fastf1.get_session(year, round_num, 'R')
            session_r.load(telemetry=False, weather=False, messages=False)
            df_r = session_r.results
            df_r['SessionType'] = 'R'
            df_r['Year'] = year
            df_r['RoundNumber'] = round_num
            df_r['EventName'] = event_name
            results.append(df_r)
        except Exception as e:
            logger.warning(f"Could not load Race for {event_name} {year}: {e}")

    if not results:
        return pd.DataFrame()
        
    season_data = pd.concat(results, ignore_index=True)
    return season_data

def main():
    years_to_ingest = [2023, 2024, 2025]
    all_data_list = []
    
    for y in years_to_ingest:
        df_season = ingest_season(y)
        if not df_season.empty:
            all_data_list.append(df_season)
            
    if all_data_list:
        final_df = pd.concat(all_data_list, ignore_index=True)
        out_path = ROOT_DIR / "data" / "processed" / "raw_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path, index=False)
        logger.info(f"Successfully saved aggregated data to {out_path}")
    else:
        logger.warning("No data was ingested.")

if __name__ == "__main__":
    main()
