import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

def generate_race_debrief(predictions_df: pd.DataFrame, year: int, round_num: int):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment.")
        return

    # Filter predictions for the specific race
    race_df = predictions_df[(predictions_df['Year'] == year) & (predictions_df['RoundNumber'] == round_num)].copy()
    
    if race_df.empty:
        logger.warning(f"No predictions found for Year {year}, Round {round_num}.")
        return
        
    # Get top 5 predicted drivers
    top_5 = race_df.head(5)
    
    prompt = f"""
    You are an expert F1 race strategist and commentator. 
    I have an ML model that predicts the finishing order for Year {year}, Round {round_num}.
    
    Here are the model's top 5 predicted finishers based on grid position, Q3 pace delta, and recent team form:
    """
    
    for _, row in top_5.iterrows():
        prompt += f"\n- {row['Abbreviation']} ({row['TeamName']}): Grid P{row['GridPosition']}, Predicted Relevance Score: {row['Prediction']:.2f}"
        
    prompt += "\n\nPlease write a short strategic debrief and prediction analysis explaining why these drivers are likely to succeed in this race."
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        print("\n--- GEMINI RACE DEBRIEF ---\n")
        print(response.text)
        print("\n---------------------------\n")
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")

def main():
    preds_path = ROOT_DIR / "data" / "processed" / "test_predictions.csv"
    if not preds_path.exists():
        logger.error(f"Predictions file not found at {preds_path}. Run train.py first.")
        return
        
    df = pd.read_csv(preds_path)
    
    # Generate debrief for the first race in the test set (2025 season if available)
    if df.empty:
        logger.warning("Predictions dataframe is empty.")
        return
        
    first_race = df.iloc[0]
    generate_race_debrief(df, first_race['Year'], first_race['RoundNumber'])

if __name__ == "__main__":
    main()
