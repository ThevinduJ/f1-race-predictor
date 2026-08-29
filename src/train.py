import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

def prepare_ranking_data(df: pd.DataFrame):
    # Sort data by event (query) to satisfy LGBMRanker requirement
    df = df.sort_values(by=['Year', 'RoundNumber'])
    
    features = ['GridPosition', 'Q3_Pace_Delta', 'Team_Rolling_Form']
    
    # We want to predict finishing position. For learning to rank, higher relevance is better.
    # We can inverse the position so P1 gets the highest relevance, or use points as relevance.
    # Let's use 'Points' as the relevance label since it inherently ranks top 10 well, 
    # but to rank everyone, let's create a relevance score: 25 - Position (max 20 drivers typically).
    df['Relevance'] = 25 - df['Position'].fillna(25)
    df['Relevance'] = df['Relevance'].apply(lambda x: max(0, x))
    
    # Create query groups (number of items per group/race)
    qids = df.groupby(['Year', 'RoundNumber']).size().to_numpy()
    
    X = df[features]
    y = df['Relevance']
    
    return X, y, qids, df

def main():
    features_path = ROOT_DIR / "data" / "processed" / "race_features.csv"
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        return

    df = pd.read_csv(features_path)
    
    # Temporal Split: Train 2023, Val 2024, Test 2025
    train_df = df[df['Year'] == 2023].copy()
    val_df = df[df['Year'] == 2024].copy()
    test_df = df[df['Year'] == 2025].copy()
    
    if train_df.empty:
        logger.warning("No training data for 2023.")
        return
        
    X_train, y_train, qids_train, _ = prepare_ranking_data(train_df)
    X_val, y_val, qids_val, _ = prepare_ranking_data(val_df)
    X_test, y_test, qids_test, test_ref = prepare_ranking_data(test_df)
    
    logger.info(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
    
    # Initialize LGBMRanker
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    eval_set = [(X_val, y_val)] if not X_val.empty else None
    eval_group = [qids_val] if not X_val.empty else None
    
    ranker.fit(
        X_train, y_train, 
        group=qids_train,
        eval_set=eval_set,
        eval_group=eval_group,
        eval_at=[5, 10]
    )
    
    model_path = ROOT_DIR / "data" / "processed" / "ranker_model.txt"
    ranker.booster_.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    if X_test.empty:
        logger.warning("No test data for 2025 yet. Falling back to the latest 2024 race for predictions.")
        last_race_val = val_df['RoundNumber'].max()
        test_df = val_df[val_df['RoundNumber'] == last_race_val].copy()
        X_test, y_test, qids_test, test_ref = prepare_ranking_data(test_df)
        
    preds = ranker.predict(X_test)
    test_ref['Prediction'] = preds
    test_ref.sort_values(by=['Year', 'RoundNumber', 'Prediction'], ascending=[True, True, False], inplace=True)
    out_preds = ROOT_DIR / "data" / "processed" / "test_predictions.csv"
    test_ref.to_csv(out_preds, index=False)
    logger.info(f"Test predictions saved to {out_preds}")

if __name__ == "__main__":
    main()
