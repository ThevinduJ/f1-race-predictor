import pandas as pd
import numpy as np
from src.features import extract_features

def test_extract_features_empty():
    df = extract_features(pd.DataFrame())
    assert df.empty

def test_extract_features_mock_data():
    mock_data = pd.DataFrame({
        'Year': [2023, 2023],
        'RoundNumber': [1, 1],
        'SessionType': ['Q', 'R'],
        'EventName': ['Test GP', 'Test GP'],
        'Abbreviation': ['VER', 'VER'],
        'DriverNumber': [1, 1],
        'TeamName': ['Red Bull', 'Red Bull'],
        'GridPosition': [1, 1],
        'Position': [1, 1],
        'Points': [0, 25],
        'Q3': [pd.Timedelta(minutes=1, seconds=30), pd.NaT]
    })
    
    features_df = extract_features(mock_data)
    
    assert not features_df.empty
    assert 'Q3_Pace_Delta' in features_df.columns
    assert 'Team_Rolling_Form' in features_df.columns
    
    # Q3 delta should be 0 since he is the only driver
    assert features_df['Q3_Pace_Delta'].iloc[0] == 0
    # First race so rolling form should be 0
    assert features_df['Team_Rolling_Form'].iloc[0] == 0
