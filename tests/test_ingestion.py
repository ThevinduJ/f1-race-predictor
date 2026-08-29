import pandas as pd

from src.ingestion import ingest_season


def test_ingest_season_invalid_year():
    # Test that an invalid year (e.g. far future) returns an empty dataframe safely
    df = ingest_season(2050)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
