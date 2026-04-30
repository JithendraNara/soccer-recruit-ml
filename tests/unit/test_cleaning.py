"""Unit tests for data cleaning pipeline."""
import pytest
import pandas as pd
import numpy as np
from src.ml.pipelines.cleaning import DataCleaning


@pytest.fixture
def dirty_player_df():
    """Create a dirty player DataFrame with missing values, duplicates, outliers."""
    return pd.DataFrame([
        {"name": "  Messi  ", "age": 36, "position": " fw ", "team": "Inter Miami",
         "season": "2024", "appearances": 30, "minutes_played": 2400,
         "goals": 25, "assists": 15, "value": 50000000, "wage": 500000},
        {"name": "Messi", "age": 36, "position": "FW", "team": "Inter Miami",
         "season": "2024", "appearances": 30, "minutes_played": 2400,
         "goals": 25, "assists": 15, "value": 50000000, "wage": 500000},  # duplicate
        {"name": "Haaland", "age": 23, "position": "FW", "team": "Man City",
         "season": "2024", "appearances": 30, "minutes_played": 2400,
         "goals": 40, "assists": 8, "value": 180000000, "wage": 375000},
        {"name": "Unknown Player", "age": -5, "position": "DEF", "team": "Unknown",
         "season": "2024", "appearances": 0, "minutes_played": 0,
         "goals": 0, "assists": 0, "value": 1000000, "wage": 50000},  # invalid age
        {"name": "Old Player", "age": 55, "position": "GK", "team": "Retired FC",
         "season": "2024", "appearances": 0, "minutes_played": 0,
         "goals": 0, "assists": 0, "value": 500000, "wage": 0},  # age too high
        {"name": "De Bruyne", "age": 32, "position": "CM", "team": "Man City",
         "season": "2023", "appearances": 25, "minutes_played": 2000,
         "goals": 8, "assists": 15, "value": 70000000, "wage": 400000},
        {"name": "De Bruyne", "age": 32, "position": "CM", "team": "Man City",
         "season": "2023", "appearances": 25, "minutes_played": 2000,
         "goals": 8, "assists": 15, "value": 70000000, "wage": 400000},  # duplicate
    ])


class TestDataCleaning:
    """Tests for DataCleaning."""

    def test_clean_player_data(self, dirty_player_df):
        dc = DataCleaning()
        cleaned = dc.clean_player_data(dirty_player_df)
        assert len(cleaned) < len(dirty_player_df)
        assert not any(cleaned["name"].str.contains("  "))  # no extra spaces
        assert all(cleaned["position"].str.isupper())

    def test_clean_removes_duplicates(self, dirty_player_df):
        dc = DataCleaning()
        cleaned = dc.clean_player_data(dirty_player_df)
        assert len(cleaned) < len(dirty_player_df)
        names = cleaned["name"].tolist()
        assert len(names) == len(set(names))

    def test_clean_filters_invalid_ages(self, dirty_player_df):
        dc = DataCleaning()
        cleaned = dc.clean_player_data(dirty_player_df)
        assert all(cleaned["age"] > 0)
        assert all(cleaned["age"] < 50)

    def test_validate_data(self, dirty_player_df):
        dc = DataCleaning()
        report = dc.validate_data(dirty_player_df)
        assert report["total_rows"] == len(dirty_player_df)
        assert report["duplicate_rows"] > 0
        assert "age" in report["missing_values"]

    def test_handle_outliers_iqr(self):
        df = pd.DataFrame({"value": [100, 200, 300, 400, 500, 1000000]})
        dc = DataCleaning()
        cleaned = dc.handle_outliers(df, "value", method="iqr")
        assert cleaned["value"].max() < 1000000  # outlier capped

    def test_normalize_columns(self):
        df = pd.DataFrame({"a": [0, 50, 100], "b": [10, 20, 30]})
        dc = DataCleaning()
        normalized = dc.normalize_columns(df, ["a"])
        assert normalized["a"].min() == 0.0
        assert normalized["a"].max() == 1.0
