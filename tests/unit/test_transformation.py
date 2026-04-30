"""Unit tests for feature transformation pipeline."""
import pytest
import pandas as pd
import numpy as np
from src.ml.pipelines.transformation import FeatureTransformation


@pytest.fixture
def sample_player_df():
    """Create a sample player DataFrame."""
    return pd.DataFrame([
        {
            "id": 1, "name": "Messi", "age": 36, "height": 170, "weight": 72,
            "appearances": 30, "minutes_played": 2400, "goals": 25, "assists": 15,
            "pass_accuracy": 85.0, "shots_per_game": 4.0, "tackles": 10,
            "interceptions": 8, "saves": 0, "clean_sheets": 0, "value": 50000000,
            "wage": 500000, "position": "FW",
        },
        {
            "id": 2, "name": "Van Dijk", "age": 32, "height": 193, "weight": 92,
            "appearances": 35, "minutes_played": 3150, "goals": 3, "assists": 2,
            "pass_accuracy": 87.0, "shots_per_game": 0.5, "tackles": 55,
            "interceptions": 40, "saves": 0, "clean_sheets": 15, "value": 65000000,
            "wage": 250000, "position": "CB",
        },
    ])


class TestFeatureTransformation:
    """Tests for FeatureTransformation."""

    def test_create_derived_features(self, sample_player_df):
        df = FeatureTransformation().create_derived_features(sample_player_df)
        assert "goals_per_game" in df.columns
        assert df.loc[df["name"] == "Messi", "goals_per_game"].iloc[0] == pytest.approx(25 / 30, rel=0.01)
        assert "is_forward" in df.columns
        assert "is_goalkeeper" in df.columns
        assert df.loc[df["name"] == "Messi", "is_forward"].iloc[0] == 1
        assert df.loc[df["name"] == "Van Dijk", "is_forward"].iloc[0] == 0

    def test_create_feature_matrix(self, sample_player_df):
        ft = FeatureTransformation()
        features, names = ft.create_feature_matrix(sample_player_df)
        assert features.shape[0] == 2
        assert len(names) == features.shape[1]
        assert "age" in names

    def test_create_feature_matrix_filters_nan(self, sample_player_df):
        df = sample_player_df.copy()
        df.loc[0, "goals"] = np.nan
        ft = FeatureTransformation()
        features, _ = ft.create_feature_matrix(df)
        assert not np.isnan(features).any()

    def test_scale_features_standard(self, sample_player_df):
        ft = FeatureTransformation()
        matrix, _ = ft.create_feature_matrix(sample_player_df)
        scaled = ft.scale_features(matrix, method="standard")
        # Scaled features should have mean ≈ 0, std ≈ 1
        assert scaled.shape == matrix.shape
        col_means = scaled.mean(axis=0)
        assert np.allclose(col_means, 0, atol=0.1)

    def test_scale_features_minmax(self, sample_player_df):
        ft = FeatureTransformation()
        matrix, _ = ft.create_feature_matrix(sample_player_df)
        scaled = ft.scale_features(matrix, method="minmax")
        assert scaled.shape == matrix.shape
        col_min = scaled.min(axis=0)
        col_max = scaled.max(axis=0)
        # All columns should be in [0, 1]; constant columns may have min==max
        for i in range(len(col_min)):
            assert col_min[i] >= 0.0, f"col {i} min={col_min[i]} < 0"
            assert col_max[i] <= 1.0 + 1e-10, f"col {i} max={col_max[i]} > 1"

    def test_get_player_vector(self, sample_player_df):
        ft = FeatureTransformation()
        vector = ft.get_player_vector(sample_player_df, player_id=1)
        assert vector.shape[0] == len(ft.feature_columns)

    def test_get_player_vector_not_found(self, sample_player_df):
        ft = FeatureTransformation()
        with pytest.raises(ValueError, match="not found"):
            ft.get_player_vector(sample_player_df, player_id=999)

    def test_position_encoding_forward(self):
        ft = FeatureTransformation()
        enc = ft.create_position_encoding("FW")
        assert enc[-1] == 1  # FWD is last

    def test_position_encoding_goalkeeper(self):
        ft = FeatureTransformation()
        enc = ft.create_position_encoding("GK")
        assert enc[0] == 1  # GK is first
