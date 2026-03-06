"""Feature engineering and transformation pipeline."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import logger


class FeatureTransformation:
    """Handles feature engineering for player data."""

    def __init__(self):
        self.feature_columns = [
            "age", "height", "weight", "appearances", "minutes_played",
            "goals", "assists", "pass_accuracy", "shots_per_game",
            "tackles", "interceptions", "saves", "clean_sheets", "value", "wage"
        ]

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        df = df.copy()

        # Goals per game
        df["goals_per_game"] = np.where(
            df["appearances"] > 0,
            df["goals"] / df["appearances"],
            0
        )

        # Assists per game
        df["assists_per_game"] = np.where(
            df["appearances"] > 0,
            df["assists"] / df["appearances"],
            0
        )

        # Minutes per game
        df["minutes_per_game"] = np.where(
            df["appearances"] > 0,
            df["minutes_played"] / df["appearances"],
            0
        )

        # Goal involvement per game
        df["goal_involvement"] = df["goals_per_game"] + df["assists_per_game"]

        # Value per goal (for forwards)
        df["value_per_goal"] = np.where(
            df["goals"] > 0,
            df["value"] / df["goals"],
            0
        )

        # Experience score (based on appearances)
        df["experience_score"] = np.clip(
            df["appearances"] / 100, 0, 10
        )

        # Position-based features
        df["is_goalkeeper"] = (df["position"] == "GK").astype(int)
        df["is_forward"] = df["position"].isin(["FW", "ST", "CF"]).astype(int)
        df["is_midfielder"] = df["position"].isin(["CM", "CDM", "CAM", "LM", "RM"]).astype(int)
        df["is_defender"] = df["position"].isin(["CB", "LB", "RB", "LWB", "RWB"]).astype(int)

        logger.info("Created derived features")
        return df

    def create_feature_matrix(
        self, df: pd.DataFrame, player_ids: List[int] = None
    ) -> tuple:
        """Create feature matrix for ML models."""
        if player_ids:
            df = df[df["id"].isin(player_ids)]

        available_features = [col for col in self.feature_columns if col in df.columns]
        feature_matrix = df[available_features].values

        # Fill NaN with 0
        feature_matrix = np.nan_to_num(feature_matrix, 0)

        logger.info(f"Created feature matrix: {feature_matrix.shape}")
        return feature_matrix, available_features

    def scale_features(
        self, features: np.ndarray, method: str = "standard"
    ) -> np.ndarray:
        """Scale features using specified method."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        scaled = scaler.fit_transform(features)
        logger.info(f"Scaled features using {method}")
        return scaled

    def get_player_vector(self, df: pd.DataFrame, player_id: int) -> np.ndarray:
        """Get feature vector for a specific player."""
        player = df[df["id"] == player_id]
        if player.empty:
            raise ValueError(f"Player {player_id} not found")

        features, _ = self.create_feature_matrix(player)
        return features[0]

    def create_position_encoding(self, position: str) -> np.ndarray:
        """Create one-hot encoding for position."""
        positions = ["GK", "DEF", "MID", "FWD"]
        encoding = np.zeros(len(positions))

        if position in ["GK"]:
            encoding[0] = 1
        elif position in ["CB", "LB", "RB", "LWB", "RWB"]:
            encoding[1] = 1
        elif position in ["CM", "CDM", "CAM", "LM", "RM"]:
            encoding[2] = 1
        else:
            encoding[3] = 1

        return encoding
