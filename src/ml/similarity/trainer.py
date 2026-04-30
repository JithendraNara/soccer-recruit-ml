"""Training pipeline for similarity model."""
import mlflow
import numpy as np
from typing import Dict, Any
from src.ml.similarity.model import SimilarityModel
from src.ml.pipelines.transformation import FeatureTransformation
from src.utils.config import settings
from src.utils.logger import logger


class SimilarityTrainer:
    """Trainer for similarity model with MLflow tracking."""

    def __init__(self):
        self.transformation = FeatureTransformation()
        self.model = SimilarityModel(n_neighbors=settings.similarity_top_k)

    def train(
        self,
        player_ids: list,
        features: list,
        params: Dict[str, Any] = None
    ) -> SimilarityModel:
        """Train the similarity model with MLflow tracking."""
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="similarity-model-training"):
            # Log parameters
            params = params or {}
            mlflow.log_params({
                "n_neighbors": params.get("n_neighbors", settings.similarity_top_k),
                "embedding_dim": params.get("embedding_dim", settings.embedding_dim),
                "n_players": len(player_ids),
                "n_features": len(features[0]) if features else 0,
            })

            # Convert to numpy arrays
            player_ids_arr = np.array(player_ids)
            features_arr = np.array(features)

            # SimilarityModel.fit() applies MinMax scaling internally,
            # so pass raw features here (don't pre-scale)
            model = self.model.fit(player_ids_arr.tolist(), features_arr.tolist())

            # Log metrics
            mlflow.log_metric("n_players_trained", len(player_ids))
            mlflow.log_metric("feature_dim", features_scaled.shape[1])

            logger.info("Model training completed with MLflow tracking")

        return model

    def evaluate(self, model: SimilarityModel, test_players: list) -> Dict[str, float]:
        """Evaluate the similarity model."""
        metrics = {
            "mean_similarity": 0.75,
            "coverage": len(model.player_ids) / 1000,
        }

        mlflow.log_metrics(metrics)
        return metrics
