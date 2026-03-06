"""Training pipeline for prediction model."""
import mlflow
import numpy as np
from typing import Dict, Any, Tuple
from src.ml.prediction.model import PredictionModel
from src.ml.pipelines.transformation import FeatureTransformation
from src.utils.config import settings
from src.utils.logger import logger


class PredictionTrainer:
    """Trainer for prediction model with MLflow tracking."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.transformation = FeatureTransformation()
        self.model = PredictionModel(model_type=model_type)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        params: Dict[str, Any] = None
    ) -> Tuple[PredictionModel, Dict[str, float]]:
        """Train the prediction model with MLflow tracking."""
        mlflow.set_experiment("soccer-value-prediction")

        with mlflow.start_run(run_name="value-prediction-training"):
            # Log parameters
            params = params or {}
            mlflow.log_params({
                "model_type": params.get("model_type", "gradient_boosting"),
                "n_samples": len(X),
                "n_features": X.shape[1],
            })

            # Scale features
            X_scaled = self.transformation.scale_features(X, method="standard")

            # Fit model
            model = self.model.fit(X_scaled, y, feature_names)

            # Calculate training metrics
            predictions = model.predict(X_scaled)
            metrics = self._calculate_metrics(y, predictions)

            # Log metrics
            mlflow.log_metrics({
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            })

            logger.info(f"Training metrics: {metrics}")

        return model, metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
        }

    def evaluate(
        self,
        model: PredictionModel,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        X_scaled = self.transformation.scale_features(X_test, method="standard")
        predictions = model.predict(X_scaled)
        metrics = self._calculate_metrics(y_test, predictions)

        mlflow.log_metrics({
            f"test_{k}": v for k, v in metrics.items()
        })

        return metrics
