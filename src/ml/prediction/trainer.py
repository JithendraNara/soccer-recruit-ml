"""Training pipeline for prediction model."""
import mlflow
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split as sklearn_split
from src.ml.prediction.model import PredictionModel
from src.ml.pipelines.transformation import FeatureTransformation
from src.utils.config import settings
from src.utils.logger import logger


class PredictionTrainer:
    """Trainer for prediction model with MLflow tracking."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.transformation = FeatureTransformation()
        self.model = PredictionModel(model_type=model_type)
        self._test_metrics: Optional[Dict[str, float]] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        params: Dict[str, Any] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        fit_quantiles: bool = True,
    ) -> Tuple[PredictionModel, Dict[str, float]]:
        """Train the prediction model with MLflow tracking and train/test split.

        Args:
            fit_quantiles: If True, also fits 10th/90th percentile models for
                          prediction intervals (via quantile regression).
        """
        mlflow.set_experiment("soccer-value-prediction")
        mlflow.set_experiment("soccer-value-prediction")

        # Stratified split if enough samples and classification-like target
        if len(y) >= 20:
            X_train, X_test, y_train, y_test = sklearn_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            # Fallback for small datasets
            split_idx = int(len(y) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

        params = params or {}

        with mlflow.start_run(run_name="value-prediction-training"):
            # Log parameters
            mlflow.log_params({
                "model_type": params.get("model_type", "gradient_boosting"),
                "n_samples": len(X),
                "n_features": X.shape[1],
                "test_size": test_size,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            # Scale features using ONLY training data, then apply same scaler to test
            # (this was a bug — original code re-fit scaler on test data, causing data leak)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit model on training data
            model = self.model.fit(X_train_scaled, y_train, feature_names, fit_quantiles=fit_quantiles)

            # Calculate training metrics (on train set)
            train_preds = model.predict(X_train_scaled)
            train_metrics = self._calculate_metrics(y_train, train_preds)

            mlflow.log_metrics({
                f"train_{k}": v for k, v in train_metrics.items()
            })

            # Calculate test metrics (on held-out set — this is the real signal)
            test_preds = model.predict(X_test_scaled)
            test_metrics = self._calculate_metrics(y_test, test_preds)

            self._test_metrics = test_metrics
            mlflow.log_metrics({
                f"test_{k}": v for k, v in test_metrics.items()
            })

            logger.info(f"Training metrics: {train_metrics}")
            logger.info(f"Test metrics: {test_metrics}")

        return model, test_metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

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
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        X_scaled = self.transformation.scale_features(X_test, method="standard")
        predictions = model.predict(X_scaled)
        metrics = self._calculate_metrics(y_test, predictions)

        mlflow.log_metrics({
            f"test_{k}": v for k, v in metrics.items()
        })

        return metrics
