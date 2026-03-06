"""Player value prediction model."""
import numpy as np
from typing import List, Dict, Any, Optional
import joblib
from pathlib import Path
from src.utils.logger import logger


class PredictionModel:
    """Model for predicting player recruitment value."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []

    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> "PredictionModel":
        """Fit the prediction model."""
        self.model = self._create_model()
        self.model.fit(X, y)
        self.feature_names = feature_names
        self.is_fitted = True

        logger.info(f"Fitted {self.model_type} prediction model")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict player values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X)
        logger.info(f"Made {len(predictions)} predictions")
        return predictions

    def predict_single(self, features: np.ndarray) -> float:
        """Predict value for a single player."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.predict(features)[0])

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            return []

        # Normalize to percentages
        total = importances.sum()
        normalized = (importances / total * 100) if total > 0 else importances

        return [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(self.feature_names, normalized)
        ]

    def save(self, path: Path) -> None:
        """Save model to disk."""
        model_data = {
            "model_type": self.model_type,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        joblib.dump(model_data, path)
        logger.info(f"Saved prediction model to {path}")

    @classmethod
    def load(cls, path: Path) -> "PredictionModel":
        """Load model from disk."""
        model_data = joblib.load(path)
        model = cls(model_type=model_data["model_type"])
        model.model = model_data["model"]
        model.is_fitted = model_data["is_fitted"]
        model.feature_names = model_data["feature_names"]

        logger.info(f"Loaded prediction model from {path}")
        return model
