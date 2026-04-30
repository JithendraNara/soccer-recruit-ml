"""Player value prediction model with optional quantile regression for prediction intervals."""
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
        self._lower_model = None
        self._upper_model = None

    def _create_model(self, **kwargs):
        """Create the underlying model."""
        if self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
            )
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=kwargs.get("alpha", 1.0))
        elif self.model_type == "quantile":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
                loss="quantile",
                alpha=kwargs.get("quantile_alpha", 0.5),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        fit_quantiles: bool = True,
    ) -> "PredictionModel":
        """Fit the prediction model and optionally quantile models for prediction intervals."""
        kwargs = {}

        # Fit main model
        self.model = self._create_model(**kwargs)
        self.model.fit(X, y)
        self.feature_names = feature_names
        self.is_fitted = True

        if fit_quantiles:
            try:
                from sklearn.ensemble import GradientBoostingRegressor

                # Lower bound (10th percentile)
                self._lower_model = GradientBoostingRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 5),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    random_state=43,
                    loss="quantile",
                    alpha=0.10,
                )
                self._lower_model.fit(X, y)

                # Upper bound (90th percentile)
                self._upper_model = GradientBoostingRegressor(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 5),
                    learning_rate=kwargs.get("learning_rate", 0.1),
                    random_state=44,
                    loss="quantile",
                    alpha=0.90,
                )
                self._upper_model.fit(X, y)

                logger.info("Fitted quantile regression models for prediction intervals")
            except Exception as e:
                logger.warning(f"Could not fit quantile models: {e}")
                self._lower_model = None
                self._upper_model = None

        logger.info(f"Fitted {self.model_type} prediction model")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict player values."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_with_intervals(
        self, X: np.ndarray, alpha: float = 0.20
    ) -> Dict[str, np.ndarray]:
        """Predict with prediction intervals using quantile regression.

        Args:
            X: Feature matrix
            alpha: Significance level (default 0.20 → 80% prediction interval)

        Returns:
            Dict with 'prediction', 'lower', 'upper'
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        prediction = self.model.predict(X)

        if self._lower_model is not None and self._upper_model is not None:
            lower = self._lower_model.predict(X)
            upper = self._upper_model.predict(X)
            # Ensure lower <= prediction <= upper
            lower = np.minimum(lower, prediction)
            upper = np.maximum(upper, prediction)
        else:
            # Fallback: use percentage-based heuristic
            margin = prediction * (alpha / 2 + 0.05)
            lower = np.maximum(0, prediction - margin)
            upper = prediction + margin

        return {
            "prediction": prediction,
            "lower": lower,
            "upper": upper,
        }

    def predict_single(self, features: np.ndarray) -> float:
        """Predict value for a single player."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.predict(features)[0])

    def predict_single_with_intervals(
        self, features: np.ndarray, alpha: float = 0.20
    ) -> Dict[str, float]:
        """Predict value for a single player with prediction interval."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        result = self.predict_with_intervals(features, alpha)
        return {
            "prediction": float(result["prediction"][0]),
            "lower": float(result["lower"][0]),
            "upper": float(result["upper"][0]),
        }

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

        total = importances.sum()
        normalized = (importances / total * 100) if total > 0 else importances

        return [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(self.feature_names, normalized)
        ]

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import cloudpickle

        model_data = {
            "model_type": self.model_type,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "_lower_model": self._lower_model,
            "_upper_model": self._upper_model,
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
        model._lower_model = model_data.get("_lower_model")
        model._upper_model = model_data.get("_upper_model")

        logger.info(f"Loaded prediction model from {path}")
        return model
