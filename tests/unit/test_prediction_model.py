"""Unit tests for prediction model."""
import pytest
import numpy as np
from src.ml.prediction.model import PredictionModel


@pytest.fixture
def sample_training_data():
    """Generate synthetic player data for training."""
    np.random.seed(42)
    n_players = 50
    # Simple feature matrix: [age, appearances, goals, assists]
    X = np.column_stack([
        np.random.randint(18, 38, n_players),          # age
        np.random.randint(0, 500, n_players),           # appearances
        np.random.randint(0, 50, n_players),            # goals
        np.random.randint(0, 30, n_players),            # assists
    ])
    # Value correlated with appearances + goals
    y = (X[:, 1] * 1000 + X[:, 2] * 50000 +
         X[:, 0] * 100000 + np.random.randn(n_players) * 100000)
    feature_names = ["age", "appearances", "goals", "assists"]
    return X, y, feature_names


class TestPredictionModel:
    """Tests for PredictionModel."""

    def test_fit_and_predict(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names)
        assert model.is_fitted

    def test_predict_single(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names)
        prediction = model.predict_single(X[0])
        assert isinstance(prediction, float)
        assert prediction > 0

    def test_predict_returns_correct_shape(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_feature_importance_shape(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names)
        importance = model.get_feature_importance()
        assert len(importance) == len(feature_names)
        names = [i["feature"] for i in importance]
        assert names == feature_names

    def test_predict_not_fitted_raises(self):
        model = PredictionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[25, 100, 5, 3]]))

    def test_get_feature_importance_not_fitted_raises(self):
        model = PredictionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_model_types(self, sample_training_data):
        X, y, feature_names = sample_training_data
        for model_type in ["gradient_boosting", "random_forest", "linear"]:
            model = PredictionModel(model_type=model_type)
            model.fit(X, y, feature_names)
            assert model.is_fitted
            pred = model.predict_single(X[0])
            assert pred > 0

    def test_unknown_model_type_raises(self):
        model = PredictionModel(model_type="unknown_type")
        with pytest.raises(ValueError, match="Unknown model type"):
            model.fit(np.array([[1, 2]]), np.array([100]), ["a", "b"])

    def test_predict_with_intervals(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names, fit_quantiles=True)
        assert model.is_fitted
        assert model._lower_model is not None
        assert model._upper_model is not None

        result = model.predict_with_intervals(X[:5])
        assert "prediction" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["prediction"]) == 5
        # lower <= prediction <= upper
        for p, lo, hi in zip(result["prediction"], result["lower"], result["upper"]):
            assert lo <= p <= hi

    def test_predict_single_with_intervals(self, sample_training_data):
        X, y, feature_names = sample_training_data
        model = PredictionModel(model_type="gradient_boosting")
        model.fit(X, y, feature_names, fit_quantiles=True)
        result = model.predict_single_with_intervals(X[0])
        assert "prediction" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] <= result["prediction"] <= result["upper"]
