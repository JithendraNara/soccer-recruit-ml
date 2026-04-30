"""Unit tests for similarity model."""
import pytest
import numpy as np
from src.ml.similarity.model import SimilarityModel


@pytest.fixture
def fitted_model():
    """Create a fitted similarity model."""
    player_ids = [1, 2, 3, 4, 5]
    # 5 players, 4 features each
    features = np.array([
        [25, 180, 75, 10],  # player 1: young, medium height
        [30, 185, 80, 8],   # player 2
        [23, 175, 70, 15],  # player 3: very young, light
        [35, 190, 85, 3],   # player 4: old, tall
        [28, 182, 78, 12],  # player 5
    ])
    model = SimilarityModel(n_neighbors=2)
    model.fit(player_ids, features)
    return model


class TestSimilarityModel:
    """Tests for SimilarityModel."""

    def test_fit(self):
        model = SimilarityModel(n_neighbors=3)
        player_ids = [1, 2, 3]
        features = np.array([[1, 2], [3, 4], [5, 6]])
        model.fit(player_ids, features)
        assert model.is_fitted
        assert len(model.player_ids) == 3

    def test_find_similar_returns_correct_count(self, fitted_model):
        similar = fitted_model.find_similar(1, top_k=2)
        assert len(similar) == 2

    def test_find_similar_excludes_query_player(self, fitted_model):
        # Only 5 players in fixture, so ask for top 4 to stay within n_neighbors <= n_samples_fit
        similar = fitted_model.find_similar(1, top_k=4)
        player_ids = [s["player_id"] for s in similar]
        assert 1 not in player_ids

    def test_find_similar_sorted_by_similarity(self, fitted_model):
        similar = fitted_model.find_similar(1, top_k=3)
        similarities = [s["similarity"] for s in similar]
        assert similarities == sorted(similarities, reverse=True)

    def test_find_similar_unknown_player_raises(self, fitted_model):
        with pytest.raises(ValueError, match="not found"):
            fitted_model.find_similar(999)

    def test_find_similar_not_fitted_raises(self):
        model = SimilarityModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.find_similar(1)

    def test_compute_similarity_matrix_shape(self, fitted_model):
        matrix = fitted_model.compute_similarity_matrix()
        assert matrix.shape == (5, 5)

    def test_compute_similarity_matrix_diagonal_is_one(self, fitted_model):
        matrix = fitted_model.compute_similarity_matrix()
        np.testing.assert_array_almost_equal(np.diag(matrix), [1.0] * 5)

    def test_similarity_bounds(self, fitted_model):
        similar = fitted_model.find_similar(1, top_k=4)
        for s in similar:
            assert 0.0 <= s["similarity"] <= 1.0
            assert 0.0 <= s["distance"] <= 1.0
            # similarity + distance ≈ 1 for cosine metric
            assert abs(s["similarity"] + s["distance"] - 1.0) < 0.01
