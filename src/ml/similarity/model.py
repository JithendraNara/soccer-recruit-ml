"""Player similarity model with MinMax-scaled features for KNN."""
import numpy as np
from typing import List, Dict, Any, Optional
import joblib
from pathlib import Path
from src.utils.logger import logger


class SimilarityModel:
    """Model for computing player similarity using MinMax-scaled KNN."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.player_ids: List[int] = []
        self.features: Optional[np.ndarray] = None
        self.is_fitted = False
        self._scaler = None

    def fit(self, player_ids: List[int], features) -> "SimilarityModel":
        """Fit the similarity model with MinMax-scaled player features.

        MinMax scaling ensures each feature is in [0, 1] so no single feature
        (e.g., absolute wage or value) dominates the cosine distance.
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import MinMaxScaler

        self.player_ids = player_ids
        self.features = np.array(features, dtype=np.float64)

        # MinMax scale all features to [0, 1] — prevents scale bias in cosine similarity
        self._scaler = MinMaxScaler()
        features_scaled = self._scaler.fit_transform(self.features)

        self.model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(player_ids)),
            metric="cosine",
            algorithm="brute",
        )
        self.model.fit(features_scaled)
        self.is_fitted = True

        logger.info(f"Fitted similarity model with {len(player_ids)} players (MinMax-scaled)")
        return self

    def find_similar(
        self, player_id: int, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find similar players to the given player."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if player_id not in self.player_ids:
            raise ValueError(f"Player {player_id} not found in model.")

        idx = self.player_ids.index(player_id)
        player_vector = self.features[idx].reshape(1, -1)

        # Scale the query vector with the SAME scaler fitted during training
        player_vector_scaled = self._scaler.transform(player_vector)

        k = top_k or self.n_neighbors
        distances, indices = self.model.kneighbors(player_vector_scaled, n_neighbors=k + 1)

        results = []
        for dist, i in zip(distances[0], indices[0]):
            if self.player_ids[i] != player_id:
                similarity = 1 - dist
                results.append({
                    "player_id": self.player_ids[i],
                    "similarity": float(similarity),
                    "distance": float(dist),
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute full pairwise similarity matrix on scaled features."""
        from sklearn.metrics.pairwise import cosine_similarity

        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        features_scaled = self._scaler.transform(self.features)
        similarity_matrix = cosine_similarity(features_scaled)
        logger.info(f"Computed similarity matrix: {similarity_matrix.shape}")
        return similarity_matrix

    def save(self, path: Path) -> None:
        """Save model to disk including the scaler."""
        model_data = {
            "n_neighbors": self.n_neighbors,
            "player_ids": self.player_ids,
            "features": self.features,
            "is_fitted": self.is_fitted,
            "scaler": self._scaler,
        }
        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path) -> "SimilarityModel":
        """Load model from disk including scaler."""
        model_data = joblib.load(path)
        model = cls(n_neighbors=model_data["n_neighbors"])
        model.player_ids = model_data["player_ids"]
        model.features = model_data["features"]
        model.is_fitted = model_data["is_fitted"]
        model._scaler = model_data.get("scaler")

        if model.is_fitted:
            from sklearn.neighbors import NearestNeighbors
            model.model = NearestNeighbors(
                n_neighbors=min(model.n_neighbors + 1, len(model.player_ids)),
                metric="cosine",
                algorithm="brute",
            )
            features_scaled = model._scaler.transform(model.features)
            model.model.fit(features_scaled)

        logger.info(f"Loaded model from {path}")
        return model
