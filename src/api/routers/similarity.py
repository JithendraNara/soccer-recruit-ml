"""Similarity API endpoints with model persistence."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
from src.data.database import get_db
from src.data.repositories import PlayerRepository
from src.ml.similarity.model import SimilarityModel
from src.ml.pipelines.transformation import FeatureTransformation
from src.api.schemas.models import SimilarPlayersResponse, SimilarPlayerResponse
from src.utils.config import settings
from src.utils.logger import logger

router = APIRouter(prefix="/similarity", tags=["similarity"])

_models_dir = Path("./models")
_models_dir.mkdir(exist_ok=True)


class SimilarityModelRegistry:
    """Registry for managing similarity model lifecycle."""

    def __init__(self):
        self._model: SimilarityModel = None
        self._feature_keys: List[str] = []

    def _build_features(self, repo: PlayerRepository, player_ids: List[int]) -> tuple:
        """Build feature matrix by union-ing keys across ALL players and adding position encoding."""
        features_dict = repo.get_features(player_ids)
        players = repo.get_all(limit=10000)
        player_map = {p.id: p for p in players}

        # Collect keys from every player, take the union
        all_keys = set()
        for pid in player_ids:
            if pid in features_dict:
                all_keys |= set(features_dict[pid].keys())

        # Exclude 'position' from the numeric matrix — it's decoded as one-hot below
        feature_keys = sorted(key for key in all_keys if key != "position")

        # Build matrix with position one-hot encoding appended
        ft = FeatureTransformation()
        features = []
        for pid in player_ids:
            row = [features_dict[pid].get(key, 0) for key in feature_keys]
            player = player_map.get(pid)
            if player and player.position:
                pos_enc = ft.create_position_encoding(player.position).tolist()
            else:
                pos_enc = [0, 0, 0, 0]
            features.append(row + pos_enc)

        # Append position key names
        pos_key_names = ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]
        all_feature_keys = feature_keys + pos_key_names

        return features, all_feature_keys

    def train(self, repo: PlayerRepository, player_ids: List[int], top_k: int = 5) -> dict:
        """Train and persist the similarity model."""
        features, feature_keys = self._build_features(repo, player_ids)
        self._feature_keys = feature_keys

        self._model = SimilarityModel(n_neighbors=top_k)
        self._model.fit(player_ids, features)

        # Persist
        model_path = _models_dir / "similarity_model.joblib"
        self._model.save(model_path)

        logger.info(f"Trained and saved similarity model: {len(player_ids)} players, {len(feature_keys)} features")
        return {
            "n_players": len(player_ids),
            "n_features": len(feature_keys),
            "feature_keys": feature_keys,
            "model_path": str(model_path),
        }

    def get(self) -> SimilarityModel:
        """Get loaded model or load from disk."""
        if self._model is None:
            model_path = _models_dir / "similarity_model.joblib"
            if model_path.exists():
                try:
                    self._model = SimilarityModel.load(model_path)
                    logger.info("Loaded similarity model from disk")
                except Exception as e:
                    logger.warning(f"Failed to load similarity model: {e}")
                    raise HTTPException(
                        status_code=503,
                        detail="Similarity model corrupted. Call POST /similarity/train first.",
                    )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Similarity model not initialized. Call POST /similarity/train first.",
                )
        return self._model


_model_registry = SimilarityModelRegistry()


@router.post("/train")
def train_similarity_model(
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Train the similarity model on current player data."""
    repo = PlayerRepository(db)
    players = repo.get_all(limit=1000)

    if not players:
        raise HTTPException(
            status_code=400,
            detail="No player data available for training",
        )

    player_ids = [p.id for p in players]
    result = _model_registry.train(repo, player_ids, top_k=top_k)

    return {
        "status": "success",
        "message": f"Model trained with {result['n_players']} players and {result['n_features']} features",
        **result,
    }


@router.get("/{player_id}/similar", response_model=SimilarPlayersResponse)
def find_similar_players(
    player_id: int,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Find similar players to the given player."""
    model = _model_registry.get()

    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    similar = model.find_similar(player_id, top_k=top_k)

    similar_players = []
    for s in similar:
        similar_player = repo.get_by_id(s["player_id"])
        if similar_player:
            similar_players.append(
                SimilarPlayerResponse(
                    player_id=s["player_id"],
                    similarity=s["similarity"],
                    distance=s["distance"],
                )
            )

    return SimilarPlayersResponse(
        player_id=player_id,
        player_name=player.name,
        similar_players=similar_players,
    )


@router.get("/matrix/{player_id}")
def get_similarity_matrix(player_id: int, db: Session = Depends(get_db)):
    """Get full similarity matrix for a player (top 50)."""
    model = _model_registry.get()

    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    matrix = model.compute_similarity_matrix()
    idx = model.player_ids.index(player_id)

    player_similarities = [
        {"player_id": model.player_ids[i], "similarity": float(matrix[idx][i])}
        for i in range(len(model.player_ids))
        if model.player_ids[i] != player_id
    ]

    player_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "player_id": player_id,
        "player_name": player.name,
        "similarities": player_similarities[:50],
    }
