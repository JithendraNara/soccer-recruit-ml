"""Similarity API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from src.data.database import get_db
from src.data.repositories import PlayerRepository
from src.ml.similarity.model import SimilarityModel
from src.ml.pipelines.transformation import FeatureTransformation
from src.api.schemas.models import SimilarPlayersResponse, SimilarPlayerResponse
from src.utils.logger import logger

router = APIRouter(prefix="/similarity", tags=["similarity"])

# Global model instance (in production, use proper model registry)
_similarity_model: SimilarityModel = None
_feature_transformer = FeatureTransformation()


def get_similarity_model() -> SimilarityModel:
    """Get or initialize similarity model."""
    global _similarity_model
    if _similarity_model is None:
        # Initialize with sample data - in production, load from saved model
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Call /similarity/train first."
        )
    return _similarity_model


@router.post("/train")
def train_similarity_model(
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Train the similarity model on current data."""
    global _similarity_model

    repo = PlayerRepository(db)
    players = repo.get_all(limit=1000)

    if not players:
        raise HTTPException(
            status_code=400,
            detail="No player data available for training"
        )

    # Get features
    player_ids = [p.id for p in players]
    features_dict = repo.get_features(player_ids)

    # Convert to list format
    feature_keys = list(features_dict[player_ids[0]].keys())
    features = [[features_dict[pid][key] for key in feature_keys] for pid in player_ids]

    # Create and train model
    _similarity_model = SimilarityModel(n_neighbors=top_k)
    _similarity_model.fit(player_ids, features)

    return {
        "status": "success",
        "message": f"Model trained with {len(player_ids)} players",
        "n_players": len(player_ids),
        "n_features": len(feature_keys)
    }


@router.get("/{player_id}/similar", response_model=SimilarPlayersResponse)
def find_similar_players(
    player_id: int,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Find similar players to the given player."""
    model = get_similarity_model()

    # Verify player exists
    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    # Find similar players
    similar = model.find_similar(player_id, top_k=top_k)

    # Get player names
    similar_players = []
    for s in similar:
        similar_player = repo.get_by_id(s["player_id"])
        if similar_player:
            similar_players.append(
                SimilarPlayerResponse(
                    player_id=s["player_id"],
                    similarity=s["similarity"],
                    distance=s["distance"]
                )
            )

    return SimilarPlayersResponse(
        player_id=player_id,
        player_name=player.name,
        similar_players=similar_players
    )


@router.get("/matrix/{player_id}")
def get_similarity_matrix(player_id: int, db: Session = Depends(get_db)):
    """Get similarity matrix for a player."""
    model = get_similarity_model()

    # Verify player exists
    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    # Compute full similarity matrix
    matrix = model.compute_similarity_matrix()
    idx = model.player_ids.index(player_id)

    # Get similarities for this player
    player_similarities = [
        {
            "player_id": model.player_ids[i],
            "similarity": float(matrix[idx][i])
        }
        for i in range(len(model.player_ids))
        if model.player_ids[i] != player_id
    ]

    # Sort by similarity
    player_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "player_id": player_id,
        "player_name": player.name,
        "similarities": player_similarities[:50]
    }
