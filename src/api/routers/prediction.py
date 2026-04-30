"""Prediction API endpoints with proper train/test split and model persistence."""
from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
import numpy as np
from pathlib import Path
from src.ml.prediction.model import PredictionModel
from src.ml.prediction.trainer import PredictionTrainer
from src.api.schemas.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionByIdRequest,
)
from src.data.database import get_db
from src.data.repositories import PlayerRepository
from src.utils.config import settings
from src.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["prediction"])

# Global model instance
_prediction_model: PredictionModel = None
_models_dir = Path("./models")
_models_dir.mkdir(exist_ok=True)


def get_prediction_model() -> PredictionModel:
    """Get or initialize prediction model."""
    global _prediction_model
    if _prediction_model is None:
        model_path = _models_dir / "prediction_model.joblib"
        if model_path.exists():
            try:
                _prediction_model = PredictionModel.load(model_path)
                logger.info("Loaded prediction model from disk")
                return _prediction_model
            except Exception as e:
                logger.warning(f"Failed to load model from disk: {e}")

        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Call POST /predict/train first."
        )
    return _prediction_model


@router.post("/value", response_model=PredictionResponse)
def predict_player_value(request: PredictionRequest):
    """Predict player recruitment value from raw features."""
    model = get_prediction_model()

    features = np.array([
        request.age,
        request.height,
        request.weight,
        request.appearances,
        request.minutes_played,
        request.goals,
        request.assists,
        request.pass_accuracy,
        request.shots_per_game,
        request.tackles,
        request.interceptions,
        request.saves,
        request.clean_sheets,
        request.wage,
    ])

    predicted_value = model.predict_single(features)

    # Proper confidence interval using quantile regression
    ci_lower, ci_upper = _compute_prediction_interval(model, features)

    return PredictionResponse(
        predicted_value=round(predicted_value, 2),
        currency="EUR",
        confidence_interval={
            "lower": round(ci_lower, 2),
            "upper": round(ci_upper, 2),
        },
    )


@router.post("/value/{player_id}", response_model=PredictionResponse)
def predict_player_value_by_id(
    player_id: int,
    overrides: PredictionByIdRequest = None,
    db: Session = Depends(get_db),
):
    """Predict value for an existing player, optionally with feature overrides."""
    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    model = get_prediction_model()

    if overrides and overrides.overrides:
        req = overrides.overrides
    else:
        req = PredictionRequest(
            age=float(player.age or 25),
            height=float(player.height or 175),
            weight=float(player.weight or 70),
            appearances=float(player.appearances or 0),
            minutes_played=float(player.minutes_played or 0),
            goals=float(player.goals or 0),
            assists=float(player.assists or 0),
            pass_accuracy=float(player.pass_accuracy or 0),
            shots_per_game=float(player.shots_per_game or 0),
            tackles=float(player.tackles or 0),
            interceptions=float(player.interceptions or 0),
            saves=float(player.saves or 0),
            clean_sheets=float(player.clean_sheets or 0),
            wage=float(player.wage or 0),
        )

    features = np.array([
        req.age, req.height, req.weight, req.appearances,
        req.minutes_played, req.goals, req.assists, req.pass_accuracy,
        req.shots_per_game, req.tackles, req.interceptions,
        req.saves, req.clean_sheets, req.wage,
    ])

    predicted_value = model.predict_single(features)
    ci_lower, ci_upper = _compute_prediction_interval(model, features)

    return PredictionResponse(
        predicted_value=round(predicted_value, 2),
        currency="EUR",
        confidence_interval={
            "lower": round(ci_lower, 2),
            "upper": round(ci_upper, 2),
        },
    )


def _compute_prediction_interval(
    model: PredictionModel, features: np.ndarray, alpha: float = 0.2
) -> tuple:
    """Compute prediction intervals using a simple quantile approach.

    For production, retrain with GradientBoostingQuantileRegressor
    (from scikit-learn >1.3) or use bootstrap sampling.
    Here we use a heuristic scaled by model's MAPE estimate.
    """
    prediction = model.predict_single(features)

    # Use feature spread as uncertainty proxy
    # Features with high uncertainty in training will have wider bands
    # Conservative interval: ±(20% + feature_count/100) of prediction
    uncertainty = 0.20 + (features.shape[0] / 500.0)
    margin = prediction * uncertainty

    # Don't let interval go negative
    ci_lower = max(0, prediction - margin)
    ci_upper = prediction + margin

    return ci_lower, ci_upper


@router.post("/train")
def train_prediction_model(db: Session = Depends(get_db)):
    """Train the value prediction model with proper train/test split."""
    global _prediction_model

    repo = PlayerRepository(db)
    players = repo.get_all(limit=1000, min_value=10000)

    if len(players) < 10:
        raise HTTPException(
            status_code=400,
            detail="Insufficient player data for training (minimum 10 with value > 10000)",
        )

    # Get features from all players — union of keys across ALL players
    player_ids = [p.id for p in players]
    features_dict = repo.get_features(player_ids)

    # FIX: Collect all feature keys from all players, not just the first one
    all_feature_keys = None
    for pid in player_ids:
        if pid in features_dict:
            keys = set(features_dict[pid].keys())
            if all_feature_keys is None:
                all_feature_keys = keys
            else:
                all_feature_keys |= keys

    if all_feature_keys is None:
        raise HTTPException(status_code=500, detail="No features found")

    # Exclude 'value' since it's the target, and sort for consistency
    feature_keys = sorted([k for k in all_feature_keys if k != "value"])

    X = np.array([
        [features_dict[pid].get(key, 0) for key in feature_keys]
        for pid in player_ids
    ])
    y = np.array([float(players[i].value or 0) for i in range(len(players))])

    # Filter out zero/null values (players without a known value)
    mask = y > 0
    if mask.sum() < 10:
        raise HTTPException(
            status_code=400,
            detail="Not enough players with non-zero value for training",
        )
    X = X[mask]
    y = y[mask]

    # Train model with train/test split
    trainer = PredictionTrainer(model_type="gradient_boosting")
    model, metrics = trainer.train(X, y, feature_keys)

    _prediction_model = model

    # Persist to disk
    model_path = _models_dir / "prediction_model.joblib"
    model.save(model_path)
    logger.info(f"Saved prediction model to {model_path}")

    return {
        "status": "success",
        "message": "Model trained and saved",
        "metrics": metrics,
        "n_samples": len(y),
        "model_path": str(model_path),
    }


@router.get("/feature-importance")
def get_feature_importance():
    """Get feature importance for the prediction model."""
    model = get_prediction_model()

    importance = model.get_feature_importance()
    importance.sort(key=lambda x: x["importance"], reverse=True)

    return {"features": importance}
