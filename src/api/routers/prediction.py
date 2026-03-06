"""Prediction API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from src.ml.prediction.model import PredictionModel
from src.ml.prediction.trainer import PredictionTrainer
from src.api.schemas.models import PredictionRequest, PredictionResponse
from src.data.database import get_db
from src.data.repositories import PlayerRepository
from src.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["prediction"])

# Global model instance
_prediction_model: PredictionModel = None


def get_prediction_model() -> PredictionModel:
    """Get or initialize prediction model."""
    global _prediction_model
    if _prediction_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Call /predict/train first."
        )
    return _prediction_model


@router.post("/value", response_model=PredictionResponse)
def predict_player_value(request: PredictionRequest):
    """Predict player recruitment value."""
    model = get_prediction_model()

    # Convert request to feature vector
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

    # Make prediction
    predicted_value = model.predict_single(features)

    # Calculate confidence interval (simplified)
    # In production, use proper prediction intervals
    ci_lower = predicted_value * 0.8
    ci_upper = predicted_value * 1.2

    return PredictionResponse(
        predicted_value=round(predicted_value, 2),
        currency="EUR",
        confidence_interval={
            "lower": round(ci_lower, 2),
            "upper": round(ci_upper, 2)
        }
    )


@router.post("/train")
def train_prediction_model(db=Depends(get_db)):
    """Train the value prediction model."""
    global _prediction_model

    repo = PlayerRepository(db)
    players = repo.get_all(limit=1000, min_value=10000)

    if len(players) < 10:
        raise HTTPException(
            status_code=400,
            detail="Insufficient player data for training"
        )

    # Get features and target
    player_ids = [p.id for p in players]
    features_dict = repo.get_features(player_ids)

    # Prepare data - exclude 'value' since it's the target
    all_feature_keys = list(features_dict[player_ids[0]].keys())
    feature_keys = [k for k in all_feature_keys if k != "value"]
    X = np.array([[features_dict[pid][key] for key in feature_keys] for pid in player_ids])
    y = np.array([players[i].value or 0 for i in range(len(players))])

    # Filter out zero values
    mask = y > 0
    X = X[mask]
    y = y[mask]

    # Train model
    trainer = PredictionTrainer(model_type="gradient_boosting")
    model, metrics = trainer.train(X, y, feature_keys)

    _prediction_model = model

    return {
        "status": "success",
        "message": "Model trained successfully",
        "metrics": metrics,
        "n_samples": len(y)
    }


@router.get("/feature-importance")
def get_feature_importance():
    """Get feature importance for the prediction model."""
    model = get_prediction_model()

    importance = model.get_feature_importance()
    importance.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "features": importance
    }
