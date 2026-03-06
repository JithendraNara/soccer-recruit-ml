"""Pydantic models for API schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class PlayerBase(BaseModel):
    """Base player schema."""

    name: str
    age: Optional[int] = None
    nationality: Optional[str] = None
    position: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    appearances: Optional[int] = 0
    minutes_played: Optional[int] = 0
    goals: Optional[int] = 0
    assists: Optional[int] = 0
    pass_accuracy: Optional[float] = None
    shots_per_game: Optional[float] = None
    tackles: Optional[int] = 0
    interceptions: Optional[int] = 0
    saves: Optional[int] = 0
    clean_sheets: Optional[int] = 0
    value: Optional[float] = None
    wage: Optional[float] = None
    league: Optional[str] = None
    team: Optional[str] = None
    season: Optional[str] = None


class PlayerCreate(PlayerBase):
    """Schema for creating a player."""

    pass


class PlayerResponse(PlayerBase):
    """Schema for player response."""

    id: int

    class Config:
        from_attributes = True


class PlayerListResponse(BaseModel):
    """Schema for player list response."""

    total: int
    players: List[PlayerResponse]


class SimilarPlayerResponse(BaseModel):
    """Schema for similar player response."""

    player_id: int
    similarity: float
    distance: float


class SimilarPlayersResponse(BaseModel):
    """Schema for similar players list response."""

    player_id: int
    player_name: str
    similar_players: List[SimilarPlayerResponse]


class PredictionRequest(BaseModel):
    """Schema for prediction request."""

    age: float
    height: float
    weight: float
    appearances: float
    minutes_played: float
    goals: float
    assists: float
    pass_accuracy: float
    shots_per_game: float
    tackles: float
    interceptions: float
    saves: float
    clean_sheets: float
    wage: float


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predicted_value: float
    currency: str = "EUR"
    confidence_interval: Optional[dict] = None


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    database: str
