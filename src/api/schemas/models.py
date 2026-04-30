"""Pydantic models for API schemas with proper field validation."""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from datetime import date


class PlayerBase(BaseModel):
    """Base player schema."""

    name: str
    age: Optional[int] = Field(default=None, ge=15, le=50)
    nationality: Optional[str] = None
    position: Optional[str] = None
    height: Optional[int] = Field(default=None, ge=140, le=220)
    weight: Optional[int] = Field(default=None, ge=40, le=130)
    appearances: Optional[int] = Field(default=0, ge=0)
    minutes_played: Optional[int] = Field(default=0, ge=0)
    goals: Optional[int] = Field(default=0, ge=0)
    assists: Optional[int] = Field(default=0, ge=0)
    pass_accuracy: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    shots_per_game: Optional[float] = Field(default=None, ge=0.0)
    tackles: Optional[int] = Field(default=0, ge=0)
    interceptions: Optional[int] = Field(default=0, ge=0)
    saves: Optional[int] = Field(default=0, ge=0)
    clean_sheets: Optional[int] = Field(default=0, ge=0)
    value: Optional[float] = Field(default=None, ge=0.0)
    wage: Optional[float] = Field(default=None, ge=0.0)
    league: Optional[str] = None
    team: Optional[str] = None
    season: Optional[str] = None

    @field_validator("position")
    @classmethod
    def normalize_position(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.strip().upper()
        return v


class PlayerCreate(PlayerBase):
    """Schema for creating a player."""

    pass


class PlayerResponse(PlayerBase):
    """Schema for player response."""

    id: int

    model_config = ConfigDict(from_attributes=True)


class PlayerListResponse(BaseModel):
    """Schema for player list response."""

    total: int
    players: List[PlayerResponse]


class SimilarPlayerResponse(BaseModel):
    """Schema for similar player response."""

    player_id: int
    similarity: float = Field(ge=0.0, le=1.0)
    distance: float = Field(ge=0.0)


class SimilarPlayersResponse(BaseModel):
    """Schema for similar players list response."""

    player_id: int
    player_name: str
    similar_players: List[SimilarPlayerResponse]


class PredictionRequest(BaseModel):
    """Schema for prediction request with field validation."""

    age: float = Field(..., ge=15, le=50)
    height: float = Field(..., ge=140, le=220)
    weight: float = Field(..., ge=40, le=130)
    appearances: float = Field(..., ge=0)
    minutes_played: float = Field(..., ge=0)
    goals: float = Field(..., ge=0)
    assists: float = Field(..., ge=0)
    pass_accuracy: float = Field(..., ge=0.0, le=100.0)
    shots_per_game: float = Field(..., ge=0.0)
    tackles: float = Field(..., ge=0)
    interceptions: float = Field(..., ge=0)
    saves: float = Field(..., ge=0)
    clean_sheets: float = Field(..., ge=0)
    wage: float = Field(..., ge=0)

    @field_validator("minutes_played")
    @classmethod
    def minutes_not_excessive(cls, v: float, info) -> float:
        if v > 60000:
            raise ValueError("minutes_played exceeds maximum possible per season")
        return v


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predicted_value: float
    currency: str = "EUR"
    confidence_interval: Optional[dict] = None


class PredictionByIdRequest(BaseModel):
    """Schema for prediction by player ID with optional overrides."""

    player_id: int
    overrides: Optional[PredictionRequest] = None


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    database: str
