"""Unit tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from src.api.schemas.models import (
    PredictionRequest,
    PlayerResponse,
    PlayerCreate,
    SimilarPlayerResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest validation."""

    def test_valid_request(self):
        req = PredictionRequest(
            age=25, height=180, weight=75, appearances=30,
            minutes_played=2000, goals=10, assists=5,
            pass_accuracy=80.0, shots_per_game=2.5, tackles=20,
            interceptions=10, saves=0, clean_sheets=0, wage=50000,
        )
        assert req.age == 25
        assert req.height == 180

    def test_invalid_age_too_low(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                age=10, height=180, weight=75, appearances=30,
                minutes_played=2000, goals=10, assists=5,
                pass_accuracy=80.0, shots_per_game=2.5, tackles=20,
                interceptions=10, saves=0, clean_sheets=0, wage=50000,
            )
        assert "age" in str(exc_info.value)

    def test_invalid_age_too_high(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                age=55, height=180, weight=75, appearances=30,
                minutes_played=2000, goals=10, assists=5,
                pass_accuracy=80.0, shots_per_game=2.5, tackles=20,
                interceptions=10, saves=0, clean_sheets=0, wage=50000,
            )
        assert "age" in str(exc_info.value)

    def test_invalid_pass_accuracy(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                age=25, height=180, weight=75, appearances=30,
                minutes_played=2000, goals=10, assists=5,
                pass_accuracy=120.0, shots_per_game=2.5, tackles=20,
                interceptions=10, saves=0, clean_sheets=0, wage=50000,
            )
        assert "pass_accuracy" in str(exc_info.value)

    def test_negative_goals_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(
                age=25, height=180, weight=75, appearances=30,
                minutes_played=2000, goals=-5, assists=5,
                pass_accuracy=80.0, shots_per_game=2.5, tackles=20,
                interceptions=10, saves=0, clean_sheets=0, wage=50000,
            )

    def test_minutes_exceeds_season_max(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                age=25, height=180, weight=75, appearances=30,
                minutes_played=70000, goals=10, assists=5,
                pass_accuracy=80.0, shots_per_game=2.5, tackles=20,
                interceptions=10, saves=0, clean_sheets=0, wage=50000,
            )
        # Should fail the minutes_not_excessive validator
        assert "minutes_played" in str(exc_info.value)

    def test_position_normalized_to_uppercase(self):
        player = PlayerCreate(name="Test Player", position=" fw ")
        assert player.position == "FW"

    def test_similarity_bounds(self):
        sim = SimilarPlayerResponse(player_id=1, similarity=0.85, distance=0.15)
        assert sim.similarity == 0.85
        assert sim.distance == 0.15

    def test_similarity_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SimilarPlayerResponse(player_id=1, similarity=1.5, distance=0.15)
        with pytest.raises(ValidationError):
            SimilarPlayerResponse(player_id=1, similarity=-0.1, distance=0.15)
