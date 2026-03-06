"""Configuration settings for the SoccerRecruit platform."""
import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""

    # App
    app_name: str = "SoccerRecruit ML"
    app_version: str = "1.0.0"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./soccer_recruit.db"

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "soccer-similarity"

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list = ["*"]

    # Model
    similarity_top_k: int = 5
    embedding_dim: int = 32

    @property
    def base_dir(self) -> Path:
        """Get base directory."""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.base_dir / "data"

    @property
    def sample_data_dir(self) -> Path:
        """Get sample data directory."""
        return self.data_dir / "sample"


settings = Settings()
