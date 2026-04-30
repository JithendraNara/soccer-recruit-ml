"""Data access layer for player data."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from src.data.models import Player, PlayerEmbedding
from src.utils.logger import logger


class PlayerRepository:
    """Repository for player data access."""

    def __init__(self, db: Session):
        self.db = db

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        position: Optional[str] = None,
        league: Optional[str] = None,
        min_value: Optional[float] = None,
    ) -> List[Player]:
        """Get all players with optional filters."""
        query = self.db.query(Player)

        filters = []
        if position:
            filters.append(Player.position == position)
        if league:
            filters.append(Player.league == league)
        if min_value is not None:
            filters.append(Player.value >= min_value)

        if filters:
            query = query.filter(and_(*filters))

        return query.offset(skip).limit(limit).all()

    def get_by_id(self, player_id: int) -> Optional[Player]:
        """Get player by ID."""
        return self.db.query(Player).filter(Player.id == player_id).first()

    def get_by_name(self, name: str) -> List[Player]:
        """Get players by name (partial match)."""
        return self.db.query(Player).filter(Player.name.ilike(f"%{name}%")).all()

    def create(self, player_data: dict) -> Player:
        """Create a new player."""
        player = Player(**player_data)
        self.db.add(player)
        self.db.commit()
        self.db.refresh(player)
        logger.info(f"Created player: {player.name}")
        return player

    def bulk_create(self, players_data: List[dict]) -> List[Player]:
        """Bulk create players."""
        players = [Player(**data) for data in players_data]
        self.db.bulk_save_objects(players)
        self.db.commit()
        logger.info(f"Bulk created {len(players)} players")
        return players

    def get_features(self, player_ids: List[int]) -> dict:
        """Get feature + metadata dict for given player IDs.

        Includes position for position encoding in similarity pipelines.
        """
        players = self.db.query(Player).filter(Player.id.in_(player_ids)).all()

        feature_keys = [
            "age", "height", "weight", "appearances", "minutes_played",
            "goals", "assists", "pass_accuracy", "shots_per_game",
            "tackles", "interceptions", "saves", "clean_sheets", "wage",
        }

        features = {}
        for player in players:
            features[player.id] = {
                key: getattr(player, key, None) or 0
                for key in feature_keys
            }
            # Include position for similarity position encoding
            features[player.id]["position"] = player.position

        return features


class EmbeddingRepository:
    """Repository for player embeddings."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, player_id: int) -> Optional[PlayerEmbedding]:
        """Get embedding for a player."""
        return self.db.query(PlayerEmbedding).filter(
            PlayerEmbedding.player_id == player_id
        ).first()

    def save(self, player_id: int, embedding: List[float]) -> PlayerEmbedding:
        """Save or update player embedding."""
        import json
        from datetime import datetime

        existing = self.get(player_id)
        embedding_str = json.dumps(embedding)

        if existing:
            existing.embedding = embedding_str
            existing.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            new_embedding = PlayerEmbedding(
                player_id=player_id,
                embedding=embedding_str,
                updated_at=datetime.now()
            )
            self.db.add(new_embedding)
            self.db.commit()
            self.db.refresh(new_embedding)
            return new_embedding
