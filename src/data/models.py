"""Database models for soccer player data."""
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from src.data.database import Base


class Player(Base):
    """Player database model."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    age = Column(Integer)
    nationality = Column(String(100))
    position = Column(String(50))
    height = Column(Integer)  # cm
    weight = Column(Integer)  # kg

    # Performance stats
    appearances = Column(Integer, default=0)
    minutes_played = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)

    # Advanced stats
    pass_accuracy = Column(Float)
    shots_per_game = Column(Float)
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    saves = Column(Integer, default=0)  # For goalkeepers
    clean_sheets = Column(Integer, default=0)

    # Contract info
    value = Column(Float)  # In euros
    wage = Column(Float)  # Weekly wage in euros
    contract_end = Column(Date)

    # Metadata
    league = Column(String(100))
    team = Column(String(255))
    season = Column(String(20))

    def __repr__(self):
        return f"<Player {self.name} - {self.position}>"


class PlayerEmbedding(Base):
    """Pre-computed player embeddings for similarity search."""

    __tablename__ = "player_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), unique=True)
    embedding = Column(String(1000))  # JSON string of embedding vector
    updated_at = Column(Date)

    player = relationship("Player", backref="embedding")
