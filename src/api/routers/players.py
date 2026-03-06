"""Player API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from src.data.database import get_db
from src.data.repositories import PlayerRepository
from src.api.schemas.models import PlayerResponse, PlayerListResponse, PlayerCreate
from src.utils.logger import logger

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=PlayerListResponse)
def list_players(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    position: Optional[str] = None,
    league: Optional[str] = None,
    min_value: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """List all players with optional filters."""
    repo = PlayerRepository(db)
    players = repo.get_all(
        skip=skip,
        limit=limit,
        position=position,
        league=league,
        min_value=min_value
    )
    total = len(players)

    return PlayerListResponse(
        total=total,
        players=[PlayerResponse.model_validate(p) for p in players]
    )


@router.get("/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, db: Session = Depends(get_db)):
    """Get a player by ID."""
    repo = PlayerRepository(db)
    player = repo.get_by_id(player_id)

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    return PlayerResponse.model_validate(player)


@router.post("", response_model=PlayerResponse, status_code=201)
def create_player(player: PlayerCreate, db: Session = Depends(get_db)):
    """Create a new player."""
    repo = PlayerRepository(db)
    created = repo.create(player.model_dump())
    return PlayerResponse.model_validate(created)


@router.get("/search/{name}", response_model=List[PlayerResponse])
def search_players(name: str, db: Session = Depends(get_db)):
    """Search players by name."""
    repo = PlayerRepository(db)
    players = repo.get_by_name(name)
    return [PlayerResponse.model_validate(p) for p in players]
