"""Script to load sample data into the database."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import SessionLocal, init_db
from src.data.repositories import PlayerRepository
from src.ml.pipelines.ingestion import DataIngestion
from src.utils.logger import logger


def main():
    """Load sample data."""
    logger.info("Initializing database...")
    init_db()

    # Load sample data
    data_dir = Path(__file__).parent.parent / "data" / "sample"
    ingestion = DataIngestion()

    if (data_dir / "players.csv").exists():
        df = ingestion.load_csv(data_dir / "players.csv")

        # Prepare data for database
        players_data = df.to_dict(orient="records")

        # Save to database
        db = SessionLocal()
        try:
            repo = PlayerRepository(db)
            repo.bulk_create(players_data)
            logger.info(f"Loaded {len(players_data)} players")
        finally:
            db.close()
    else:
        logger.error("No sample data found")


if __name__ == "__main__":
    main()
