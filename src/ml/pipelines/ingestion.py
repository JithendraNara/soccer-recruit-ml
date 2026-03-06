"""Data ingestion pipeline."""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import logger
from src.utils.config import settings


class DataIngestion:
    """Handles ingestion of soccer data from various sources."""

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_json(self, file_path: Path) -> pd.DataFrame:
        """Load data from JSON file."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_json(file_path)
        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_directory(self, directory: Path) -> pd.DataFrame:
        """Load all data files from a directory."""
        all_data = []

        for file_path in directory.glob("*.csv"):
            df = self.load_csv(file_path)
            all_data.append(df)

        for file_path in directory.glob("*.json"):
            df = self.load_json(file_path)
            all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined)} rows")
            return combined

        logger.warning(f"No data files found in {directory}")
        return pd.DataFrame()

    def to_dict_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of dictionaries."""
        return df.to_dict(orient="records")
