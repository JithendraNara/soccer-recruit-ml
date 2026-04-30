"""Data cleaning pipeline."""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import logger


class DataCleaning:
    """Handles data cleaning and preprocessing."""

    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean player data."""
        df = df.copy()

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        text_columns = df.select_dtypes(include=["object"]).columns
        df[text_columns] = df[text_columns].fillna("Unknown")

        # Normalize text fields before dedup so whitespace variants collapse
        df["name"] = df["name"].str.strip()
        df["position"] = df["position"].str.strip().str.upper()

        # Remove duplicates — only use columns that actually exist
        dedup_cols = [c for c in ["name", "team", "season"] if c in df.columns]
        if dedup_cols:
            initial_len = len(df)
            df = df.drop_duplicates(subset=dedup_cols)
            logger.info(f"Removed {initial_len - len(df)} duplicate rows")

        # Filter invalid records
        df = df[df["age"] > 0]
        df = df[df["age"] < 50]
        df = df[df["appearances"] >= 0]

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df

    def normalize_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize specified columns to 0-1 range."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    logger.info(f"Normalized column: {col}")
        return df

    def handle_outliers(
        self, df: pd.DataFrame, column: str, method: str = "iqr"
    ) -> pd.DataFrame:
        """Handle outliers in a column."""
        df = df.copy()

        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            df.loc[outliers, column] = df[column].median()
            logger.info(f"Handled {outliers.sum()} outliers in {column}")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return report."""
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }
        logger.info(f"Data validation: {report['total_rows']} rows valid")
        return report
