"""Carga: grava o dataset processado em data/processed."""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_FILE


def save_processed(df: pd.DataFrame, dest: Path = PROCESSED_DATA_FILE) -> Path:
    """Grava em parquet, que preserva os tipos (categorias e booleanos)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest
