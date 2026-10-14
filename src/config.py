"""Configuração central: paths e origem do dataset.

Fonte única de verdade para valores compartilhados entre as etapas do
pipeline. Sem lógica de negócio, apenas constantes.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Dataset (Online Shoppers Purchasing Intention / UCI) --------------------
DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
RAW_DATA_FILE: Path = RAW_DATA_DIR / "online_shoppers_intention.csv"
PROCESSED_DATA_FILE: Path = PROCESSED_DATA_DIR / "online_shoppers.parquet"
