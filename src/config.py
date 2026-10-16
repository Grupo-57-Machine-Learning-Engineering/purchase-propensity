"""Configuração central: paths, origem do dataset e logging.

Fonte única de verdade para valores compartilhados entre as etapas do
pipeline. Os valores configuráveis vêm do `.env` (modelo em `.env.example`);
sem ele, ou com a variável vazia, valem os padrões abaixo. Variável de
ambiente já definida tem prioridade sobre o `.env`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def env(name: str, default: str) -> str:
    """Lê uma variável do ambiente; ausente ou vazia, devolve `default`."""
    return os.getenv(name) or default


def path_from_env(name: str, default: str) -> Path:
    """Lê um path do ambiente; caminho relativo é resolvido a partir da raiz."""
    path = Path(env(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Paths -------------------------------------------------------------------
DATA_DIR: Path = path_from_env("DATA_DIR", "data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Dataset (Customer Propensity to Purchase / Kaggle) ----------------------
# Versão fixa por padrão para o ETL ser reproduzível mesmo se o autor publicar outra.
DATASET_HANDLE: str = env(
    "DATASET_HANDLE", "benpowis/customer-propensity-to-purchase-data/versions/3"
)
RAW_DATA_FILENAME: str = "training_sample.csv"
RAW_DATA_FILE: Path = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_FILE: Path = PROCESSED_DATA_DIR / "customer_propensity.parquet"

# --- Logging -----------------------------------------------------------------
LOG_LEVEL: str = env("LOG_LEVEL", "INFO").upper()
