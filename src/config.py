"""Configuração central: paths, origem do dataset, logging e parâmetros do treino.

Fonte única de verdade para valores compartilhados entre as etapas do
pipeline. Dois arquivos alimentam este módulo:

- `.env` (modelo em `.env.example`): ambiente da máquina, como pasta de dados,
  nível de log e credenciais do MLflow. Sem ele, ou com a variável vazia, valem os
  padrões abaixo. Variável de ambiente já definida tem prioridade sobre o `.env`.
- `configs/params.yaml`: seed, split, cross-validation, threshold e
  hiperparâmetros. Fica no git e o DVC rastreia cada chave.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PARAMS_FILE: Path = PROJECT_ROOT / "configs" / "params.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def env(name: str, default: str) -> str:
    """Lê uma variável do ambiente; ausente ou vazia, devolve `default`."""
    return os.getenv(name) or default


def path_from_env(name: str, default: str) -> Path:
    """Lê um path do ambiente; caminho relativo é resolvido a partir da raiz."""
    path = Path(env(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """Lê os parâmetros do treino (YAML)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_PARAMS = load_params()


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

# --- Treino ------------------------------------------------------------------
TARGET_COLUMN: str = "ordered"
# As 3 flags do fim do funil descrevem a compra acontecendo, não a intenção antes
# dela (EDA em docs/eda-findings.md, seção 8): nenhum pedido acontece sem
# `saw_checkout`, e `sign_in` e `checked_delivery_detail` quase só aparecem em quem
# comprou. Usá-las como feature é vazamento, então ficam fora do treino por padrão.
LEAKAGE_COLUMNS: tuple[str, ...] = ("sign_in", "saw_checkout", "checked_delivery_detail")
RANDOM_SEED: int = _PARAMS["seed"]
TEST_SIZE: float = _PARAMS["split"]["test_size"]
DECISION_THRESHOLD: float = _PARAMS["threshold"]
CV_FOLDS: int = _PARAMS["cv"]["folds"]
LOGISTIC_REGRESSION_PARAMS: dict[str, Any] = _PARAMS["models"]["logistic_regression"]
RANDOM_FOREST_PARAMS: dict[str, Any] = _PARAMS["models"]["random_forest"]

# --- MLflow ------------------------------------------------------------------
# URI e credenciais vêm só das variáveis de ambiente (MLFLOW_TRACKING_URI,
# MLFLOW_TRACKING_USERNAME e MLFLOW_TRACKING_PASSWORD), que o próprio MLflow lê.
MLFLOW_EXPERIMENT_NAME: str = "purchase-propensity"
MLFLOW_REGISTERED_MODEL_NAME: str = "purchase-propensity"
