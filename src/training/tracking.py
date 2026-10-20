"""Registro de runs no MLflow: params, métricas, tags e o modelo treinado.

A URI do servidor e as credenciais não aparecem no código: o MLflow lê
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` do
ambiente (o `src.config` carrega o `.env` local). Sem URI, grava em `./mlruns`.
Esta etapa não registra modelo no Registry.
"""

import getpass
import hashlib
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator

from src.config import MLFLOW_EXPERIMENT_NAME, PROJECT_ROOT

UNKNOWN = "unknown"
HASH_CHUNK_BYTES = 1024 * 1024
DATASET_VERSION_LENGTH = 12


def _git_output(*args: str) -> str:
    """Roda um comando git na raiz do projeto; sem git ou sem repo, devolve vazio."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def get_git_sha() -> str:
    """Commit atual do código que gerou a run, ou `unknown` fora de um repo git."""
    return _git_output("rev-parse", "HEAD") or UNKNOWN


def get_author() -> str:
    """Quem rodou: `user.name` do git, ou o usuário do sistema se não houver."""
    return _git_output("config", "user.name") or getpass.getuser()


def get_dataset_version(path: Path) -> str:
    """Versão do dataset: início do SHA-256 do arquivo, sem depender do DVC."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()[:DATASET_VERSION_LENGTH]


def build_tags(dataset_path: Path) -> dict[str, str]:
    """Tags padrão de toda run: `autor`, `git_sha` e `dataset_version`."""
    return {
        "autor": get_author(),
        "git_sha": get_git_sha(),
        "dataset_version": get_dataset_version(dataset_path),
    }


def log_run(
    run_name: str,
    model: BaseEstimator,
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    tags: dict[str, str],
    input_example: pd.DataFrame,
) -> str:
    """Grava uma run no experimento do projeto e devolve o `run_id`."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=input_example)
    return run.info.run_id
