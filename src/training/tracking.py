"""Registro de runs no MLflow: params, métricas, tags e o modelo treinado.

A URI do servidor e as credenciais não aparecem no código: o MLflow lê
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` do
ambiente (o `src.config` carrega o `.env` local). Sem URI, grava em `./mlruns`.
Esta etapa só loga a run; quem registra no Model Registry é
`src/training/registry.py`, a partir do `model_id` do `LoggedModel` criado aqui.
"""

import getpass
import hashlib
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.sklearn import log_model
from sklearn.base import BaseEstimator

from src.config import MLFLOW_EXPERIMENT_NAME, PROJECT_ROOT

UNKNOWN = "unknown"
HASH_CHUNK_BYTES = 1024 * 1024
DATASET_VERSION_LENGTH = 12
# Armazenamento interno de árvore do scikit-learn: usado por Random Forest, HistGB
# etc. O skops recusa esse tipo por padrão (não valida os índices dos nós), mas o
# modelo vem do nosso próprio treino, nunca de um arquivo externo, então confiar
# nele aqui é seguro.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


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
    """Grava uma run no experimento do projeto e devolve o `run_id`.

    O `model_id` do `LoggedModel` (MLflow 3.x) não é devolvido aqui porque não é
    usado para registrar a run no Registry: `promote_best_model`
    (`src/training/registry.py`) busca o `model_id` certo pelo `run_id`, depois de
    decidir qual run é a melhor.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        log_model(
            model,
            name="model",
            input_example=input_example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    return run.info.run_id
