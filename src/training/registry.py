"""Promove o melhor modelo. Rode com `python -m src.training.registry`.

Registra no MLflow Model Registry o run com maior `cv_pr_auc_mean`
(`src/training/compare.py` decide qual é) e aponta o alias `champion` pra essa
versão — aliases, não mais "stages", que o MLflow 2.9+/3.x descontinuou.

Usa `models:/<model_id>` (o `LoggedModel` do MLflow 3.x, criado por
`mlflow.sklearn.log_model(model, name=..., ...)` em `src/training/tracking.py`)
em vez de `runs:/<run_id>/model`: contra o servidor do Dagshub, listar e baixar
artefato pelo path do run falha (a listagem de artefatos do run não devolve os
arquivos), mas pelo `model_id` funciona. Confirmado com um teste manual contra o
servidor real antes de escrever este módulo.
"""

import argparse

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_REGISTERED_MODEL_NAME
from src.logger import get_logger
from src.training.compare import rank_runs

CHAMPION_ALIAS = "champion"

logger = get_logger(__name__)


def _best_model_id(run: Run) -> str:
    """`model_id` do modelo logado em `run`; levanta erro se não houver nenhum."""
    model_outputs = run.outputs.model_outputs if run.outputs else None
    if not model_outputs:
        raise ValueError(f"Run {run.info.run_id} não tem modelo logado para promover.")
    return model_outputs[0].model_id


def promote_best_model(name: str = MLFLOW_REGISTERED_MODEL_NAME) -> str:
    """Registra o run com maior `cv_pr_auc_mean` e aponta `champion` pra ele."""
    client = MlflowClient()
    runs = rank_runs(client)
    if not runs:
        raise ValueError("Nenhum run encontrado no experimento para promover.")
    best_run = runs[0]

    model_id = _best_model_id(best_run)
    result = mlflow.register_model(f"models:/{model_id}", name)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, result.version)
    logger.info(
        "Run %s (%s) promovido a champion: %s versao %s",
        best_run.info.run_id,
        best_run.data.params.get("model", "?"),
        name,
        result.version,
    )
    return result.version


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Promove o melhor run para o Model Registry.")
    parser.add_argument("--name", default=MLFLOW_REGISTERED_MODEL_NAME)
    args = parser.parse_args()
    promote_best_model(name=args.name)


if __name__ == "__main__":
    main()
