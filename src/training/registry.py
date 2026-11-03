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
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_REGISTERED_MODEL_NAME
from src.logger import get_logger
from src.training.compare import rank_runs

CHAMPION_ALIAS = "champion"

logger = get_logger(__name__)


def _best_model_id(client: MlflowClient, run: Run) -> str:
    """`model_id` do modelo logado em `run`; levanta erro se não houver nenhum.

    `search_runs` (usado por `rank_runs`) não traz `outputs`; só `get_run` traz.
    """
    full_run = client.get_run(run.info.run_id)
    model_outputs = full_run.outputs.model_outputs if full_run.outputs else None
    if not model_outputs:
        raise ValueError(f"Run {run.info.run_id} não tem modelo logado para promover.")
    return model_outputs[0].model_id


def _current_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    """Versão com o alias `champion`, ou `None` se o modelo ou o alias não existirem."""
    try:
        return client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        return None


def promote_best_model(name: str = MLFLOW_REGISTERED_MODEL_NAME) -> str:
    """Registra o run com maior `cv_pr_auc_mean` e aponta `champion` pra ele.

    Se o `champion` já é esse modelo, não cria versão nova: rodar o pipeline de
    novo sem mudar o vencedor não enche o Registry de versões repetidas.
    """
    client = MlflowClient()
    runs = rank_runs(client)
    if not runs:
        raise ValueError("Nenhum run encontrado no experimento para promover.")
    model_uri = f"models:/{_best_model_id(client, runs[0])}"

    champion = _current_champion(client, name)
    if champion is not None and champion.source == model_uri:
        logger.info("%s versao %s já é o champion; nada a promover", name, champion.version)
        return str(champion.version)
    return _register_as_champion(client, name, model_uri, runs[0])


def _register_as_champion(client: MlflowClient, name: str, model_uri: str, run: Run) -> str:
    """Cria uma versão nova de `name` a partir de `model_uri` e aponta `champion` pra ela."""
    version = str(mlflow.register_model(model_uri, name).version)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    logger.info(
        "Run %s (%s) promovido a champion: %s versao %s",
        run.info.run_id,
        run.data.params.get("model", "?"),
        name,
        version,
    )
    return version


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Promove o melhor run para o Model Registry.")
    parser.add_argument("--name", default=MLFLOW_REGISTERED_MODEL_NAME)
    args = parser.parse_args()
    promote_best_model(name=args.name)


if __name__ == "__main__":
    main()
