"""Testes da promoção do melhor modelo para o Model Registry."""

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.config import MLFLOW_EXPERIMENT_NAME
from src.training.registry import CHAMPION_ALIAS, promote_best_model


def _log_run_with_model(client: MlflowClient, run_name: str, cv_pr_auc_mean: float) -> str:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model", type(model).__name__)
        mlflow.log_metric("cv_pr_auc_mean", cv_pr_auc_mean)
        mlflow.sklearn.log_model(model, name="model")
    return run.info.run_id


def test_promote_best_model_registers_the_highest_cv_pr_auc_run(
    local_mlflow: MlflowClient,
) -> None:
    pior_run_id = _log_run_with_model(local_mlflow, "pior", cv_pr_auc_mean=0.3)
    melhor_run_id = _log_run_with_model(local_mlflow, "melhor", cv_pr_auc_mean=0.7)

    version = promote_best_model(name="modelo-teste")

    champion = local_mlflow.get_model_version_by_alias("modelo-teste", CHAMPION_ALIAS)
    assert champion.version == version
    assert champion.run_id == melhor_run_id
    assert champion.run_id != pior_run_id


def test_promote_best_model_without_runs_raises(local_mlflow: MlflowClient) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with pytest.raises(ValueError, match="Nenhum run"):
        promote_best_model(name="modelo-teste-vazio")


def test_promote_best_model_champion_is_loadable(local_mlflow: MlflowClient) -> None:
    _log_run_with_model(local_mlflow, "unico", cv_pr_auc_mean=0.5)

    promote_best_model(name="modelo-carregavel")

    loaded = mlflow.sklearn.load_model(f"models:/modelo-carregavel@{CHAMPION_ALIAS}")
    assert isinstance(loaded, LogisticRegression)
