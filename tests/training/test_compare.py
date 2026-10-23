"""Testes da comparação de runs por `cv_pr_auc_mean`."""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_EXPERIMENT_NAME
from src.training.compare import format_table, rank_runs


def _log_run(client: MlflowClient, run_name: str, model: str, cv_pr_auc_mean: float) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model)
        mlflow.log_metric("cv_pr_auc_mean", cv_pr_auc_mean)
        mlflow.log_metric("test_pr_auc", cv_pr_auc_mean)


def test_rank_runs_orders_by_cv_pr_auc_mean_descending(local_mlflow: MlflowClient) -> None:
    _log_run(local_mlflow, "pior", "LogisticRegression", 0.3)
    _log_run(local_mlflow, "melhor", "RandomForestClassifier", 0.7)

    runs = rank_runs(local_mlflow)

    assert [run.info.run_name for run in runs] == ["melhor", "pior"]


def test_rank_runs_returns_empty_without_the_experiment(local_mlflow: MlflowClient) -> None:
    assert rank_runs(local_mlflow) == []


def test_format_table_lists_model_and_metrics_per_run(local_mlflow: MlflowClient) -> None:
    _log_run(local_mlflow, "melhor", "RandomForestClassifier", 0.7)

    table = format_table(rank_runs(local_mlflow))

    assert "melhor" in table
    assert "RandomForestClassifier" in table
    assert "0.7000" in table
