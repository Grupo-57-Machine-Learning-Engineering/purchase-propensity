"""Testes da comparação de runs por `cv_pr_auc_mean`."""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import LEAKAGE_COLUMNS, MLFLOW_EXPERIMENT_NAME
from src.training.compare import format_table, rank_runs

NO_LEAKAGE = ",".join(LEAKAGE_COLUMNS)


def _log_run(
    client: MlflowClient,
    run_name: str,
    model: str,
    cv_pr_auc_mean: float,
    excluded_columns: str = NO_LEAKAGE,
) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model)
        mlflow.log_param("excluded_columns", excluded_columns)
        mlflow.log_metric("cv_pr_auc_mean", cv_pr_auc_mean)
        mlflow.log_metric("test_pr_auc", cv_pr_auc_mean)


def test_rank_runs_orders_by_cv_pr_auc_mean_descending(local_mlflow: MlflowClient) -> None:
    _log_run(local_mlflow, "pior", "LogisticRegression", 0.3)
    _log_run(local_mlflow, "melhor", "RandomForestClassifier", 0.7)

    runs = rank_runs(local_mlflow)

    assert [run.info.run_name for run in runs] == ["melhor", "pior"]


def test_rank_runs_returns_empty_without_the_experiment(local_mlflow: MlflowClient) -> None:
    assert rank_runs(local_mlflow) == []


def test_rank_runs_skips_failed_runs_even_with_the_best_metric(
    local_mlflow: MlflowClient,
) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="caiu-antes-do-log-model") as failed_run:
        mlflow.log_param("excluded_columns", NO_LEAKAGE)
        mlflow.log_metric("cv_pr_auc_mean", 0.9)
    local_mlflow.set_terminated(failed_run.info.run_id, status="FAILED")
    _log_run(local_mlflow, "concluida", "LogisticRegression", 0.5)

    runs = rank_runs(local_mlflow)

    assert [run.info.run_name for run in runs] == ["concluida"]


def test_rank_runs_skips_runs_trained_with_leakage_columns(local_mlflow: MlflowClient) -> None:
    _log_run(local_mlflow, "com-vazamento", "LogisticRegression", 0.9, excluded_columns="")
    _log_run(local_mlflow, "sem-vazamento", "LogisticRegression", 0.4)

    runs = rank_runs(local_mlflow)

    assert [run.info.run_name for run in runs] == ["sem-vazamento"]


def test_rank_runs_skips_runs_without_the_exclusion_param(local_mlflow: MlflowClient) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="antiga-sem-param"):
        mlflow.log_metric("cv_pr_auc_mean", 0.9)
    _log_run(local_mlflow, "atual", "LogisticRegression", 0.4)

    runs = rank_runs(local_mlflow)

    assert [run.info.run_name for run in runs] == ["atual"]


def test_format_table_lists_model_and_metrics_per_run(local_mlflow: MlflowClient) -> None:
    _log_run(local_mlflow, "melhor", "RandomForestClassifier", 0.7)

    table = format_table(rank_runs(local_mlflow))

    assert "melhor" in table
    assert "RandomForestClassifier" in table
    assert "0.7000" in table
