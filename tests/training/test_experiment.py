"""Testes do fluxo de treino comum, de ponta a ponta com dados sintéticos.

Cada teste roda com os dois modelos do projeto (`BASELINE` e `RANDOM_FOREST`).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import CV_FOLDS, LEAKAGE_COLUMNS
from src.training.baseline import BASELINE
from src.training.experiment import (
    Experiment,
    build_params,
    quality_metrics,
    run_experiment,
    timed,
)
from src.training.random_forest import RANDOM_FOREST
from src.training.split import stratified_split

EXPERIMENTS = pytest.mark.parametrize("experiment", [BASELINE, RANDOM_FOREST], ids=["lr", "rf"])


def test_timed_returns_the_result_and_a_duration() -> None:
    result, seconds = timed(lambda: 42)

    assert result == 42
    assert seconds >= 0.0


def test_quality_metrics_drops_the_durations() -> None:
    metrics = {"test_pr_auc": 0.4, "fit_duration_seconds": 1.2}

    assert quality_metrics(metrics) == {"test_pr_auc": 0.4}


@EXPERIMENTS
def test_build_params_describes_model_split_and_exclusions(
    experiment: Experiment, processed_file: Path
) -> None:
    data = stratified_split(pd.read_parquet(processed_file))
    model = experiment.build_model()

    params = build_params(model, experiment.describe_model(model), data, threshold=0.5)

    assert params["model"] == type(model).__name__
    assert params["class_weight"] == "balanced"
    assert params["n_features"] == 1
    assert params["n_train"] + params["n_test"] == 800
    assert params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)


@EXPERIMENTS
def test_run_experiment_logs_run_with_metrics_and_tags(
    experiment: Experiment, processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_experiment(experiment, data_path=processed_file)

    run = local_mlflow.get_run(run_id)
    assert run.info.run_name == experiment.run_name
    assert 0.0 <= run.data.metrics["test_pr_auc"] <= 1.0
    assert 0.0 <= run.data.metrics["cv_pr_auc_mean"] <= 1.0
    assert run.data.metrics["fit_duration_seconds"] >= 0.0
    assert run.data.metrics["cv_duration_seconds"] >= 0.0
    assert {"autor", "git_sha", "dataset_version"} <= set(run.data.tags)
    assert run.data.params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)
    assert run.data.params["cv_folds"] == str(CV_FOLDS)


def test_run_experiment_accepts_custom_run_name(
    processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_experiment(BASELINE, data_path=processed_file, run_name="baseline-teste")

    assert local_mlflow.get_run(run_id).info.run_name == "baseline-teste"


@EXPERIMENTS
def test_run_experiment_writes_quality_metrics_for_dvc(
    experiment: Experiment, processed_file: Path, local_mlflow: MlflowClient, tmp_path: Path
) -> None:
    metrics_path = tmp_path / "metrics.json"

    run_experiment(experiment, data_path=processed_file, metrics_path=metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["cv_pr_auc_mean"] <= 1.0
    assert "test_pr_auc" in metrics
    assert not any(name.endswith("_seconds") for name in metrics)
