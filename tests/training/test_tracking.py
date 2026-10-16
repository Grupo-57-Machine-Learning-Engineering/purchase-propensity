"""Testes do registro no MLflow, sempre num banco SQLite temporário."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.config import MLFLOW_EXPERIMENT_NAME
from src.training.tracking import build_tags, get_dataset_version, log_run


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """Aponta o MLflow para uma pasta temporária, nunca para o servidor do .env."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(uri)
    mlflow.set_tracking_uri(None)


def test_dataset_version_is_stable_and_follows_content(tmp_path: Path) -> None:
    file = tmp_path / "data.parquet"
    file.write_bytes(b"abc")
    first = get_dataset_version(file)

    assert get_dataset_version(file) == first
    file.write_bytes(b"abd")
    assert get_dataset_version(file) != first
    assert len(first) == 12


def test_build_tags_has_the_three_expected_keys(tmp_path: Path) -> None:
    file = tmp_path / "data.parquet"
    file.write_bytes(b"abc")

    tags = build_tags(file)

    assert set(tags) == {"autor", "git_sha", "dataset_version"}
    assert all(tags.values())


def test_log_run_records_params_metrics_tags_and_model(local_mlflow: MlflowClient) -> None:
    features = pd.DataFrame({"sign_in": [True, False, True, False]})
    model = LogisticRegression().fit(features, [1, 0, 1, 0])

    run_id = log_run(
        run_name="test-run",
        model=model,
        params={"C": 1.0},
        metrics={"test_pr_auc": 0.5},
        tags={"autor": "tester"},
        input_example=features,
    )

    run = local_mlflow.get_run(run_id)
    experiment = local_mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None
    assert run.info.experiment_id == experiment.experiment_id
    assert run.info.run_name == "test-run"
    assert run.data.params["C"] == "1.0"
    assert run.data.metrics["test_pr_auc"] == 0.5
    assert run.data.tags["autor"] == "tester"
    logged_models = local_mlflow.search_logged_models([experiment.experiment_id])
    assert [logged.name for logged in logged_models] == ["model"]
