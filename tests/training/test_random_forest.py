"""Testes do random forest, de ponta a ponta com dados sintéticos."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import CV_FOLDS, LEAKAGE_COLUMNS
from src.training.random_forest import (
    DEFAULT_RUN_NAME,
    TUNED_PARAMS,
    build_params,
    build_random_forest,
    run_random_forest,
)
from src.training.split import stratified_split


@pytest.fixture
def processed_file(tmp_path: Path) -> Path:
    """Parquet sintético no mesmo formato do `make etl`, com ~10% de positivos."""
    rng = np.random.default_rng(0)
    size = 800
    sign_in = rng.random(size) < 0.5
    ordered = sign_in & (rng.random(size) < 0.2)
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "sign_in": sign_in,
            "basket_icon_click": rng.random(size) < 0.5,
            "saw_checkout": ordered,
            "checked_delivery_detail": ordered,
            "ordered": ordered,
        }
    )
    path = tmp_path / "customer_propensity.parquet"
    df.to_parquet(path, index=False)
    return path


def test_build_random_forest_balances_classes_and_fixes_seed() -> None:
    model = build_random_forest(seed=3)

    assert model.class_weight == "balanced"
    assert model.random_state == 3


def test_build_random_forest_uses_tuned_params_by_default() -> None:
    model = build_random_forest()

    assert TUNED_PARAMS.items() <= model.get_params().items()


def test_build_random_forest_params_replace_the_tuned_ones() -> None:
    model = build_random_forest(params={"min_samples_leaf": 7})

    assert model.min_samples_leaf == 7
    assert model.class_weight == "balanced"


def test_build_params_describes_model_split_and_exclusions(processed_file: Path) -> None:
    data = stratified_split(pd.read_parquet(processed_file))
    model = build_random_forest()

    params = build_params(model, data, threshold=0.5)

    assert params["model"] == "RandomForestClassifier"
    assert params["min_samples_leaf"] == model.min_samples_leaf
    assert params["max_features"] == str(model.max_features)
    assert params["n_features"] == 1
    assert params["n_train"] + params["n_test"] == 800
    assert params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)


def test_run_random_forest_logs_run_with_metrics_and_tags(
    processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_random_forest(data_path=processed_file)

    run = local_mlflow.get_run(run_id)
    assert run.info.run_name == DEFAULT_RUN_NAME
    assert 0.0 <= run.data.metrics["test_pr_auc"] <= 1.0
    assert 0.0 <= run.data.metrics["cv_pr_auc_mean"] <= 1.0
    assert run.data.metrics["fit_duration_seconds"] >= 0.0
    assert run.data.metrics["cv_duration_seconds"] >= 0.0
    assert {"autor", "git_sha", "dataset_version"} <= set(run.data.tags)
    assert run.data.params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)
    assert run.data.params["cv_folds"] == str(CV_FOLDS)


def test_run_random_forest_accepts_custom_run_name(
    processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_random_forest(data_path=processed_file, run_name="random-forest-teste")

    assert local_mlflow.get_run(run_id).info.run_name == "random-forest-teste"


def test_run_random_forest_writes_quality_metrics_for_dvc(
    processed_file: Path, local_mlflow: MlflowClient, tmp_path: Path
) -> None:
    metrics_path = tmp_path / "metrics.json"

    run_random_forest(data_path=processed_file, metrics_path=metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["cv_pr_auc_mean"] <= 1.0
    assert "test_pr_auc" in metrics
    assert not any(name.endswith("_seconds") for name in metrics)
