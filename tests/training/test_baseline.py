"""Testes do baseline de regressão logística, de ponta a ponta com dados sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import LEAKAGE_COLUMNS
from src.training.baseline import DEFAULT_RUN_NAME, build_baseline, build_params, run_baseline
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


def test_build_baseline_balances_classes_and_fixes_seed() -> None:
    model = build_baseline(seed=3)

    assert model.class_weight == "balanced"
    assert model.random_state == 3


def test_build_params_describes_model_split_and_exclusions(processed_file: Path) -> None:
    data = stratified_split(pd.read_parquet(processed_file))

    params = build_params(build_baseline(), data, threshold=0.5)

    assert params["model"] == "LogisticRegression"
    assert params["n_features"] == 1
    assert params["n_train"] + params["n_test"] == 800
    assert params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)


def test_run_baseline_logs_run_with_metrics_and_tags(
    processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_baseline(data_path=processed_file)

    run = local_mlflow.get_run(run_id)
    assert run.info.run_name == DEFAULT_RUN_NAME
    assert 0.0 <= run.data.metrics["test_pr_auc"] <= 1.0
    assert {"autor", "git_sha", "dataset_version"} <= set(run.data.tags)
    assert run.data.params["excluded_columns"] == ",".join(LEAKAGE_COLUMNS)


def test_run_baseline_accepts_custom_run_name(
    processed_file: Path, local_mlflow: MlflowClient
) -> None:
    run_id = run_baseline(data_path=processed_file, run_name="baseline-teste")

    assert local_mlflow.get_run(run_id).info.run_name == "baseline-teste"
