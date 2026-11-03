"""Testes da busca de hiperparâmetros, com dados sintéticos e espaço pequeno."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from src.config import LEAKAGE_COLUMNS
from src.training import tuning
from src.training.compare import rank_runs
from src.training.split import stratified_split
from src.training.tuning import build_report, run_search, search_hyperparameters

SMALL_SPACE = {"n_estimators": [5, 10], "min_samples_leaf": [1, 20]}


@pytest.fixture
def processed_file(tmp_path: Path) -> Path:
    """Parquet sintético no mesmo formato do `make etl`, com ~10% de positivos."""
    rng = np.random.default_rng(0)
    size = 800
    basket = rng.random(size) < 0.5
    ordered = basket & (rng.random(size) < 0.2)
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "sign_in": ordered,
            "basket_icon_click": basket,
            "returning_user": rng.random(size) < 0.5,
            "saw_checkout": ordered,
            "checked_delivery_detail": ordered,
            "ordered": ordered,
        }
    )
    path = tmp_path / "customer_propensity.parquet"
    df.to_parquet(path, index=False)
    return path


def test_search_returns_best_params_from_the_space_and_sorted_candidates(
    processed_file: Path,
) -> None:
    data = stratified_split(pd.read_parquet(processed_file))

    result = search_hyperparameters(
        RandomForestClassifier(random_state=0), SMALL_SPACE, data, n_iter=4, folds=3
    )

    assert result.best_params["n_estimators"] in SMALL_SPACE["n_estimators"]
    assert result.best_params["min_samples_leaf"] in SMALL_SPACE["min_samples_leaf"]
    scores = result.candidates["mean_test_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert result.best_score == scores[0]


def test_build_report_is_json_serializable_and_keeps_the_top(processed_file: Path) -> None:
    data = stratified_split(pd.read_parquet(processed_file))
    result = search_hyperparameters(
        RandomForestClassifier(random_state=0), SMALL_SPACE, data, n_iter=4, folds=3
    )

    report = json.loads(json.dumps(build_report(result, n_iter=4)))

    assert report["best_params"] == result.best_params
    assert report["excluded_columns"] == list(LEAKAGE_COLUMNS)
    assert len(report["top_candidates"]) == 4
    assert report["top_candidates"][0]["params"] == result.best_params


def test_run_search_writes_report_and_logs_a_run_outside_the_ranking(
    processed_file: Path,
    local_mlflow: MlflowClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tuning, "RF_SEARCH_SPACE", SMALL_SPACE)
    report_path = tmp_path / "tuning.json"

    report = run_search(data_path=processed_file, n_iter=2, report_path=report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    runs = local_mlflow.search_runs(local_mlflow.search_experiments()[0].experiment_id)
    assert [run.data.tags["kind"] for run in runs] == ["search"]
    assert "excluded_columns" not in runs[0].data.params
    assert rank_runs(local_mlflow) == []
