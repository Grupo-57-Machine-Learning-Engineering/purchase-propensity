"""Teste do pipeline de ponta a ponta, sem rede: o ETL é trocado pelo parquet sintético."""

from pathlib import Path

import pytest
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient

from src import pipeline
from src.config import MLFLOW_REGISTERED_MODEL_NAME
from src.training.registry import CHAMPION_ALIAS


def test_run_pipeline_trains_both_models_and_promotes_a_loadable_champion(
    processed_file: Path, local_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "run_etl", lambda: processed_file)

    version = pipeline.run_pipeline()

    runs = local_mlflow.search_runs([local_mlflow.search_experiments()[0].experiment_id])
    assert sorted(run.data.params["model"] for run in runs) == [
        "LogisticRegression",
        "RandomForestClassifier",
    ]
    champion_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    assert hasattr(load_model(champion_uri), "predict_proba")
    champion = local_mlflow.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == version
