"""Fixtures compartilhadas dos testes de treino."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """Aponta o MLflow para uma pasta temporária, nunca para o servidor do .env."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(uri)
    mlflow.set_tracking_uri(None)
