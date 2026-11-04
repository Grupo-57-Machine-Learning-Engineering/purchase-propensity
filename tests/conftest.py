"""Fixtures compartilhadas por todos os testes."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200 usuários, 10% de positivos, com as três colunas de vazamento."""
    size = 200
    ordered = [i % 10 == 0 for i in range(size)]
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "sign_in": [i % 3 == 0 for i in range(size)],
            "basket_icon_click": [i % 2 == 0 for i in range(size)],
            "saw_checkout": ordered,
            "checked_delivery_detail": ordered,
            "ordered": ordered,
        }
    )


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


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """Aponta o MLflow para uma pasta temporária, nunca para o servidor do .env."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(uri)
    mlflow.set_tracking_uri(None)
