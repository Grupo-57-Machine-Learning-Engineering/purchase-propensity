"""Testes da carga em parquet."""

from pathlib import Path

import pandas as pd

from src.data.load import save_processed


def test_save_processed_preserva_tipos(tmp_path: Path) -> None:
    df = pd.DataFrame({"month": pd.Categorical(["Jun"]), "revenue": [True]})

    saved = pd.read_parquet(save_processed(df, tmp_path / "sub" / "dados.parquet"))

    assert isinstance(saved["month"].dtype, pd.CategoricalDtype)
    assert saved["revenue"].dtype == bool
