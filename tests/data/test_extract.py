"""Testes da extração, sem rede: o kagglehub é substituído por um diretório local."""

from pathlib import Path

import pytest

from src.data import extract


def test_download_raw_copia_csv_do_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "kaggle"
    dataset_dir.mkdir()
    (dataset_dir / "dados.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(extract.kagglehub, "dataset_download", lambda handle: str(dataset_dir))

    dest = extract.download_raw(handle="dono/dataset", dest=tmp_path / "raw" / "dados.csv")

    assert dest.read_text() == "a,b\n1,2\n"


def test_download_raw_nao_baixa_se_arquivo_existe(tmp_path: Path) -> None:
    dest = tmp_path / "dados.csv"
    dest.write_text("a,b\n")

    assert extract.download_raw(handle="handle/inexistente", dest=dest) == dest
