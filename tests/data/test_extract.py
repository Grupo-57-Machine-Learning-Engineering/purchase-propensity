"""Testes da extração, sem rede: o zip é montado em memória."""

import io
import zipfile
from pathlib import Path

from src.data.extract import download_raw, extract_member


def test_extract_member_le_arquivo_do_zip() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dados.csv", "a,b\n1,2\n")

    assert extract_member(buffer.getvalue(), "dados.csv") == b"a,b\n1,2\n"


def test_download_raw_nao_baixa_se_arquivo_existe(tmp_path: Path) -> None:
    dest = tmp_path / "dados.csv"
    dest.write_text("a,b\n")

    assert download_raw(url="http://invalido", dest=dest) == dest
