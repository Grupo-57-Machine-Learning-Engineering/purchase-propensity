"""Testes da leitura de configuração pelo ambiente (.env)."""

from pathlib import Path

import pytest

from src import config

VAR = "PURCHASE_PROPENSITY_TESTE"


def test_env_usa_padrao_se_variavel_ausente(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VAR, raising=False)

    assert config.env(VAR, "padrao") == "padrao"


def test_env_usa_padrao_se_variavel_vazia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, "")

    assert config.env(VAR, "padrao") == "padrao"


def test_env_usa_valor_definido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, "definido")

    assert config.env(VAR, "padrao") == "definido"


def test_path_from_env_resolve_relativo_na_raiz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, "outra/pasta")

    assert config.path_from_env(VAR, "data") == config.PROJECT_ROOT / "outra" / "pasta"


def test_path_from_env_aceita_caminho_absoluto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(VAR, str(tmp_path))

    assert config.path_from_env(VAR, "data") == tmp_path
