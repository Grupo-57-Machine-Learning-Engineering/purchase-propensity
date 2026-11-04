"""Smoke tests: garantem que o pacote importa e a config existe.

Mantêm `make test` verde desde o dia 1, mesmo com os módulos ainda vazios.
"""

import subprocess
import sys

import src
from src import config


def test_package_importa() -> None:
    assert src.__version__


def test_project_root_existe() -> None:
    assert config.PROJECT_ROOT.is_dir()


def test_package_forces_utf8_output_for_mlflow_emoji() -> None:
    # Saída redirecionada, como no dvc repro: sem o UTF-8, o Windows usaria cp1252.
    result = subprocess.run(
        [sys.executable, "-c", "import src; print('\U0001f3c3')"],
        capture_output=True,
        check=True,
    )

    assert result.stdout.decode("utf-8").strip() == "\U0001f3c3"
