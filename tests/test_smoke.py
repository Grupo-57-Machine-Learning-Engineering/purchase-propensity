"""Smoke tests: garantem que o pacote importa e a config existe.

Mantêm `make test` verde desde o dia 1, mesmo com os módulos ainda vazios.
"""

import src
from src import config


def test_package_importa() -> None:
    assert src.__version__


def test_project_root_existe() -> None:
    assert config.PROJECT_ROOT.is_dir()
