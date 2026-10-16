"""Testes do logger centralizado."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from src.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_get_logger_devolve_logger_com_o_nome_pedido() -> None:
    assert get_logger("src.teste").name == "src.teste"


def test_get_logger_nao_duplica_handlers() -> None:
    get_logger("src.primeiro")
    handlers_antes = len(logging.getLogger().handlers)

    get_logger("src.segundo")

    assert len(logging.getLogger().handlers) == handlers_antes


def test_mensagem_sai_no_formato_padrao() -> None:
    """Roda num processo limpo, onde o logging ainda não foi configurado."""
    codigo = "from src.logger import get_logger; get_logger('mod').info('oi')"

    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=PROJECT_ROOT,
        env={**os.environ, "LOG_LEVEL": "INFO"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert " | INFO     | mod | oi" in resultado.stderr
