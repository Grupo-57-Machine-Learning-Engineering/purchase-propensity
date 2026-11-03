"""Testes do logger centralizado."""

import os
import subprocess
import sys
from pathlib import Path

from src.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_get_logger_devolve_logger_com_o_nome_pedido() -> None:
    assert get_logger("src.teste").name == "src.teste"


def test_mensagem_sai_uma_vez_no_formato_padrao() -> None:
    """Roda num processo limpo, onde o logging ainda não foi configurado.

    No pytest o root logger já tem handlers e o `basicConfig` vira no-op, então
    só num subprocesso a configuração é de fato exercitada. Chamar `get_logger`
    duas vezes garante que a segunda chamada não duplica o handler.
    """
    codigo = (
        "from src.logger import get_logger; get_logger('primeiro'); get_logger('mod').info('oi')"
    )

    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=PROJECT_ROOT,
        env={**os.environ, "LOG_LEVEL": "INFO"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert resultado.stderr.count("oi") == 1
    assert " | INFO     | mod | oi" in resultado.stderr
