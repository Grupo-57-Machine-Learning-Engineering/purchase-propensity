"""Logger centralizado: formato e nível únicos para todos os módulos.

Uso: `logger = get_logger(__name__)`. O nível vem de `LOG_LEVEL` (ver `src/config.py`).
"""

import logging

from src.config import LOG_LEVEL

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Devolve o logger de `name`, garantindo a configuração global do logging.

    O `basicConfig` só age enquanto o root logger não tem handlers, então chamar
    `get_logger` em vários módulos não duplica handlers nem as mensagens.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    return logging.getLogger(name)
