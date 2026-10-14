"""Extração: baixa o dataset bruto da UCI para data/raw."""

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.config import DATASET_URL, RAW_DATA_FILE

DOWNLOAD_TIMEOUT_SECONDS = 60

logger = logging.getLogger(__name__)


def download_raw(url: str = DATASET_URL, dest: Path = RAW_DATA_FILE) -> Path:
    """Baixa o zip da UCI e grava o CSV em `dest`. Não baixa de novo se já existir."""
    if dest.exists():
        logger.info("Dataset bruto já existe em %s", dest)
        return dest
    logger.info("Baixando dataset de %s", url)
    with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        content = response.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(extract_member(content, dest.name))
    return dest


def extract_member(zip_content: bytes, member: str) -> bytes:
    """Lê um arquivo de dentro de um zip em memória."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as archive:
        return archive.read(member)


def read_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto."""
    return pd.read_csv(path)
