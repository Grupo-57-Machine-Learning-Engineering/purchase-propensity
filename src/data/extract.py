"""Extração: baixa o dataset do Kaggle e copia o CSV bruto para data/raw."""

import logging
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from src.config import DATASET_HANDLE, RAW_DATA_FILE

logger = logging.getLogger(__name__)


def download_raw(handle: str = DATASET_HANDLE, dest: Path = RAW_DATA_FILE) -> Path:
    """Baixa o dataset e copia para `dest` o arquivo de mesmo nome.

    Não baixa de novo se `dest` já existir. Baixa o dataset inteiro porque o
    download de arquivo único do kagglehub entrega um zip com extensão .csv.
    """
    if dest.exists():
        logger.info("Dataset bruto já existe em %s", dest)
        return dest
    logger.info("Baixando %s do Kaggle", handle)
    dataset_dir = Path(kagglehub.dataset_download(handle))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset_dir / dest.name, dest)
    return dest


def read_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto."""
    return pd.read_csv(path)
