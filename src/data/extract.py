"""Extração: baixa o dataset do Kaggle e copia o CSV bruto para data/raw."""

import logging
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from src.config import DATASET_HANDLE, RAW_DATA_FILE, RAW_DATA_FILENAME

logger = logging.getLogger(__name__)


def download_raw(
    handle: str = DATASET_HANDLE,
    dest: Path = RAW_DATA_FILE,
    filename: str = RAW_DATA_FILENAME,
) -> Path:
    """Baixa o dataset e copia `filename` do download para `dest`.

    Não baixa de novo se `dest` já existir. Baixa o dataset inteiro porque o
    download de arquivo único do kagglehub entrega um zip com extensão .csv.
    A cópia passa por um arquivo temporário e é finalizada com `replace`
    (atômico no mesmo disco), então uma interrupção nunca deixa um CSV
    truncado em `dest`.
    """
    if dest.exists():
        logger.info("Dataset bruto já existe em %s", dest)
        return dest
    logger.info("Baixando %s do Kaggle", handle)
    dataset_dir = Path(kagglehub.dataset_download(handle))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        shutil.copyfile(dataset_dir / filename, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def read_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto."""
    return pd.read_csv(path)
