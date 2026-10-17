"""Pipeline de ETL: extract -> transform -> load. Rode com `make etl`."""

from pathlib import Path

from src.data.extract import download_raw, read_raw
from src.data.load import save_processed
from src.data.transform import transform
from src.logger import get_logger

logger = get_logger(__name__)


def run_etl() -> Path:
    """Executa o ETL completo e devolve o caminho do dataset processado."""
    processed = transform(read_raw(download_raw()))
    output = save_processed(processed)
    logger.info("ETL concluído: %d linhas em %s", len(processed), output)
    return output


if __name__ == "__main__":
    run_etl()
