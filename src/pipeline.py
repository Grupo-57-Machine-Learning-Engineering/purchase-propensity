"""Pipeline de ponta a ponta. Rode com `python -m src.pipeline`.

ETL, treino dos dois modelos e promoção do melhor a `champion` no Model Registry,
na mesma ordem dos estágios do `dvc.yaml`. Existe para rodar onde o DVC não está,
como a imagem Docker, que só tem as dependências de produção. Localmente, prefira
`make dvc`, que pula os estágios que não mudaram.
"""

from src.data.pipeline import run_etl
from src.logger import get_logger
from src.training.baseline import BASELINE
from src.training.experiment import run_experiment
from src.training.random_forest import RANDOM_FOREST
from src.training.registry import promote_best_model

logger = get_logger(__name__)


def run_pipeline() -> str:
    """Roda ETL, treinos e promoção; devolve a versão do modelo `champion`."""
    data_path = run_etl()
    for experiment in (BASELINE, RANDOM_FOREST):
        run_experiment(experiment, data_path=data_path)
    version = promote_best_model()
    logger.info("Pipeline concluído: champion na versão %s", version)
    return version


if __name__ == "__main__":
    run_pipeline()
