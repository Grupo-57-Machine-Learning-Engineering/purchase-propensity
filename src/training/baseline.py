"""Baseline de regressão logística. Rode com `python -m src.training.baseline`.

Lê o dataset processado, separa treino e teste, treina o baseline sem as colunas
de vazamento e grava a run no MLflow. Paths e constantes vêm de `src/config.py`,
e nada aqui depende do DVC: ele só precisa chamar este módulo.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import (
    DECISION_THRESHOLD,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_FILE,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.logger import get_logger
from src.training.split import SplitData, stratified_split
from src.training.tracking import build_tags, log_run
from src.training.train import fit_and_evaluate

DEFAULT_RUN_NAME = "baseline-logreg-balanced-no-leakage"

logger = get_logger(__name__)


def build_baseline(seed: int = RANDOM_SEED) -> LogisticRegression:
    """Regressão logística com peso balanceado, por causa dos 4,2% de positivos."""
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)


def build_params(
    model: LogisticRegression,
    data: SplitData,
    threshold: float,
) -> dict[str, str | int | float]:
    """Parâmetros que descrevem o modelo, o split e os dados da run."""
    return {
        "model": type(model).__name__,
        "C": model.C,
        "class_weight": str(model.class_weight),
        "max_iter": model.max_iter,
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "threshold": threshold,
        "excluded_columns": ",".join(LEAKAGE_COLUMNS),
        "n_features": data.features_train.shape[1],
        "n_train": len(data.features_train),
        "n_test": len(data.features_test),
    }


def run_baseline(
    data_path: Path = PROCESSED_DATA_FILE,
    run_name: str = DEFAULT_RUN_NAME,
    threshold: float = DECISION_THRESHOLD,
) -> str:
    """Treina o baseline, grava a run no MLflow e devolve o `run_id`."""
    data = stratified_split(pd.read_parquet(data_path))
    logger.info("Treinando baseline em %d linhas de treino", len(data.features_train))
    result = fit_and_evaluate(build_baseline(), data, threshold)
    logger.info("Métricas: %s", {name: round(v, 4) for name, v in result.metrics.items()})
    run_id = log_run(
        run_name=run_name,
        model=result.model,
        params=build_params(build_baseline(), data, threshold),
        metrics=result.metrics,
        tags=build_tags(data_path),
        input_example=data.features_train.head(5),
    )
    logger.info("Run registrada no MLflow: %s", run_id)
    return run_id


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina o baseline e loga no MLflow.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Nome da run no MLflow.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_FILE)
    args = parser.parse_args()
    run_baseline(data_path=args.data_path, run_name=args.run_name)


if __name__ == "__main__":
    main()
