"""Random Forest. Rode com `python -m src.training.random_forest`.

Mesmo fluxo do baseline (`src/training/baseline.py`): lê o dataset processado,
separa treino e teste, treina sem as colunas de vazamento, valida por
cross-validation e grava a run no MLflow. A ideia é comparar com a regressão
logística pela mesma tabela de métricas (`src/training/compare.py`).
"""

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    CV_FOLDS,
    DECISION_THRESHOLD,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_FILE,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.logger import get_logger
from src.training.cross_validation import cross_validate_model
from src.training.metrics import save_metrics
from src.training.split import SplitData, stratified_split
from src.training.tracking import build_tags, log_run
from src.training.train import fit_and_evaluate

DEFAULT_RUN_NAME = "random-forest-tuned-no-leakage"

# Melhores parâmetros da busca (`make tune-rf`, detalhes em
# reports/tuning/random_forest.json). Ficam fixos aqui para o treino do pipeline
# (`make dvc`) não depender da busca, que é cara. Rodou a busca de novo? Copie o
# `best_params` do JSON para cá.
TUNED_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "max_features": 0.3,
}

logger = get_logger(__name__)


def build_random_forest(
    seed: int = RANDOM_SEED, params: dict[str, Any] | None = None
) -> RandomForestClassifier:
    """Random Forest com peso balanceado (4,2% de positivos) e os parâmetros da busca.

    `params` substitui `TUNED_PARAMS`; a busca passa `{}` para partir dos padrões
    do scikit-learn.
    """
    return RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        **(TUNED_PARAMS if params is None else params),
    )


def build_params(
    model: RandomForestClassifier,
    data: SplitData,
    threshold: float,
) -> dict[str, str | int | float]:
    """Parâmetros que descrevem o modelo, o split e os dados da run."""
    return {
        "model": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth or -1,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": str(model.max_features),
        "class_weight": str(model.class_weight),
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "threshold": threshold,
        "excluded_columns": ",".join(LEAKAGE_COLUMNS),
        "n_features": data.features_train.shape[1],
        "n_train": len(data.features_train),
        "n_test": len(data.features_test),
        "cv_folds": CV_FOLDS,
    }


def run_random_forest(
    data_path: Path = PROCESSED_DATA_FILE,
    run_name: str = DEFAULT_RUN_NAME,
    threshold: float = DECISION_THRESHOLD,
    metrics_path: Path | None = None,
) -> str:
    """Treina o Random Forest, valida por cross-validation e grava a run no MLflow."""
    data = stratified_split(pd.read_parquet(data_path))
    logger.info("Treinando random forest em %d linhas de treino", len(data.features_train))

    fit_start = time.perf_counter()
    result = fit_and_evaluate(build_random_forest(), data, threshold)
    fit_duration = time.perf_counter() - fit_start
    logger.info("Métricas: %s", {name: round(v, 4) for name, v in result.metrics.items()})

    cv_start = time.perf_counter()
    cv_result = cross_validate_model(build_random_forest(), data, folds=CV_FOLDS)
    cv_duration = time.perf_counter() - cv_start
    logger.info(
        "Cross-validation (%d folds): %s",
        CV_FOLDS,
        {name: round(v, 4) for name, v in cv_result.metrics.items()},
    )

    if metrics_path is not None:
        # Só métricas de qualidade: os tempos mudam a cada execução e poluiriam o diff.
        save_metrics({**result.metrics, **cv_result.metrics}, metrics_path)

    metrics = {
        **result.metrics,
        **cv_result.metrics,
        "fit_duration_seconds": fit_duration,
        "cv_duration_seconds": cv_duration,
        "total_duration_seconds": fit_duration + cv_duration,
    }
    run_id = log_run(
        run_name=run_name,
        model=result.model,
        params=build_params(build_random_forest(), data, threshold),
        metrics=metrics,
        tags=build_tags(data_path),
        input_example=data.features_train.head(5),
    )
    logger.info("Run registrada no MLflow: %s", run_id)
    return run_id


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina o random forest e loga no MLflow.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Nome da run no MLflow.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_FILE)
    parser.add_argument(
        "--metrics-out", type=Path, help="JSON com as métricas de qualidade (para o DVC)."
    )
    args = parser.parse_args()
    run_random_forest(
        data_path=args.data_path, run_name=args.run_name, metrics_path=args.metrics_out
    )


if __name__ == "__main__":
    main()
