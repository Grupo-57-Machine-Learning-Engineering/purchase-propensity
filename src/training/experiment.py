"""Fluxo de treino comum a todos os modelos: split, treino, cross-validation e MLflow.

Regressão logística e Random Forest seguem o mesmo fluxo e só trocam o estimador
(`src/models/`). Cada modelo vira um `Experiment`, e `run_experiment` faz o resto:
lê o dataset processado, separa treino e teste sem as colunas de vazamento, treina,
valida por cross-validation, grava as métricas para o DVC e loga a run no MLflow.
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from sklearn.base import BaseEstimator

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

ParamValue = str | int | float
T = TypeVar("T")

DURATION_SUFFIX = "_seconds"

logger = get_logger(__name__)


@dataclass(frozen=True)
class Experiment:
    """Um modelo do projeto: como construí-lo, como descrevê-lo e o nome da run."""

    run_name: str
    build_model: Callable[[], BaseEstimator]
    describe_model: Callable[[Any], dict[str, ParamValue]]


def timed(task: Callable[[], T]) -> tuple[T, float]:
    """Executa `task` e devolve o resultado e a duração em segundos."""
    start = time.perf_counter()
    result = task()
    return result, time.perf_counter() - start


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    """Métricas com 4 casas, para o log."""
    return {name: round(value, 4) for name, value in metrics.items()}


def train_and_validate(
    experiment: Experiment, data: SplitData, threshold: float
) -> tuple[BaseEstimator, dict[str, float]]:
    """Treina no holdout e valida por k-fold, medindo o tempo das duas etapas.

    Devolve o modelo treinado e as métricas `train_*`, `test_*`, `cv_*` e as
    durações (`*_seconds`), para comparar os modelos na mesma tabela.
    """
    result, fit_seconds = timed(lambda: fit_and_evaluate(experiment.build_model(), data, threshold))
    cv_result, cv_seconds = timed(lambda: cross_validate_model(experiment.build_model(), data))
    logger.info("Métricas: %s", rounded(result.metrics))
    logger.info("Cross-validation (%d folds): %s", CV_FOLDS, rounded(cv_result.metrics))
    return result.model, {
        **result.metrics,
        **cv_result.metrics,
        "fit_duration_seconds": fit_seconds,
        "cv_duration_seconds": cv_seconds,
        "total_duration_seconds": fit_seconds + cv_seconds,
    }


def quality_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Só as métricas de qualidade: as durações mudam a cada execução."""
    return {name: v for name, v in metrics.items() if not name.endswith(DURATION_SUFFIX)}


def build_params(
    model: BaseEstimator,
    model_params: dict[str, ParamValue],
    data: SplitData,
    threshold: float,
) -> dict[str, ParamValue]:
    """Parâmetros da run: hiperparâmetros do modelo, split, exclusões e dados."""
    return {
        "model": type(model).__name__,
        **model_params,
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "threshold": threshold,
        "excluded_columns": ",".join(LEAKAGE_COLUMNS),
        "n_features": data.features_train.shape[1],
        "n_train": len(data.features_train),
        "n_test": len(data.features_test),
        "cv_folds": CV_FOLDS,
    }


def run_experiment(
    experiment: Experiment,
    data_path: Path = PROCESSED_DATA_FILE,
    run_name: str | None = None,
    threshold: float = DECISION_THRESHOLD,
    metrics_path: Path | None = None,
) -> str:
    """Treina o modelo do `experiment`, grava as métricas e loga a run no MLflow.

    Com `metrics_path`, grava as métricas de qualidade em JSON para o DVC.
    Devolve o `run_id`.
    """
    data = stratified_split(pd.read_parquet(data_path))
    logger.info("Treinando %s em %d linhas", experiment.run_name, len(data.features_train))
    model, metrics = train_and_validate(experiment, data, threshold)
    if metrics_path is not None:
        save_metrics(quality_metrics(metrics), metrics_path)
    run_id = log_run(
        run_name=run_name or experiment.run_name,
        model=model,
        params=build_params(model, experiment.describe_model(model), data, threshold),
        metrics=metrics,
        tags=build_tags(data_path),
        input_example=data.features_train.head(5),
    )
    logger.info("Run registrada no MLflow: %s", run_id)
    return run_id


def main(experiment: Experiment, description: str) -> None:
    """Linha de comando comum: `--run-name`, `--data-path` e `--metrics-out`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-name", default=experiment.run_name, help="Nome da run no MLflow.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_FILE)
    parser.add_argument(
        "--metrics-out", type=Path, help="JSON com as métricas de qualidade (para o DVC)."
    )
    args = parser.parse_args()
    run_experiment(
        experiment, data_path=args.data_path, run_name=args.run_name, metrics_path=args.metrics_out
    )
