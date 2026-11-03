"""Busca de hiperparâmetros do Random Forest. Rode com `make tune-rf`.

Random search com os mesmos folds e a mesma métrica da cross-validation
(`src/training/cross_validation.py`), então o melhor score aqui é comparável com
o `cv_pr_auc_mean` das runs. A busca não treina o modelo final: os melhores
parâmetros vão para `models.random_forest` em `configs/params.yaml`, e o treino
normal (`make dvc`) usa esses valores. Assim o pipeline continua determinístico e
a busca, que é cara, só roda quando alguém pede.

O resultado fica em `reports/tuning/random_forest.json` (versionado no git) e
numa run do MLflow com tag `kind=search`. Essa run não loga `excluded_columns`,
então não entra no ranking de `src/training/compare.py`: não tem modelo para
promover.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    CV_FOLDS,
    LEAKAGE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.logger import get_logger
from src.models.random_forest import build_random_forest
from src.training.experiment import timed
from src.training.split import SplitData, stratified_split
from src.training.tracking import build_tags

DEFAULT_RUN_NAME = "random-forest-search"
DEFAULT_N_ITER = 30
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "tuning" / "random_forest.json"
SEARCH_SCORING = "average_precision"
TOP_CANDIDATES = 10

# As 20 flags binárias formam só ~4.200 combinações distintas nas 318 mil linhas de
# treino. Árvores sem limite decoram as combinações raras (PR-AUC de 0,50 no treino
# contra 0,37 no teste), então o espaço foca em regularização: folha mínima,
# profundidade e quantas features cada divisão considera.
RF_SEARCH_SPACE: dict[str, Any] = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 6, 8, 10, 12, 16],
    "min_samples_leaf": [1, 5, 10, 25, 50, 100, 200],
    "max_features": ["sqrt", 0.3, 0.5, 0.7, None],
}

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Melhores parâmetros, o score deles e os candidatos ordenados por score."""

    best_params: dict[str, Any]
    best_score: float
    best_score_std: float
    candidates: pd.DataFrame


def search_hyperparameters(
    model: BaseEstimator,
    search_space: dict[str, Any],
    data: SplitData,
    n_iter: int = DEFAULT_N_ITER,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> SearchResult:
    """Random search por PR-AUC no treino de `data`, com k-fold estratificado.

    Os folds usam o mesmo `seed` da cross-validation das runs, então os dois
    scores saem das mesmas partições. Não reajusta o melhor modelo (`refit=False`).
    """
    search = RandomizedSearchCV(
        model,
        search_space,
        n_iter=n_iter,
        scoring=SEARCH_SCORING,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        refit=False,
        random_state=seed,
    ).fit(data.features_train, data.target_train)
    candidates = rank_candidates(search.cv_results_)
    return SearchResult(
        best_params=search.best_params_,
        best_score=float(candidates.at[0, "mean_test_score"]),
        best_score_std=float(candidates.at[0, "std_test_score"]),
        candidates=candidates,
    )


def rank_candidates(cv_results: dict[str, Any]) -> pd.DataFrame:
    """Candidatos da busca, do melhor para o pior score."""
    columns = ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    ranked = pd.DataFrame(cv_results).sort_values("rank_test_score")
    return ranked.loc[:, columns].reset_index(drop=True)


def build_report(result: SearchResult, n_iter: int) -> dict[str, Any]:
    """Resumo da busca: melhores parâmetros e os candidatos do topo."""
    top = result.candidates.head(TOP_CANDIDATES)
    return {
        "scoring": SEARCH_SCORING,
        "cv_folds": CV_FOLDS,
        "n_iter": n_iter,
        "excluded_columns": list(LEAKAGE_COLUMNS),
        "best_params": result.best_params,
        "best_cv_pr_auc_mean": result.best_score,
        "best_cv_pr_auc_std": result.best_score_std,
        "top_candidates": [
            {
                "params": row["params"],
                "cv_pr_auc_mean": float(row["mean_test_score"]),
                "cv_pr_auc_std": float(row["std_test_score"]),
            }
            for _, row in top.iterrows()
        ],
    }


def save_report(report: dict[str, Any], path: Path) -> Path:
    """Grava o resumo da busca em JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def log_search(report: dict[str, Any], report_path: Path, tags: dict[str, str]) -> str:
    """Grava a busca como run do MLflow, sem modelo e fora do ranking."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=DEFAULT_RUN_NAME) as run:
        mlflow.set_tags({**tags, "kind": "search"})
        mlflow.log_params({f"best_{name}": v for name, v in report["best_params"].items()})
        mlflow.log_params({"n_iter": report["n_iter"], "cv_folds": report["cv_folds"]})
        mlflow.log_metrics(
            {
                "best_cv_pr_auc_mean": report["best_cv_pr_auc_mean"],
                "best_cv_pr_auc_std": report["best_cv_pr_auc_std"],
            }
        )
        mlflow.log_artifact(str(report_path))
    return run.info.run_id


def run_search(
    data_path: Path = PROCESSED_DATA_FILE,
    n_iter: int = DEFAULT_N_ITER,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Roda a busca do Random Forest, grava o JSON e loga no MLflow."""
    data = stratified_split(pd.read_parquet(data_path))
    logger.info("Buscando hiperparâmetros: %d candidatos x %d folds", n_iter, CV_FOLDS)

    result, seconds = timed(
        lambda: search_hyperparameters(
            build_random_forest(params={}), RF_SEARCH_SPACE, data, n_iter
        )
    )
    logger.info("Busca concluída em %.0f s; melhores: %s", seconds, result.best_params)

    report = build_report(result, n_iter)
    save_report(report, report_path)
    run_id = log_search(report, report_path, build_tags(data_path))
    logger.info("Busca registrada no MLflow: %s", run_id)
    return report


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(description="Busca hiperparâmetros do Random Forest.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_FILE)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    run_search(data_path=args.data_path, n_iter=args.n_iter, report_path=args.report_out)


if __name__ == "__main__":
    main()
