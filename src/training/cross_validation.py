"""Validação cruzada estratificada, independente do modelo.

Complementa o holdout único de `fit_and_evaluate`: média e desvio padrão entre
folds dão uma estimativa mais estável para comparar modelos (regressão
logística, random forest, HistGB) na mesma tabela.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CV_FOLDS, RANDOM_SEED
from src.training.split import SplitData

# PR-AUC e ROC-AUC usam probabilidade, não dependem de threshold.
CV_SCORING: dict[str, str] = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


@dataclass(frozen=True)
class CrossValResult:
    """Média e desvio padrão de cada métrica entre os folds (`cv_<nome>_mean/std`)."""

    metrics: dict[str, float]


def cross_validate_model(
    model: BaseEstimator,
    data: SplitData,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> CrossValResult:
    """Valida `model` por k-fold estratificado no conjunto de treino de `data`.

    `folds` não pode passar do número de exemplos da classe minoritária no
    treino, senão o `StratifiedKFold` levanta erro (algum fold ficaria sem
    nenhum positivo).
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        clone(model),
        data.features_train,
        data.target_train,
        cv=splitter,
        scoring=CV_SCORING,
    )
    metrics = {
        f"cv_{name}_{stat}": float(fn(scores[f"test_{name}"]))
        for name in CV_SCORING
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }
    return CrossValResult(metrics)
