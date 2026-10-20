"""Função de treino genérica: serve para qualquer estimador do scikit-learn.

Recebe o modelo já configurado e o `SplitData`, então regressão logística,
random forest e HistGB reaproveitam o mesmo fluxo e as mesmas métricas.
"""

from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone

from src.config import DECISION_THRESHOLD
from src.training.metrics import compute_metrics, prefix_metrics
from src.training.split import SplitData


@dataclass(frozen=True)
class TrainResult:
    """Modelo treinado e as métricas de treino e de teste (`train_*` e `test_*`)."""

    model: BaseEstimator
    metrics: dict[str, float]


def fit_and_evaluate(
    model: BaseEstimator,
    data: SplitData,
    threshold: float = DECISION_THRESHOLD,
) -> TrainResult:
    """Treina uma cópia de `model` no treino e mede nos dois conjuntos.

    O `model` recebido não é alterado. O modelo precisa ter `predict_proba`.
    """
    fitted = clone(model).fit(data.features_train, data.target_train)
    train_proba = fitted.predict_proba(data.features_train)[:, 1]
    test_proba = fitted.predict_proba(data.features_test)[:, 1]
    metrics = {
        **prefix_metrics(compute_metrics(data.target_train, train_proba, threshold), "train"),
        **prefix_metrics(compute_metrics(data.target_test, test_proba, threshold), "test"),
    }
    return TrainResult(fitted, metrics)
