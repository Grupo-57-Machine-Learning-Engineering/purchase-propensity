"""Métricas de classificação binária, independentes do modelo.

A métrica principal é a PR-AUC (`average_precision`), porque o alvo é muito
desbalanceado (cerca de 4,2% de positivos) e a acurácia engana. ROC-AUC entra como
métrica secundária; precisão e recall dependem do threshold escolhido.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.config import DECISION_THRESHOLD


def compute_metrics(
    target_true: ArrayLike,
    proba: ArrayLike,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Calcula PR-AUC, ROC-AUC, precisão e recall a partir das probabilidades.

    PR-AUC e ROC-AUC não dependem do threshold. Precisão e recall usam
    `proba >= threshold` como previsão positiva; sem nenhum positivo previsto, a
    precisão vale 0. Levanta `ValueError` se `target_true` tiver uma classe só.
    """
    if len(np.unique(np.asarray(target_true))) < 2:
        raise ValueError("target_true precisa ter as duas classes para calcular as métricas")
    predicted = np.asarray(proba) >= threshold
    return {
        "pr_auc": float(average_precision_score(target_true, proba)),
        "roc_auc": float(roc_auc_score(target_true, proba)),
        "precision": float(precision_score(target_true, predicted, zero_division=0)),
        "recall": float(recall_score(target_true, predicted, zero_division=0)),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Põe `prefix_` na frente de cada nome, ex.: `pr_auc` vira `test_pr_auc`."""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}
