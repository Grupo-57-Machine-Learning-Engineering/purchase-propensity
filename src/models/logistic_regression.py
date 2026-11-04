"""Regressão logística, o baseline do projeto."""

from typing import Any

from sklearn.linear_model import LogisticRegression

from src.config import LOGISTIC_REGRESSION_PARAMS, RANDOM_SEED


def build_logistic_regression(
    seed: int = RANDOM_SEED, params: dict[str, Any] | None = None
) -> LogisticRegression:
    """Regressão logística com peso balanceado, por causa dos 4,2% de positivos.

    Os hiperparâmetros vêm de `configs/params.yaml`; `params` os substitui.
    """
    return LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        **(LOGISTIC_REGRESSION_PARAMS if params is None else params),
    )


def describe_logistic_regression(model: LogisticRegression) -> dict[str, str | int | float]:
    """Hiperparâmetros que identificam o modelo na run do MLflow."""
    params = model.get_params()
    return {
        "C": params["C"],
        "class_weight": str(params["class_weight"]),
        "max_iter": params["max_iter"],
    }
