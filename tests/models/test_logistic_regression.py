"""Testes da regressão logística (baseline)."""

from src.config import LOGISTIC_REGRESSION_PARAMS
from src.models.logistic_regression import (
    build_logistic_regression,
    describe_logistic_regression,
)


def test_build_logistic_regression_balances_classes_and_fixes_seed() -> None:
    model = build_logistic_regression(seed=3)

    assert model.class_weight == "balanced"
    assert model.random_state == 3


def test_build_logistic_regression_uses_the_configured_params() -> None:
    model = build_logistic_regression()

    assert LOGISTIC_REGRESSION_PARAMS.items() <= model.get_params().items()


def test_describe_logistic_regression_lists_the_hyperparameters() -> None:
    params = describe_logistic_regression(build_logistic_regression())

    assert params == {
        "C": LOGISTIC_REGRESSION_PARAMS["C"],
        "class_weight": "balanced",
        "max_iter": LOGISTIC_REGRESSION_PARAMS["max_iter"],
    }
