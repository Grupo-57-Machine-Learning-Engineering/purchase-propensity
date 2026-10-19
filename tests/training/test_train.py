"""Testes da função de treino com dados sintéticos e dois modelos diferentes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from src.training.split import SplitData, stratified_split
from src.training.train import fit_and_evaluate


@pytest.fixture
def split_data() -> SplitData:
    """Alvo ligado a `basket_icon_click`, com ruído, para o modelo ter o que aprender.

    `sign_in` entra só como ruído: é coluna de vazamento e fica fora do treino.
    """
    rng = np.random.default_rng(0)
    size = 600
    basket_icon_click = rng.random(size) < 0.5
    noise = rng.random(size) < 0.1
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "sign_in": rng.random(size) < 0.5,
            "basket_icon_click": basket_icon_click,
            "saw_checkout": basket_icon_click,
            "checked_delivery_detail": basket_icon_click,
            "ordered": basket_icon_click ^ noise,
        }
    )
    return stratified_split(df)


@pytest.mark.parametrize(
    "model",
    [LogisticRegression(max_iter=200), HistGradientBoostingClassifier(max_iter=20)],
)
def test_fit_and_evaluate_works_with_any_classifier(
    model: BaseEstimator, split_data: SplitData
) -> None:
    result = fit_and_evaluate(model, split_data)

    assert result.metrics["test_pr_auc"] > 0.5
    assert set(result.metrics) == {
        f"{split}_{name}"
        for split in ("train", "test")
        for name in ("pr_auc", "roc_auc", "precision", "recall")
    }


def test_fit_and_evaluate_does_not_change_input_model(split_data: SplitData) -> None:
    model = LogisticRegression(max_iter=200)

    result = fit_and_evaluate(model, split_data)

    check_is_fitted(result.model)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_fit_and_evaluate_is_deterministic(split_data: SplitData) -> None:
    first = fit_and_evaluate(LogisticRegression(max_iter=200), split_data)
    second = fit_and_evaluate(LogisticRegression(max_iter=200), split_data)

    assert first.metrics == second.metrics
