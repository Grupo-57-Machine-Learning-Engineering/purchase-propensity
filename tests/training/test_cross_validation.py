"""Testes da cross-validation com dados sintéticos e dois modelos diferentes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.training.cross_validation import CV_SCORING, cross_validate_model
from src.training.split import SplitData, stratified_split


@pytest.fixture
def split_data() -> SplitData:
    """Alvo ligado a `basket_icon_click`, com ruído, para o modelo ter o que aprender."""
    rng = np.random.default_rng(0)
    size = 600
    basket_icon_click = rng.random(size) < 0.5
    noise = rng.random(size) < 0.1
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "basket_icon_click": basket_icon_click,
            "ordered": basket_icon_click ^ noise,
        }
    )
    return stratified_split(df, exclude=())


@pytest.mark.parametrize(
    "model",
    [LogisticRegression(max_iter=200), HistGradientBoostingClassifier(max_iter=20)],
)
def test_cross_validate_model_returns_mean_and_std_per_metric(
    model: BaseEstimator, split_data: SplitData
) -> None:
    result = cross_validate_model(model, split_data, folds=5)

    assert set(result.metrics) == {
        f"cv_{name}_{stat}" for name in CV_SCORING for stat in ("mean", "std")
    }
    for name in CV_SCORING:
        assert 0.0 <= result.metrics[f"cv_{name}_mean"] <= 1.0
        assert result.metrics[f"cv_{name}_std"] >= 0.0


def test_cross_validate_model_does_not_change_input_model(split_data: SplitData) -> None:
    model = LogisticRegression(max_iter=200)

    cross_validate_model(model, split_data, folds=5)

    assert not hasattr(model, "coef_")


def test_cross_validate_model_is_deterministic(split_data: SplitData) -> None:
    first = cross_validate_model(LogisticRegression(max_iter=200), split_data, folds=5, seed=1)
    second = cross_validate_model(LogisticRegression(max_iter=200), split_data, folds=5, seed=1)

    assert first.metrics == second.metrics
