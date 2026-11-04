"""Testes do Random Forest."""

from src.config import RANDOM_FOREST_PARAMS
from src.models.random_forest import build_random_forest, describe_random_forest


def test_build_random_forest_balances_classes_and_fixes_seed() -> None:
    model = build_random_forest(seed=3)

    assert model.class_weight == "balanced"
    assert model.random_state == 3


def test_build_random_forest_uses_the_configured_params() -> None:
    model = build_random_forest()

    assert RANDOM_FOREST_PARAMS.items() <= model.get_params().items()


def test_build_random_forest_params_replace_the_configured_ones() -> None:
    model = build_random_forest(params={"min_samples_leaf": 7})

    assert model.min_samples_leaf == 7
    assert model.max_depth is None
    assert model.class_weight == "balanced"


def test_describe_random_forest_logs_unlimited_depth_as_minus_one() -> None:
    params = describe_random_forest(build_random_forest(params={}))

    assert params["max_depth"] == -1
    assert params["max_features"] == "sqrt"
