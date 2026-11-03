"""Testes do split estratificado e da seleção de features."""

import pandas as pd
import pytest

from src.training.split import select_features, stratified_split


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200 usuários, 10% de positivos, com as duas colunas de vazamento."""
    size = 200
    ordered = [i % 10 == 0 for i in range(size)]
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(size)],
            "sign_in": [i % 3 == 0 for i in range(size)],
            "basket_icon_click": [i % 2 == 0 for i in range(size)],
            "saw_checkout": ordered,
            "checked_delivery_detail": ordered,
            "ordered": ordered,
        }
    )


def test_select_features_drops_id_target_and_leakage(synthetic_df: pd.DataFrame) -> None:
    features = select_features(synthetic_df)

    assert list(features.columns) == ["basket_icon_click"]


def test_select_features_accepts_custom_exclusion(synthetic_df: pd.DataFrame) -> None:
    features = select_features(synthetic_df, exclude=("saw_checkout", "checked_delivery_detail"))

    assert "saw_checkout" not in features.columns


def test_select_features_fails_on_missing_excluded_column(synthetic_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not_a_column"):
        select_features(synthetic_df, exclude=("not_a_column",))


def test_split_keeps_class_ratio(synthetic_df: pd.DataFrame) -> None:
    data = stratified_split(synthetic_df, test_size=0.25)

    assert len(data.features_test) == 50
    assert data.target_train.mean() == pytest.approx(0.1)
    assert data.target_test.mean() == pytest.approx(0.1)


def test_split_is_reproducible_with_same_seed(synthetic_df: pd.DataFrame) -> None:
    first = stratified_split(synthetic_df, seed=7)
    second = stratified_split(synthetic_df, seed=7)

    assert first.features_test.index.equals(second.features_test.index)


def test_split_changes_with_different_seed(synthetic_df: pd.DataFrame) -> None:
    first = stratified_split(synthetic_df, seed=1)
    second = stratified_split(synthetic_df, seed=2)

    assert not first.features_test.index.equals(second.features_test.index)


def test_split_has_no_row_overlap_and_target_is_int(synthetic_df: pd.DataFrame) -> None:
    data = stratified_split(synthetic_df)

    assert data.features_train.index.intersection(data.features_test.index).empty
    assert data.target_train.dtype == int
