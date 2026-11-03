"""Testes da seleção de features."""

import pandas as pd
import pytest

from src.features.selection import select_features


def test_select_features_drops_id_target_and_leakage(synthetic_df: pd.DataFrame) -> None:
    features = select_features(synthetic_df)

    assert list(features.columns) == ["basket_icon_click"]


def test_select_features_accepts_custom_exclusion(synthetic_df: pd.DataFrame) -> None:
    features = select_features(synthetic_df, exclude=("saw_checkout", "checked_delivery_detail"))

    assert "saw_checkout" not in features.columns


def test_select_features_fails_on_missing_excluded_column(synthetic_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not_a_column"):
        select_features(synthetic_df, exclude=("not_a_column",))
