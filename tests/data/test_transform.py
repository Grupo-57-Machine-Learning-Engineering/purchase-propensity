"""Testes da transformação do dataset bruto."""

import pandas as pd
import pytest

from src.data.transform import to_snake_case, transform


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [("UserID", "user_id"), ("basket_icon_click", "basket_icon_click")],
)
def test_to_snake_case(raw_name: str, expected: str) -> None:
    assert to_snake_case(raw_name) == expected


def test_transform_renomeia_id_e_converte_flags() -> None:
    raw = pd.DataFrame({"UserID": ["a1"], "sign_in": [1], "ordered": [0]})

    df = transform(raw)

    assert list(df.columns) == ["user_id", "sign_in", "ordered"]
    assert df["sign_in"].dtype == bool
    assert not df["ordered"].iloc[0]


def test_transform_recusa_flag_fora_de_0_1() -> None:
    raw = pd.DataFrame({"UserID": ["a1"], "sign_in": [2]})

    with pytest.raises(ValueError, match="sign_in"):
        transform(raw)
