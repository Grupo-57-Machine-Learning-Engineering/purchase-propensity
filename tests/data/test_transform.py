"""Testes da transformação do dataset bruto."""

import pandas as pd
import pytest

from src.data.transform import to_snake_case, transform


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("ProductRelated_Duration", "product_related_duration"),
        ("BounceRates", "bounce_rates"),
        ("Revenue", "revenue"),
    ],
)
def test_to_snake_case(raw_name: str, expected: str) -> None:
    assert to_snake_case(raw_name) == expected


def _raw_sample(month: str) -> pd.DataFrame:
    return pd.DataFrame({"Month": [month], "VisitorType": ["New_Visitor"], "Revenue": [True]})


def test_transform_padroniza_june_e_ordena_meses() -> None:
    df = transform(_raw_sample("June"))

    assert list(df.columns) == ["month", "visitor_type", "revenue"]
    assert df["month"].iloc[0] == "Jun"
    assert df["month"].cat.ordered


def test_transform_recusa_mes_desconhecido() -> None:
    with pytest.raises(ValueError, match="Mês fora do padrão"):
        transform(_raw_sample("Junho"))
