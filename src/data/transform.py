"""Transformação: padroniza nomes de colunas e valores do dataset bruto.

Não remove linhas: o tratamento de duplicadas e outliers é decisão da EDA.
"""

import re

import pandas as pd

# O dataset grava "June" por extenso e os outros meses com 3 letras.
MONTH_FIXES: dict[str, str] = {"June": "Jun"}
MONTH_DTYPE = pd.CategoricalDtype(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ordered=True,
)


def to_snake_case(name: str) -> str:
    """Converte `ProductRelated_Duration` em `product_related_duration`."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para snake_case e tipa as categóricas."""
    df = raw.rename(columns=to_snake_case)
    months = df["month"].replace(MONTH_FIXES)
    unknown = set(months) - set(MONTH_DTYPE.categories)
    if unknown:
        raise ValueError(f"Mês fora do padrão no dataset bruto: {sorted(unknown)}")
    df["month"] = months.astype(MONTH_DTYPE)
    df["visitor_type"] = df["visitor_type"].astype("category")
    return df
