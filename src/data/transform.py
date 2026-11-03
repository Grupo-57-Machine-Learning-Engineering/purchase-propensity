"""Transformação: padroniza nomes e tipa as flags do dataset bruto.

Não remove linhas nem colunas: vazamento (ex.: `saw_checkout`) e seleção de
features são decisões da EDA.
"""

import re

import pandas as pd

ID_COLUMN = "user_id"


def to_snake_case(name: str) -> str:
    """Converte `UserID` em `user_id`; nomes já em snake_case não mudam."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Renomeia as colunas e converte as flags 0/1 em booleanos."""
    df = raw.rename(columns=to_snake_case)
    flags = df.columns.drop(ID_COLUMN)
    invalid = [column for column in flags if not bool(df[column].isin([0, 1]).all())]
    if invalid:
        raise ValueError(f"Colunas com valores fora de 0/1: {invalid}")
    return df.astype(dict.fromkeys(flags, "bool"))
