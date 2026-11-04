"""Seleção de features: o que o modelo pode ver.

Tira do dataset processado o id, o alvo e as colunas de vazamento. As 3 flags do
fim do funil (`LEAKAGE_COLUMNS` em `src/config.py`) descrevem a compra
acontecendo, não a intenção antes dela; detalhes em `docs/eda-findings.md`.
"""

import pandas as pd

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN
from src.data.transform import ID_COLUMN


def select_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    exclude: tuple[str, ...] = LEAKAGE_COLUMNS,
) -> pd.DataFrame:
    """Devolve só as colunas de entrada: sem id, sem alvo e sem as colunas excluídas.

    Levanta `KeyError` se alguma coluna a excluir não existir, para um erro de
    digitação não deixar uma coluna com vazamento passar em silêncio.
    """
    return df.drop(columns=[ID_COLUMN, target, *exclude])
