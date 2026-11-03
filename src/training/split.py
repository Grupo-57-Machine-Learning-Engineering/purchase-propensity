"""Split estratificado e seleção de features, independentes do modelo.

Qualquer modelo (regressão logística, random forest, HistGB) usa as mesmas
funções, então todos treinam e são avaliados nas mesmas linhas.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LEAKAGE_COLUMNS, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from src.data.transform import ID_COLUMN


@dataclass(frozen=True)
class SplitData:
    """Features e alvo já separados em treino e teste."""

    features_train: pd.DataFrame
    features_test: pd.DataFrame
    target_train: pd.Series
    target_test: pd.Series


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


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    exclude: tuple[str, ...] = LEAKAGE_COLUMNS,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> SplitData:
    """Divide em treino e teste mantendo a proporção do alvo, com seed fixo."""
    features = select_features(df, target, exclude)
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        df[target].astype(int),
        test_size=test_size,
        stratify=df[target],
        random_state=seed,
    )
    return SplitData(features_train, features_test, target_train, target_test)
