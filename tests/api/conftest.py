"""Fixtures da API: um modelo pequeno treinado com as mesmas colunas de `Visitor`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.api.schemas import EXAMPLE_VISITOR, FEATURE_COLUMNS


@pytest.fixture
def features() -> pd.DataFrame:
    """Flags aleatórias com as colunas de treino, na ordem do modelo."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((200, len(FEATURE_COLUMNS))) < 0.3, columns=FEATURE_COLUMNS)


@pytest.fixture
def fitted_model(features: pd.DataFrame) -> RandomForestClassifier:
    """Random Forest pequeno em que o carrinho decide a compra."""
    target = features["basket_add_detail"].astype(int)
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(features, target)


@pytest.fixture
def visitor() -> dict[str, str | bool]:
    """Uma visita válida no formato da API."""
    return dict(EXAMPLE_VISITOR)
