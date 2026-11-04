"""Carrega o modelo `champion` do Model Registry e pontua as visitas.

A URI e as credenciais do MLflow vêm do ambiente (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD`), como no treino. O
modelo é carregado uma vez, quando a API sobe.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient

from src.api.schemas import FEATURE_COLUMNS, USER_ID_FIELD, Prediction, Visitor
from src.config import MLFLOW_REGISTERED_MODEL_NAME
from src.training.registry import CHAMPION_ALIAS


@dataclass(frozen=True)
class LoadedModel:
    """Estimador do scikit-learn (com `predict_proba`) e a versão dele no Registry."""

    model: Any
    name: str
    version: str


def check_features(model: Any) -> None:  # noqa: ANN401
    """Garante que o modelo foi treinado com as colunas e a ordem de `Visitor`.

    Sem isso, um modelo treinado com outras features só falharia na primeira
    previsão, ou pior, daria scores errados em silêncio.
    """
    trained_on = list(getattr(model, "feature_names_in_", []))
    if trained_on != FEATURE_COLUMNS:
        raise ValueError(
            f"O modelo foi treinado com {trained_on}, mas a API recebe {FEATURE_COLUMNS}."
        )


def load_champion(
    name: str = MLFLOW_REGISTERED_MODEL_NAME, alias: str = CHAMPION_ALIAS
) -> LoadedModel:
    """Carrega a versão de `name` apontada por `alias`.

    Resolve o alias antes e carrega pelo número da versão, para que a versão
    informada pela API seja exatamente a do modelo em memória, mesmo que o alias
    mude no meio do caminho.
    """
    version = str(MlflowClient().get_model_version_by_alias(name, alias).version)
    model = load_model(f"models:/{name}/{version}")
    check_features(model)
    return LoadedModel(model=model, name=name, version=version)


def predict(loaded: LoadedModel, visitors: list[Visitor], threshold: float) -> list[Prediction]:
    """Score de cada visita, na ordem recebida, e se ele passa do `threshold`."""
    features = pd.DataFrame(
        [visitor.model_dump(exclude={USER_ID_FIELD}) for visitor in visitors],
        columns=FEATURE_COLUMNS,
    )
    scores = loaded.model.predict_proba(features)[:, 1]
    return [
        Prediction(user_id=visitor.user_id, score=float(score), purchase=bool(score >= threshold))
        for visitor, score in zip(visitors, scores, strict=True)
    ]
