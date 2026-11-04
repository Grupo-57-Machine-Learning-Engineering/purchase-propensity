"""Testes da carga do `champion` e da checagem de colunas."""

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

from src.api.model import check_features, load_champion
from src.config import LEAKAGE_COLUMNS
from src.training.registry import promote_best_model
from src.training.tracking import log_run


def test_load_champion_loads_the_promoted_random_forest(
    local_mlflow: MlflowClient, fitted_model: RandomForestClassifier, features: pd.DataFrame
) -> None:
    # Mesmo caminho do treino: log_run com skops e promoção pelo registry.
    log_run(
        run_name="rf-teste",
        model=fitted_model,
        params={"model": "RandomForestClassifier", "excluded_columns": ",".join(LEAKAGE_COLUMNS)},
        metrics={"cv_pr_auc_mean": 0.5},
        tags={},
        input_example=features.head(2),
    )
    version = promote_best_model(name="modelo-teste")

    loaded = load_champion(name="modelo-teste")

    assert loaded.version == version
    assert loaded.name == "modelo-teste"
    assert loaded.model.predict_proba(features).shape == (len(features), 2)


def test_check_features_rejects_a_model_trained_with_other_columns(
    features: pd.DataFrame,
) -> None:
    reordered = features[list(reversed(features.columns))]
    model = RandomForestClassifier(n_estimators=2).fit(reordered, features["loc_uk"])

    with pytest.raises(ValueError, match="treinado com"):
        check_features(model)
