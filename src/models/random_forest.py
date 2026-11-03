"""Random Forest com os hiperparâmetros da busca (`make tune-rf`)."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_FOREST_PARAMS, RANDOM_SEED


def build_random_forest(
    seed: int = RANDOM_SEED, params: dict[str, Any] | None = None
) -> RandomForestClassifier:
    """Random Forest com peso balanceado (4,2% de positivos) e os parâmetros da busca.

    Os hiperparâmetros vêm de `configs/params.yaml`; `params` os substitui. A
    busca passa `{}` para partir dos padrões do scikit-learn.
    """
    return RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        **(RANDOM_FOREST_PARAMS if params is None else params),
    )


def describe_random_forest(model: RandomForestClassifier) -> dict[str, str | int | float]:
    """Hiperparâmetros que identificam o modelo na run do MLflow.

    `max_depth` vazio (árvore sem limite) vira -1, porque o MLflow não loga `None`.
    """
    params = model.get_params()
    return {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"] or -1,
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": str(params["max_features"]),
        "class_weight": str(params["class_weight"]),
    }
