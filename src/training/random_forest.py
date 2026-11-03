"""Random Forest. Rode com `python -m src.training.random_forest`.

Mesmo fluxo do baseline (`src/training/experiment.py`), trocando só o modelo
(`src/models/random_forest.py`), para comparar os dois pela mesma tabela de
métricas (`src/training/compare.py`).
"""

from src.models.random_forest import build_random_forest, describe_random_forest
from src.training.experiment import Experiment, main

RANDOM_FOREST = Experiment(
    run_name="random-forest-tuned-no-leakage",
    build_model=build_random_forest,
    describe_model=describe_random_forest,
)


if __name__ == "__main__":
    main(RANDOM_FOREST, "Treina o Random Forest e loga no MLflow.")
