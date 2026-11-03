"""Baseline de regressão logística. Rode com `python -m src.training.baseline`.

O fluxo (split, treino, cross-validation, MLflow) é o de `src/training/experiment.py`;
aqui fica só a escolha do modelo (`src/models/logistic_regression.py`).
"""

from src.models.logistic_regression import (
    build_logistic_regression,
    describe_logistic_regression,
)
from src.training.experiment import Experiment, main

BASELINE = Experiment(
    run_name="baseline-logreg-balanced-no-leakage",
    build_model=build_logistic_regression,
    describe_model=describe_logistic_regression,
)


if __name__ == "__main__":
    main(BASELINE, "Treina a regressão logística (baseline) e loga no MLflow.")
