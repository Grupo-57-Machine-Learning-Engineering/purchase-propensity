"""Compara os runs do experimento. Rode com `python -m src.training.compare`.

PR-AUC médio entre os folds (`cv_pr_auc_mean`, `src/training/cross_validation.py`)
é mais estável que o holdout único (`test_pr_auc`) para decidir qual modelo é o
melhor, então é o critério de ordenação aqui — e também o que
`src/training/registry.py` usa para escolher qual run promover a campeão.
"""

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_EXPERIMENT_NAME

RANKING_METRIC = "cv_pr_auc_mean"
TABLE_METRICS: tuple[str, ...] = (
    "cv_pr_auc_mean",
    "cv_pr_auc_std",
    "test_pr_auc",
    "total_duration_seconds",
)


def rank_runs(client: MlflowClient | None = None) -> list[Run]:
    """Runs concluídas do experimento, da maior para a menor `cv_pr_auc_mean`.

    Só runs com `status = FINISHED`: uma run que cai entre logar a métrica e
    logar o modelo (`FAILED`) pode ficar no topo do ranking sem ter modelo
    nenhum para promover.
    """
    client = client or MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{RANKING_METRIC} DESC"],
    )


def format_table(runs: list[Run]) -> str:
    """Tabela comparativa: nome da run, modelo, métricas de CV/teste e tempo total."""
    header = ["run_name", "model", *TABLE_METRICS]
    rows = [header]
    for run in runs:
        model_name = run.data.params.get("model", "?")
        values = [f"{run.data.metrics.get(m, float('nan')):.4f}" for m in TABLE_METRICS]
        rows.append([run.info.run_name, model_name, *values])
    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    )


def main() -> None:
    """Imprime a tabela comparativa dos runs do experimento."""
    print(format_table(rank_runs()))


if __name__ == "__main__":
    main()
