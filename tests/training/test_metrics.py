"""Testes das métricas com dados sintéticos pequenos."""

import pytest

from src.training.metrics import compute_metrics, prefix_metrics

TARGET = [0, 0, 0, 0, 1, 1]


def test_perfect_ranking_gives_perfect_auc() -> None:
    proba = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]

    metrics = compute_metrics(TARGET, proba)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_precision_and_recall_follow_threshold() -> None:
    proba = [0.1, 0.6, 0.3, 0.4, 0.8, 0.4]

    metrics = compute_metrics(TARGET, proba, threshold=0.5)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_higher_threshold_trades_recall_for_precision() -> None:
    proba = [0.1, 0.6, 0.3, 0.4, 0.8, 0.4]

    low = compute_metrics(TARGET, proba, threshold=0.3)
    high = compute_metrics(TARGET, proba, threshold=0.7)

    assert low["recall"] > high["recall"]
    assert high["precision"] == pytest.approx(1.0)


def test_precision_is_zero_without_predicted_positives() -> None:
    metrics = compute_metrics(TARGET, [0.1, 0.2, 0.3, 0.4, 0.4, 0.4], threshold=0.9)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_pr_auc_penalizes_wrong_ranking_more_than_chance() -> None:
    inverted = compute_metrics(TARGET, [0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    assert inverted["pr_auc"] < sum(TARGET) / len(TARGET)


def test_single_class_target_raises() -> None:
    with pytest.raises(ValueError, match="duas classes"):
        compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])


def test_prefix_metrics_renames_keys() -> None:
    assert prefix_metrics({"pr_auc": 0.5}, "test") == {"test_pr_auc": 0.5}
