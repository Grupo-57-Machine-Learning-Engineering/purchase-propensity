"""Testes dos endpoints, com um modelo em memória no lugar do Registry."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api.app import create_app
from src.api.model import LoadedModel
from src.api.schemas import MAX_VISITORS
from src.config import DECISION_THRESHOLD


@pytest.fixture
def client(fitted_model: RandomForestClassifier) -> Iterator[TestClient]:
    loaded = LoadedModel(model=fitted_model, name="modelo-teste", version="3")
    with TestClient(create_app(loader=lambda: loaded)) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_name": "modelo-teste", "model_version": "3"}


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_root_redirects_to_the_docs(client: TestClient, method: str) -> None:
    response = client.request(method, "/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_predict_scores_each_visitor_in_order(
    client: TestClient, visitor: dict[str, str | bool]
) -> None:
    sem_carrinho = {**visitor, "user_id": "sem-carrinho", "basket_add_detail": False}
    com_carrinho = {**visitor, "user_id": "com-carrinho", "basket_add_detail": True}

    response = client.post("/predict", json={"visitors": [sem_carrinho, com_carrinho]})

    body = response.json()
    assert response.status_code == 200
    assert body["model_version"] == "3"
    assert body["threshold"] == DECISION_THRESHOLD
    first, second = body["predictions"]
    assert [first["user_id"], second["user_id"]] == ["sem-carrinho", "com-carrinho"]
    assert first["score"] < second["score"]
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["score"] <= 1.0
        assert prediction["purchase"] == (prediction["score"] >= DECISION_THRESHOLD)


def test_predict_accepts_visitors_without_user_id(
    client: TestClient, visitor: dict[str, str | bool]
) -> None:
    del visitor["user_id"]

    response = client.post("/predict", json={"visitors": [visitor]})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["user_id"] is None


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"basket_icon_click": None}, id="flag-ausente"),
        pytest.param({"loc_uk": "talvez"}, id="flag-invalida"),
        pytest.param({"sign_in": True}, id="coluna-de-vazamento"),
    ],
)
def test_predict_rejects_invalid_visitors(
    client: TestClient, visitor: dict[str, str | bool], change: dict[str, object]
) -> None:
    invalid = {key: value for key, value in {**visitor, **change}.items() if value is not None}

    response = client.post("/predict", json={"visitors": [invalid]})

    assert response.status_code == 422


@pytest.mark.parametrize("size", [0, MAX_VISITORS + 1])
def test_predict_rejects_empty_or_oversized_batches(
    client: TestClient, visitor: dict[str, str | bool], size: int
) -> None:
    response = client.post("/predict", json={"visitors": [visitor] * size})

    assert response.status_code == 422
