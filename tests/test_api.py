import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the `with` block runs the startup (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1


def test_predict(client, example_customer):
    r = client.post("/predict", json=example_customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert len(body["top_reasons"]) == 3


def test_predict_batch(client, example_customer):
    loyal = {**example_customer, "tenure": 70, "Contract": "Two year", "TotalCharges": 5985.0}
    r = client.post("/predict_batch", json={"customers": [example_customer, loyal]})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_predict_rejects_invalid_input(client, example_customer):
    bad = {**example_customer, "Contract": "Ten year", "tenure": -5}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in example_customer.items() if k != "tenure"}
    assert client.post("/predict", json=missing).status_code == 422
