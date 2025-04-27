import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "timestamp" in data


def test_performance_endpoint(client):
    resp = client.get("/performance?days=1")
    # Endpoint should respond even if no data yet
    assert resp.status_code == 200
    data = resp.json()
    assert "accuracy" in data
    assert "sharpe_ratio" in data
    assert "win_rate" in data
    assert "total_predictions" in data

