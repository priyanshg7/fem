"""
Tests for Health and Readiness Endpoints.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "api_v1" in data

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True
    assert "models" in data
    assert "forward_ensemble_stage3b" in data["models"]
    assert "inverse_unet_d5_stage3d" in data["models"]
    assert data["models"]["forward_ensemble_stage3b"]["loaded"] is True
    assert data["models"]["inverse_unet_d5_stage3d"]["loaded"] is True

def test_readiness_probe():
    response = client.get("/api/v1/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["all_models_loaded"] is True
