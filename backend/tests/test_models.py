"""
Tests for Model Registry and Metadata Endpoints.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_list_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["total_models"] >= 2
    assert data["active_forward_model"] == "forward_ensemble_stage3b"
    assert data["active_inverse_model"] == "inverse_unet_d5_stage3d"
    
    model_ids = [m["model_id"] for m in data["models"]]
    assert "forward_ensemble_stage3b" in model_ids
    assert "inverse_unet_d5_stage3d" in model_ids

def test_get_forward_model_detail():
    response = client.get("/api/v1/models/forward_ensemble_stage3b")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "forward_ensemble_stage3b"
    assert data["category"] == "forward"
    assert data["version"] == "3.2.0-frozen"
    assert "operating_domain" in data
    assert data["operating_domain"]["canonical_central_hole_rule"] == "Dc = H / 12.0"
    assert "validation_benchmark" in data
    assert data["validation_benchmark"]["confirmatory_R2"] == 0.9634

def test_get_inverse_model_detail():
    response = client.get("/api/v1/models/inverse_unet_d5_stage3d")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "inverse_unet_d5_stage3d"
    assert data["category"] == "inverse"
    assert data["version"] == "3.5.0-audited"
    assert "operating_domain" in data
    assert "io_specification" in data

def test_nonexistent_model_returns_404():
    response = client.get("/api/v1/models/nonexistent_model_xyz")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "MODEL_NOT_FOUND"
