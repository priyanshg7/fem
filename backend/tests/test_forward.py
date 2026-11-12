"""
Tests for Forward Prediction API Endpoint.
"""
from fastapi.testclient import TestClient
from backend.app.main import app
from src.stage3B_inference import ForwardStressEnsemblePredictor

client = TestClient(app)

def test_forward_prediction_success():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 66.67,
        "defect_diameter_mm": 15.0,
        "defect_center_x_mm": 250.0,
        "defect_center_y_mm": 300.0,
        "applied_traction_psi": 100.0
    }
    response = client.post("/api/v1/predict/forward", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["provenance"] == "ML_SURROGATE"
    assert "prediction_id" in data
    assert data["prediction_id"].startswith("fwd_")
    assert "predicted_max_stress_psi" in data
    assert "predicted_Kt_element" in data
    assert data["predicted_Kt_element"] > 2.0
    assert data["predicted_Kt_element"] < 6.0
    assert "ensemble_uncertainty_psi" in data
    assert "base_model_predictions_Kt" in data
    assert len(data["base_model_predictions_Kt"]) == 7

def test_forward_prediction_numerical_agreement_with_research():
    """Verify that backend output matches research ForwardStressEnsemblePredictor to within 1e-3."""
    standalone = ForwardStressEnsemblePredictor()
    L, H, Dc, d, x, y = 1000.0, 700.0, 58.3333, 12.0, 300.0, 250.0
    
    # Standalone prediction
    ref_res = standalone.predict(L, H, Dc, d, x, y, applied_traction_psi=100.0)
    
    # API prediction
    payload = {
        "plate_length_mm": L,
        "plate_height_mm": H,
        "central_hole_diameter_mm": Dc,
        "defect_diameter_mm": d,
        "defect_center_x_mm": x,
        "defect_center_y_mm": y,
        "applied_traction_psi": 100.0
    }
    response = client.post("/api/v1/predict/forward", json=payload)
    assert response.status_code == 200
    api_res = response.json()
    
    # Check numerical equivalence
    assert abs(api_res["predicted_max_stress_psi"] - ref_res["predicted_max_stress_psi"]) < 1e-2
    assert abs(api_res["predicted_Kt_element"] - ref_res["predicted_Kt_element"]) < 1e-3

def test_forward_prediction_rejects_collision():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.33,
        "defect_diameter_mm": 20.0,
        "defect_center_x_mm": 500.0, # Collision with center hole
        "defect_center_y_mm": 500.0
    }
    response = client.post("/api/v1/predict/forward", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_GEOMETRY"

def test_forward_prediction_rejects_boundary_penetration():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 600.0,
        "central_hole_diameter_mm": 50.0,
        "defect_diameter_mm": 25.0,
        "defect_center_x_mm": 10.0, # 10 - 12.5 < 0 mm penetration
        "defect_center_y_mm": 300.0
    }
    response = client.post("/api/v1/predict/forward", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_GEOMETRY"
