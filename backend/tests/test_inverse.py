"""
Tests for Inverse Defect Localization API Endpoint.
"""
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from src.stage3D_inference import SpatialDefectLocalizer

client = TestClient(app)

def test_inverse_prediction_with_64x64_grid():
    # Synthetic 64x64 stress grid with peak at quadrant 2 (bottom-left)
    grid = np.ones((64, 64), dtype=float) * 100.0
    grid[15:25, 15:25] = 280.0 # elevated stress notch
    
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "stress_grid_64x64": grid.tolist(),
        "applied_traction_psi": 100.0,
        "return_heatmap": True
    }
    response = client.post("/api/v1/predict/inverse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["provenance"] == "ML_SURROGATE"
    assert "prediction_id" in data
    assert data["prediction_id"].startswith("inv_")
    assert "predicted_defect_x_mm" in data
    assert "predicted_defect_y_mm" in data
    assert 0.0 <= data["predicted_defect_x_mm"] <= 1000.0
    assert 0.0 <= data["predicted_defect_y_mm"] <= 800.0
    assert "uncertainty_radius_mm" in data
    assert "uncertainty_covariance_mm2" in data
    assert "uncertainty_ellipse" in data
    assert "abstention" in data
    assert "heatmap_grid_64x64" in data
    assert len(data["heatmap_grid_64x64"]) == 64
    assert len(data["heatmap_grid_64x64"][0]) == 64

def test_inverse_prediction_with_scattered_coordinates():
    # Scattered measurements across domain
    np.random.seed(42)
    N = 500
    x_pts = np.random.uniform(50.0, 950.0, N)
    y_pts = np.random.uniform(50.0, 750.0, N)
    coords = np.column_stack([x_pts, y_pts]).tolist()
    
    # Stress values with a concentration near (750, 600)
    dists = np.hypot(x_pts - 750.0, y_pts - 600.0)
    values = (100.0 + 150.0 * np.exp(-dists / 80.0)).tolist()
    
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "stress_coordinates": coords,
        "stress_values_psi": values,
        "applied_traction_psi": 100.0,
        "confidence_abstention_threshold_mm": 25.0
    }
    response = client.post("/api/v1/predict/inverse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Inversion should place predicted defect in upper-right quadrant near (750, 600)
    assert data["predicted_defect_x_mm"] > 500.0
    assert data["predicted_defect_y_mm"] > 400.0
    assert data["abstention"]["status"] in ["ACCEPTED", "FLAGGED_FOR_REVIEW"]

def test_inverse_prediction_numerical_agreement():
    """Verify agreement between API and standalone SpatialDefectLocalizer for identical grid."""
    standalone = SpatialDefectLocalizer("unet")
    dummy_grid = np.ones((64, 64), dtype=np.float32)
    dummy_grid[20:30, 40:50] = 3.2 # normalized stress
    
    # Standalone prediction (features passed as normalized grid)
    ref_res = standalone.predict(dummy_grid, 1000.0, 800.0)
    
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "stress_grid_64x64": (dummy_grid * 100.0).tolist(), # 100 psi traction
        "applied_traction_psi": 100.0
    }
    response = client.post("/api/v1/predict/inverse", json=payload)
    assert response.status_code == 200
    api_res = response.json()
    
    assert abs(api_res["predicted_defect_x_mm"] - ref_res["predicted_x_mm"]) < 1e-1
    assert abs(api_res["predicted_defect_y_mm"] - ref_res["predicted_y_mm"]) < 1e-1

def test_inverse_rejects_missing_inputs():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0
        # No grid or coordinates provided
    }
    response = client.post("/api/v1/predict/inverse", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "MISSING_STRESS_FIELD_INPUT"
