"""
Tests for Physics and Geometry Validation Endpoint.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_valid_canonical_geometry():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 66.67, # 800 / 12
        "defect_diameter_mm": 15.0,
        "defect_center_x_mm": 250.0,
        "defect_center_y_mm": 300.0
    }
    response = client.post("/api/v1/validate/geometry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["can_proceed_to_inference"] is True
    assert data["canonical_geometry_compliance"] is True
    assert data["clearances"]["aspect_ratio"] == 1.25

def test_collision_with_central_hole_rejected():
    # Defect placed right in the center (overlapping central hole)
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 66.67,
        "defect_diameter_mm": 15.0,
        "defect_center_x_mm": 500.0, # At central hole center
        "defect_center_y_mm": 400.0
    }
    response = client.post("/api/v1/validate/geometry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["can_proceed_to_inference"] is False
    assert any(i["code"] == "CENTRAL_HOLE_COLLISION" for i in data["issues"])

def test_outer_boundary_penetration_rejected():
    # Defect placed outside right edge (x = 1010 mm)
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 66.67,
        "defect_diameter_mm": 20.0,
        "defect_center_x_mm": 995.0, # 995 + 10 = 1005 > 1000 mm
        "defect_center_y_mm": 300.0
    }
    response = client.post("/api/v1/validate/geometry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["can_proceed_to_inference"] is False
    assert any(i["code"] == "PLATE_BOUNDARY_PENETRATION" for i in data["issues"])

def test_non_canonical_central_hole_produces_warning():
    payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 120.0, # Departed from canonical 66.67 mm
        "defect_diameter_mm": 15.0,
        "defect_center_x_mm": 250.0,
        "defect_center_y_mm": 300.0
    }
    response = client.post("/api/v1/validate/geometry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["can_proceed_to_inference"] is True
    assert data["status"] == "VALID_WITH_WARNINGS"
    assert any(i["code"] == "NON_CANONICAL_CENTRAL_HOLE" for i in data["issues"])
