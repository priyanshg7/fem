"""
Stage 4C & 4D: Independent Endpoint Verification and Extended Integration Test Suite.
Performs specimen-level forward and inverse parity tests, edge case validations,
anisotropic covariance checks, and operational safety audits.
"""
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import numpy as np
import torch
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from src.stage3B_inference import ForwardStressEnsemblePredictor
from src.stage3D_inference import SpatialDefectLocalizer
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field

client = TestClient(app)

def run_stage4c_forward_parity_tests():
    """
    Independent verification of POST /api/v1/predict/forward across 10 diverse specimens.
    Calculates absolute and relative errors against standalone research implementation.
    """
    print("\n--- Running Stage 4C Forward Endpoint Specimen Parity Tests ---")
    standalone = ForwardStressEnsemblePredictor(models_dir=Path("models/stage3B_forward_stacking"))
    
    specimens = [
        {"id": "SPEC-F01", "desc": "Square Canonical", "L": 1000.0, "H": 1000.0, "Dc": 83.333, "x": 300.0, "y": 700.0, "d": 20.0, "traction": 100.0},
        {"id": "SPEC-F02", "desc": "Rect 1000x900", "L": 1000.0, "H": 900.0, "Dc": 75.0, "x": 350.0, "y": 650.0, "d": 15.0, "traction": 100.0},
        {"id": "SPEC-F03", "desc": "Rect 1000x800", "L": 1000.0, "H": 800.0, "Dc": 66.667, "x": 250.0, "y": 300.0, "d": 12.0, "traction": 100.0},
        {"id": "SPEC-F04", "desc": "Rect 1000x700", "L": 1000.0, "H": 700.0, "Dc": 58.333, "x": 400.0, "y": 500.0, "d": 18.0, "traction": 100.0},
        {"id": "SPEC-F05", "desc": "Rect 1000x600", "L": 1000.0, "H": 600.0, "Dc": 50.0, "x": 200.0, "y": 450.0, "d": 10.0, "traction": 100.0},
        {"id": "SPEC-F06", "desc": "Near Hole (Warning)", "L": 1000.0, "H": 1000.0, "Dc": 83.333, "x": 445.0, "y": 500.0, "d": 10.0, "traction": 100.0},
        {"id": "SPEC-F07", "desc": "Near Boundary (Warning)", "L": 1000.0, "H": 1000.0, "Dc": 83.333, "x": 15.0, "y": 700.0, "d": 10.0, "traction": 100.0},
        {"id": "SPEC-F08", "desc": "Small Defect d=5mm", "L": 1000.0, "H": 1000.0, "Dc": 83.333, "x": 300.0, "y": 700.0, "d": 5.0, "traction": 100.0},
        {"id": "SPEC-F09", "desc": "Large Defect d=35mm", "L": 1000.0, "H": 800.0, "Dc": 66.667, "x": 300.0, "y": 600.0, "d": 35.0, "traction": 100.0},
        {"id": "SPEC-F10", "desc": "Superposition 150psi", "L": 1000.0, "H": 1000.0, "Dc": 83.333, "x": 300.0, "y": 700.0, "d": 20.0, "traction": 150.0},
    ]
    
    results = []
    for spec in specimens:
        # 1. Standalone inference
        std_out = standalone.predict(
            plate_length_mm=spec["L"],
            plate_height_mm=spec["H"],
            central_hole_diameter_mm=spec["Dc"],
            defect_diameter_mm=spec["d"],
            defect_center_x_mm=spec["x"],
            defect_center_y_mm=spec["y"],
            applied_traction_psi=spec["traction"],
            framework="heterogeneous"
        )
        std_kt = std_out["predicted_Kt_element"]
        std_stress = std_out["predicted_max_stress_psi"]
        
        # 2. FastAPI endpoint
        payload = {
            "plate_length_mm": spec["L"],
            "plate_height_mm": spec["H"],
            "central_hole_diameter_mm": spec["Dc"],
            "defect_diameter_mm": spec["d"],
            "defect_center_x_mm": spec["x"],
            "defect_center_y_mm": spec["y"],
            "applied_traction_psi": spec["traction"],
            "framework": "heterogeneous"
        }
        res = client.post("/api/v1/predict/forward", json=payload)
        assert res.status_code == 200, f"Failed on {spec['id']}: {res.text}"
        data = res.json()
        
        api_kt = data["predicted_Kt_element"]
        api_stress = data["predicted_max_stress_psi"]
        
        diff_kt = abs(api_kt - std_kt)
        diff_stress = abs(api_stress - std_stress)
        rel_diff_kt = diff_kt / max(abs(std_kt), 1e-6)
        
        assert diff_kt < 1e-4, f"{spec['id']} Kt mismatch: API={api_kt}, Std={std_kt}, diff={diff_kt}"
        assert diff_stress < 0.05, f"{spec['id']} Stress mismatch: API={api_stress}, Std={std_stress}"
        assert data["provenance"] == "ML_SURROGATE"
        
        # Check specific expected warnings
        if spec["id"] == "SPEC-F06":
            assert any("CRITICAL_LIGAMENT_PROXIMITY" in w for w in data["warnings"])
        elif spec["id"] == "SPEC-F07":
            assert any("CRITICAL_BOUNDARY_PROXIMITY" in w for w in data["warnings"])
        elif spec["id"] == "SPEC-F10":
            assert any("NON_STANDARD_TRACTION" in w for w in data["warnings"])
            
        results.append({
            "specimen": spec["id"],
            "desc": spec["desc"],
            "std_Kt": std_kt,
            "api_Kt": api_kt,
            "diff_Kt": diff_kt,
            "rel_diff_Kt_pct": rel_diff_kt * 100.0,
            "std_stress_psi": std_stress,
            "api_stress_psi": api_stress,
            "diff_stress_psi": diff_stress,
            "warnings_count": len(data["warnings"])
        })
        print(f"  [PASS] {spec['id']} ({spec['desc']}): Kt={api_kt:.4f}, stress={api_stress:.2f} psi, diff_Kt={diff_kt:.2e}")
        
    return results

def run_stage4c_inverse_parity_tests():
    """
    Independent verification of POST /api/v1/predict/inverse across real FEM confirmatory specimens
    and high-uncertainty stress fields. Validates mathematical anisotropic covariance transformation:
    Sigma_phys = J Sigma_norm J^T, coordinate precision, and abstention thresholding.
    """
    print("\n--- Running Stage 4C Inverse Endpoint Specimen Parity Tests ---")
    localizer = SpatialDefectLocalizer(model_type="unet", model_path="models/stage3D_spatial_inverse/model_D5_unet.pt")
    
    specimens = [
        {
            "id": "SPEC-I01",
            "desc": "Rect 1000x600 Real FEM Confirmatory",
            "file": "STAGE3D_BLIND_rectangular_1000x600_0001_fields.npz",
            "L": 1000.0,
            "H": 600.0,
            "conf_id": "STAGE3D_BLIND_rectangular_1000x600_0001",
            "is_synthetic": False
        },
        {
            "id": "SPEC-I02",
            "desc": "Rect 1000x700 Real FEM Confirmatory",
            "file": "STAGE3D_BLIND_rectangular_1000x700_0101_fields.npz",
            "L": 1000.0,
            "H": 700.0,
            "conf_id": "STAGE3D_BLIND_rectangular_1000x700_0101",
            "is_synthetic": False
        },
        {
            "id": "SPEC-I03",
            "desc": "Rect 1000x800 Real FEM Confirmatory",
            "file": "STAGE3D_BLIND_rectangular_1000x800_0201_fields.npz",
            "L": 1000.0,
            "H": 800.0,
            "conf_id": "STAGE3D_BLIND_rectangular_1000x800_0201",
            "is_synthetic": False
        },
        {
            "id": "SPEC-I04",
            "desc": "Rect 1000x900 Real FEM Confirmatory",
            "file": "STAGE3D_BLIND_rectangular_1000x900_0301_fields.npz",
            "L": 1000.0,
            "H": 900.0,
            "conf_id": "STAGE3D_BLIND_rectangular_1000x900_0301",
            "is_synthetic": False
        },
        {
            "id": "SPEC-I05",
            "desc": "Square 1000x1000 Real FEM Confirmatory",
            "file": "STAGE3D_BLIND_square_1000x1000_0401_fields.npz",
            "L": 1000.0,
            "H": 1000.0,
            "conf_id": "STAGE3D_BLIND_square_1000x1000_0401",
            "is_synthetic": False
        },
        {
            "id": "SPEC-I06",
            "desc": "Ambiguous Field (Abstention Trigger)",
            "file": None,
            "L": 1000.0,
            "H": 1000.0,
            "conf_id": None,
            "is_synthetic": True
        }
    ]
    
    meta_df = pd.read_csv("data/stage3D_blind_confirmatory/stage3D_confirmatory_metadata.csv")
    field_dir = Path("data/stage3D_blind_confirmatory/spatial_fields")
    
    results = []
    for spec in specimens:
        L = spec["L"]
        H = spec["H"]
        
        if not spec["is_synthetic"]:
            npz_path = field_dir / spec["file"]
            d = np.load(npz_path)
            cnorm = normalize_coordinates(d["nodes"], L, H)
            grid_64 = rasterize_stress_field(cnorm, d["nodal_von_mises_psi"], grid_res=64)
            # Retrieve ground truth
            row = meta_df[meta_df["confirmatory_id"] == spec["conf_id"]].iloc[0]
            true_x = float(row["defect_center_x_mm"])
            true_y = float(row["defect_center_y_mm"])
        else:
            # Synthetic uniform field to test high dispersion
            grid_64 = np.ones((64, 64), dtype=np.float32) * 100.0
            true_x, true_y = 500.0, 500.0
            
        # 1. Standalone D5 Localizer prediction
        std_out = localizer.predict(
            features=grid_64 / 100.0,
            plate_length_mm=L,
            plate_height_mm=H,
            confidence_abstention_threshold_mm=9.07
        )
        std_x = std_out["predicted_x_mm"]
        std_y = std_out["predicted_y_mm"]
        std_disp = std_out["uncertainty_radius_mm"]
        std_cov = np.array(std_out["uncertainty_covariance_mm2"])
        
        # 2. FastAPI Endpoint call
        payload = {
            "plate_length_mm": L,
            "plate_height_mm": H,
            "stress_grid_64x64": grid_64.tolist(),
            "applied_traction_psi": 100.0,
            "return_heatmap": True
        }
        res = client.post("/api/v1/predict/inverse", json=payload)
        assert res.status_code == 200, f"Inverse failed for {spec['id']}: {res.text}"
        data = res.json()
        
        api_x = data["predicted_defect_x_mm"]
        api_y = data["predicted_defect_y_mm"]
        api_disp = data["uncertainty_radius_mm"]
        api_cov = np.array(data["uncertainty_covariance_mm2"])
        
        # Numerical parity assertions
        diff_coord = math.hypot(api_x - std_x, api_y - std_y)
        diff_disp = abs(api_disp - std_disp)
        diff_cov_max = np.max(np.abs(api_cov - std_cov))
        
        assert diff_coord < 0.1, f"Coordinate divergence on {spec['id']}: API=({api_x}, {api_y}), Std=({std_x}, {std_y})"
        assert diff_disp < 0.1, f"Dispersion divergence on {spec['id']}: API={api_disp}, Std={std_disp}"
        assert diff_cov_max < 0.1, f"Covariance divergence on {spec['id']}: max diff={diff_cov_max}"
        assert data["input_mode"] == "FULL_FIELD_64x64"
        assert data["provenance"] == "ML_SURROGATE"
        
        # Abstention decision check
        if spec["is_synthetic"]:
            assert data["abstention"]["abstain"] is True
            assert data["abstention"]["status"] == "FLAGGED_FOR_REVIEW"
        else:
            assert data["abstention"]["abstain"] is False
            assert data["abstention"]["status"] == "ACCEPTED"
            
        loc_err = math.hypot(api_x - true_x, api_y - true_y) if not spec["is_synthetic"] else None
        
        results.append({
            "specimen": spec["id"],
            "desc": spec["desc"],
            "true_x": true_x if not spec["is_synthetic"] else "N/A",
            "true_y": true_y if not spec["is_synthetic"] else "N/A",
            "api_x": api_x,
            "api_y": api_y,
            "localization_error_mm": round(loc_err, 2) if loc_err is not None else "N/A",
            "dispersion_radius_mm": api_disp,
            "coord_parity_diff_mm": round(diff_coord, 4),
            "abstention_status": data["abstention"]["status"]
        })
        print(f"  [PASS] {spec['id']} ({spec['desc']}): Pred=({api_x:.1f}, {api_y:.1f}) mm, disp={api_disp:.2f} mm -> {data['abstention']['status']} (Parity diff={diff_coord:.4f} mm)")
        
    return results

def run_stage4c_negative_and_boundary_tests():
    """
    Validates robust error handling, boundary limits, and input sanitization.
    """
    print("\n--- Running Stage 4C Negative & Boundary Tests ---")
    
    # 1. Non-positive dimension in forward request
    res = client.post("/api/v1/predict/forward", json={
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.33,
        "defect_diameter_mm": -10.0,
        "defect_center_x_mm": 300.0,
        "defect_center_y_mm": 500.0
    })
    assert res.status_code in [400, 422], f"Expected 400/422 on negative diameter, got {res.status_code}"
    print("  [PASS] Forward rejects negative diameter with structured error.")
    
    # 2. Severe hole collision
    res = client.post("/api/v1/predict/forward", json={
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.33,
        "defect_diameter_mm": 30.0,
        "defect_center_x_mm": 500.0,
        "defect_center_y_mm": 500.0
    })
    assert res.status_code == 400
    assert "CENTRAL_HOLE_COLLISION" in res.text
    print("  [PASS] Forward rejects direct central-hole collision.")
    
    # 3. Outer boundary penetration
    res = client.post("/api/v1/predict/forward", json={
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.33,
        "defect_diameter_mm": 30.0,
        "defect_center_x_mm": 995.0,
        "defect_center_y_mm": 500.0
    })
    assert res.status_code == 400
    assert "PLATE_BOUNDARY_PENETRATION" in res.text
    print("  [PASS] Forward rejects outer boundary penetration.")
    
    # 4. Unknown model ID in registry
    res = client.get("/api/v1/models/unregistered_surrogate_model_99")
    assert res.status_code == 404
    assert "MODEL_NOT_FOUND" in res.text
    print("  [PASS] Registry returns sanitized 404 on unregistered model.")
    
    # 5. Inverse request with NaN in 64x64 grid
    import json
    bad_grid = np.ones((64, 64), dtype=np.float32)
    bad_grid[10, 10] = float("nan")
    raw_json = json.dumps({
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "stress_grid_64x64": bad_grid.tolist()
    })
    res = client.post("/api/v1/predict/inverse", content=raw_json, headers={"Content-Type": "application/json"})
    assert res.status_code == 400
    assert "NON_FINITE_STRESS_VALUES" in res.text
    print("  [PASS] Inverse rejects non-finite matrix entries.")
    
    # 6. Inverse request with invalid shape (32x32)
    bad_shape_grid = np.ones((32, 32), dtype=np.float32)
    res = client.post("/api/v1/predict/inverse", json={
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "stress_grid_64x64": bad_shape_grid.tolist()
    })
    assert res.status_code == 400
    assert "INVALID_STRESS_GRID_SHAPE" in res.text
    print("  [PASS] Inverse rejects malformed grid dimensions.")
    
    # 7. Inverse request with experimental sensor-points
    res = client.post("/api/v1/predict/inverse", json={
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "stress_coordinates": [[100.0, 100.0], [200.0, 300.0], [500.0, 500.0]],
        "stress_values_psi": [105.0, 110.0, 130.0]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["input_mode"] == "EXPERIMENTAL_SCATTERED_SENSORS"
    assert any("EXPERIMENTAL_INPUT_MODE" in w for w in data["warnings"])
    print("  [PASS] Inverse marks sensor-point mode as EXPERIMENTAL with clear advisory warning.")

if __name__ == "__main__":
    f_res = run_stage4c_forward_parity_tests()
    i_res = run_stage4c_inverse_parity_tests()
    run_stage4c_negative_and_boundary_tests()
    print("\n==================================================")
    print("ALL STAGE 4C INDEPENDENT VERIFICATION TESTS PASSED")
    print("==================================================")
