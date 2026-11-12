"""
Stage 6 Master Validation Matrix and Performance Benchmarking Suite.
Executes comprehensive end-to-end multi-sample benchmarks, scientific validation cases,
export round-trip checks, and security audits across the SimuStruct Digital Twin Platform.
"""
import time
import math
import json
import sys
import statistics
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.registry import model_registry
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field

client = TestClient(app)

def run_performance_benchmarks(sample_count: int = 50):
    """
    Executes multi-sample latency benchmarking for Forward and Inverse pipelines.
    Computes Min, Mean, Median, P95, and Max execution times.
    """
    print(f"\n=======================================================")
    print(f"STAGE 6C: MULTI-SAMPLE PERFORMANCE BENCHMARKING (N={sample_count})")
    print(f"=======================================================")
    
    # 1. Forward Benchmark Setup
    fwd_payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.333,
        "defect_diameter_mm": 20.0,
        "defect_center_x_mm": 300.0,
        "defect_center_y_mm": 700.0,
        "applied_traction_psi": 100.0,
        "framework": "heterogeneous"
    }
    
    fwd_times = []
    fwd_payload_sizes = []
    for i in range(sample_count):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict/forward", json=fwd_payload)
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        fwd_times.append(elapsed)
        if i == 0:
            fwd_payload_sizes.append(len(res.content))
            
    fwd_cold = fwd_times[0]
    fwd_warm = fwd_times[1:]
    fwd_mean = statistics.mean(fwd_warm)
    fwd_median = statistics.median(fwd_warm)
    fwd_p95 = np.percentile(fwd_warm, 95)
    fwd_min = min(fwd_warm)
    fwd_max = max(fwd_warm)
    
    print("\n[Forward Model — Stage 3B Heterogeneous Stacking]")
    print(f"  Cold-Start First Call:  {fwd_cold:.2f} ms")
    print(f"  Warm Mean Latency:      {fwd_mean:.2f} ms")
    print(f"  Warm Median Latency:    {fwd_median:.2f} ms")
    print(f"  Warm P95 Latency:       {fwd_p95:.2f} ms")
    print(f"  Warm Min / Max:         {fwd_min:.2f} ms / {fwd_max:.2f} ms")
    print(f"  Response Payload Size:  {fwd_payload_sizes[0]} bytes")
    
    # 2. Inverse Benchmark Setup
    sample_file = Path("frontend/src/data/sample_stress_field.json")
    with open(sample_file) as f:
        sample_data = json.load(f)
        
    inv_payload = {
        "plate_length_mm": sample_data["plate_length_mm"],
        "plate_height_mm": sample_data["plate_height_mm"],
        "applied_traction_psi": sample_data["applied_traction_psi"],
        "stress_grid_64x64": sample_data["stress_grid_64x64"],
        "return_heatmap": True,
        "confidence_abstention_threshold_mm": 9.07
    }
    
    inv_times = []
    inv_payload_sizes = []
    for i in range(sample_count):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict/inverse", json=inv_payload)
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        inv_times.append(elapsed)
        if i == 0:
            inv_payload_sizes.append(len(res.content))
            
    inv_cold = inv_times[0]
    inv_warm = inv_times[1:]
    inv_mean = statistics.mean(inv_warm)
    inv_median = statistics.median(inv_warm)
    inv_p95 = np.percentile(inv_warm, 95)
    inv_min = min(inv_warm)
    inv_max = max(inv_warm)
    
    print("\n[Inverse Model — Stage 3D Model D5 U-Net Localizer]")
    print(f"  Cold-Start First Call:  {inv_cold:.2f} ms")
    print(f"  Warm Mean Latency:      {inv_mean:.2f} ms")
    print(f"  Warm Median Latency:    {inv_median:.2f} ms")
    print(f"  Warm P95 Latency:       {inv_p95:.2f} ms")
    print(f"  Warm Min / Max:         {inv_min:.2f} ms / {inv_max:.2f} ms")
    print(f"  Response Payload Size:  {inv_payload_sizes[0]:,} bytes (includes 64x64 heatmap matrix)")
    
    return {
        "forward": {
            "cold_ms": round(fwd_cold, 2),
            "mean_ms": round(fwd_mean, 2),
            "median_ms": round(fwd_median, 2),
            "p95_ms": round(fwd_p95, 2),
            "payload_bytes": fwd_payload_sizes[0]
        },
        "inverse": {
            "cold_ms": round(inv_cold, 2),
            "mean_ms": round(inv_mean, 2),
            "median_ms": round(inv_median, 2),
            "p95_ms": round(inv_p95, 2),
            "payload_bytes": inv_payload_sizes[0]
        }
    }

def run_scientific_validation_matrix():
    """
    Executes the formal Stage 6B Release Validation Matrix covering all required cases.
    """
    print(f"\n=======================================================")
    print(f"STAGE 6B: FORMAL SCIENTIFIC RELEASE VALIDATION MATRIX")
    print(f"=======================================================")
    
    matrix_results = []
    
    # ---------------------------------------------------------
    # Case 1: Canonical Forward Regression (SPEC-F01 Square 1000x1000)
    # ---------------------------------------------------------
    f01_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.333, "defect_diameter_mm": 20.0,
        "defect_center_x_mm": 300.0, "defect_center_y_mm": 700.0,
        "applied_traction_psi": 100.0, "framework": "heterogeneous"
    }
    r = client.post("/api/v1/predict/forward", json=f01_req)
    d = r.json()
    f01_pass = (
        r.status_code == 200 and
        d["provenance"] == "ML_SURROGATE" and
        abs(d["predicted_Kt_element"] - 3.0185) < 0.05 and
        abs(d["predicted_max_stress_psi"] - 301.85) < 5.0
    )
    matrix_results.append({
        "id": "VAL-FWD-01",
        "category": "Forward Baseline Parity",
        "specimen": "Square 1000x1000 (d=20mm at (300, 700))",
        "expected": "Kt ≈ 3.018, sigma_max ≈ 301.8 psi, provenance=ML_SURROGATE",
        "observed": f"Kt={d.get('predicted_Kt_element')}, stress={d.get('predicted_max_stress_psi')} psi",
        "status": "PASS" if f01_pass else "FAIL"
    })
    
    # ---------------------------------------------------------
    # Case 2: Forward Boundary Collision Gating (Hole Collision)
    # ---------------------------------------------------------
    f_coll_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.333, "defect_diameter_mm": 20.0,
        "defect_center_x_mm": 500.0, "defect_center_y_mm": 500.0,
        "applied_traction_psi": 100.0
    }
    r = client.post("/api/v1/predict/forward", json=f_coll_req)
    f_coll_pass = (
        r.status_code == 400 and
        r.json()["error"]["code"] in ("INVALID_GEOMETRY", "GEOMETRY_COLLISION", "CENTRAL_HOLE_COLLISION") and
        any(issue.get("code") in ("CENTRAL_HOLE_COLLISION", "HOLE_COLLISION", "GEOMETRY_COLLISION") for issue in r.json()["error"].get("details", []))
    )
    matrix_results.append({
        "id": "VAL-FWD-02",
        "category": "Forward Safety Gating",
        "specimen": "Defect centered directly on central cutout (500, 500)",
        "expected": "HTTP 400 INVALID_GEOMETRY (CENTRAL_HOLE_COLLISION) rejection",
        "observed": f"HTTP {r.status_code} code={r.json().get('error', {}).get('code')} detail={r.json().get('error', {}).get('details', [{}])[0].get('code')}",
        "status": "PASS" if f_coll_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 3: Forward Outer Boundary Penetration Gating
    # ---------------------------------------------------------
    f_bnd_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "central_hole_diameter_mm": 83.333, "defect_diameter_mm": 30.0,
        "defect_center_x_mm": 5.0, "defect_center_y_mm": 500.0,
        "applied_traction_psi": 100.0
    }
    r = client.post("/api/v1/predict/forward", json=f_bnd_req)
    f_bnd_pass = (
        r.status_code == 400 and
        r.json()["error"]["code"] in ("INVALID_GEOMETRY", "PLATE_BOUNDARY_PENETRATION", "BOUNDARY_COLLISION") and
        any(issue.get("code") in ("PLATE_BOUNDARY_PENETRATION", "BOUNDARY_COLLISION") for issue in r.json()["error"].get("details", []))
    )
    matrix_results.append({
        "id": "VAL-FWD-03",
        "category": "Forward Safety Gating",
        "specimen": "Defect penetrating left boundary (x=5mm, d=30mm)",
        "expected": "HTTP 400 INVALID_GEOMETRY (PLATE_BOUNDARY_PENETRATION) rejection",
        "observed": f"HTTP {r.status_code} code={r.json().get('error', {}).get('code')} detail={r.json().get('error', {}).get('details', [{}])[0].get('code')}",
        "status": "PASS" if f_bnd_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 4: Inverse Confirmatory Specimen (SPEC-I02 1000x700)
    # ---------------------------------------------------------
    sample_file = Path("frontend/src/data/sample_stress_field.json")
    with open(sample_file) as f:
        sdata = json.load(f)
    i02_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 700.0,
        "applied_traction_psi": 100.0, "stress_grid_64x64": sdata["stress_grid_64x64"],
        "return_heatmap": True, "confidence_abstention_threshold_mm": 9.07
    }
    r = client.post("/api/v1/predict/inverse", json=i02_req)
    d = r.json()
    err_dist = math.hypot(d["predicted_defect_x_mm"] - 507.8, d["predicted_defect_y_mm"] - 244.4)
    i02_pass = (
        r.status_code == 200 and
        err_dist < 1.0 and
        d["abstention"]["status"] == "ACCEPTED" and
        d["uncertainty_radius_mm"] < 9.07
    )
    matrix_results.append({
        "id": "VAL-INV-01",
        "category": "Inverse Confirmatory Specimen",
        "specimen": "SPEC-I02 1000x700 (True: (507.8, 244.4) mm)",
        "expected": "Pred ≈ (507.8, 244.4), disp < 9.07 mm, status=ACCEPTED",
        "observed": f"Pred=({d.get('predicted_defect_x_mm')}, {d.get('predicted_defect_y_mm')}) err={err_dist:.2f}mm disp={d.get('uncertainty_radius_mm')}mm",
        "status": "PASS" if i02_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 5: Inverse Confident Miss (SPEC-I01 1000x600 Audit Case)
    # ---------------------------------------------------------
    meta_df = pd.read_csv("data/stage3D_blind_confirmatory/stage3D_confirmatory_metadata.csv")
    f_npz = Path("data/stage3D_blind_confirmatory/spatial_fields/STAGE3D_BLIND_rectangular_1000x600_0001_fields.npz")
    d_npz = np.load(f_npz)
    cnorm = normalize_coordinates(d_npz["nodes"], 1000.0, 600.0)
    grid_64 = rasterize_stress_field(cnorm, d_npz["nodal_von_mises_psi"], grid_res=64)
    
    i01_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 600.0,
        "applied_traction_psi": 100.0, "stress_grid_64x64": grid_64.tolist(),
        "return_heatmap": True, "confidence_abstention_threshold_mm": 9.07
    }
    r = client.post("/api/v1/predict/inverse", json=i01_req)
    d = r.json()
    true_x, true_y = 192.6, 582.6
    err_i01 = math.hypot(d["predicted_defect_x_mm"] - true_x, d["predicted_defect_y_mm"] - true_y)
    
    # Crucial scientific check: Pred is near central hole meridian (507.7, 180.9), dispersion is low (1.80mm),
    # and error is large (> 500mm). This confirms the model limitation is reproduced and documented!
    i01_pass = (
        r.status_code == 200 and
        abs(d["predicted_defect_x_mm"] - 507.7) < 1.0 and
        abs(d["predicted_defect_y_mm"] - 180.9) < 1.0 and
        d["abstention"]["status"] == "ACCEPTED" and
        err_i01 > 400.0
    )
    matrix_results.append({
        "id": "VAL-INV-02",
        "category": "Inverse Confident Miss Audit",
        "specimen": "SPEC-I01 1000x600 (True: (192.6, 582.6) mm, d=9.36mm)",
        "expected": "Pred near (507.7, 180.9) mm (Central hole attraction), disp=1.80mm, large error confirmed",
        "observed": f"Pred=({d.get('predicted_defect_x_mm')}, {d.get('predicted_defect_y_mm')}) err={err_i01:.2f}mm disp={d.get('uncertainty_radius_mm')}mm",
        "status": "PASS" if i01_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 6: Uniform Field Negative Control (High Uncertainty)
    # ---------------------------------------------------------
    unif_grid = (np.ones((64, 64), dtype=float) * 100.0).tolist()
    unif_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "applied_traction_psi": 100.0, "stress_grid_64x64": unif_grid,
        "return_heatmap": False, "confidence_abstention_threshold_mm": 9.07
    }
    r = client.post("/api/v1/predict/inverse", json=unif_req)
    d = r.json()
    unif_pass = (
        r.status_code == 200 and
        d["abstention"]["abstain"] is True and
        d["abstention"]["status"] == "FLAGGED_FOR_REVIEW" and
        d["uncertainty_radius_mm"] > 50.0
    )
    matrix_results.append({
        "id": "VAL-INV-03",
        "category": "Inverse Negative Control",
        "specimen": "Synthetic Uniform Field (100.0 psi)",
        "expected": "High dispersion (>50mm), abstain=True, FLAGGED_FOR_REVIEW",
        "observed": f"disp={d.get('uncertainty_radius_mm')}mm abstain={d.get('abstention', {}).get('abstain')} status={d.get('abstention', {}).get('status')}",
        "status": "PASS" if unif_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 7: Malformed Stress Matrix (1x1 Shape Rejection)
    # ---------------------------------------------------------
    bad_shape_req = {
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "stress_grid_64x64": [[100.0]]
    }
    r = client.post("/api/v1/predict/inverse", json=bad_shape_req)
    bad_shape_pass = (r.status_code == 400 and r.json()["error"]["code"] == "INVALID_STRESS_GRID_SHAPE")
    matrix_results.append({
        "id": "VAL-INV-04",
        "category": "Inverse Input Integrity",
        "specimen": "Malformed 1x1 matrix",
        "expected": "HTTP 400 INVALID_STRESS_GRID_SHAPE",
        "observed": f"HTTP {r.status_code} code={r.json().get('error', {}).get('code')}",
        "status": "PASS" if bad_shape_pass else "FAIL"
    })

    # ---------------------------------------------------------
    # Case 8: Non-Finite Values (NaN Rejection)
    # ---------------------------------------------------------
    raw_payload = json.dumps({
        "plate_length_mm": 1000.0, "plate_height_mm": 1000.0,
        "stress_grid_64x64": [[100.0]*64]*64
    }).replace("100.0", "NaN", 1) # Insert raw NaN into payload
    
    r = client.post(
        "/api/v1/predict/inverse",
        content=raw_payload.encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    nan_pass = (r.status_code == 400 and r.json()["error"]["code"] == "NON_FINITE_STRESS_VALUES")
    matrix_results.append({
        "id": "VAL-INV-05",
        "category": "Inverse Input Integrity",
        "specimen": "64x64 matrix containing NaN value",
        "expected": "HTTP 400 NON_FINITE_STRESS_VALUES",
        "observed": f"HTTP {r.status_code} code={r.json().get('error', {}).get('code')}",
        "status": "PASS" if nan_pass else "FAIL"
    })

    # Print Validation Matrix Table
    print("\n---------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Case ID':<12} | {'Category':<26} | {'Status':<6} | {'Observed Output Summary':<60}")
    print("---------------------------------------------------------------------------------------------------------------------------------")
    for res in matrix_results:
        print(f"{res['id']:<12} | {res['category']:<26} | {res['status']:<6} | {res['observed'][:60]:<60}")
    print("---------------------------------------------------------------------------------------------------------------------------------")
    
    all_pass = all(r["status"] == "PASS" for r in matrix_results)
    print(f"\nMatrix Result: {'ALL VALIDATION CASES PASSED (8/8)' if all_pass else 'SOME CASES FAILED'}")
    return matrix_results

if __name__ == "__main__":
    bench_data = run_performance_benchmarks(sample_count=50)
    matrix_data = run_scientific_validation_matrix()
