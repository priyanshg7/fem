# Stage 4D — API Documentation, Integration Testing and Backend Contract Freeze Report

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 4D — API Documentation, Integration Testing and Backend Contract Freeze  
**Author:** Engineering Applications & Deployment Group  
**Status:** Frozen & Certified for Stage 5 React Frontend Integration  
**Date:** November 2026  

---

## 1. Executive Summary

This report completes Stage 4D by formally freezing the **SimuStruct ML Backend API Contract** (Version `1.0.0`), documenting all public endpoints, verifying OpenAPI / Swagger specifications, auditing end-to-end integration workflows, and confirming backend readiness for the Stage 5 React Engineering Dashboard.

The backend exposes the frozen research surrogates:
* **Forward Surrogate (`forward_ensemble_stage3b`):** Multi-level heterogeneous stacking ensemble predicting scalar maximum Von Mises stress $\sigma_{\max}$ and stress concentration factor $K_t$.
* **Inverse Surrogate (`inverse_unet_d5_stage3d`):** Model D5 spatial U-Net localizer predicting defect coordinates $(\hat{x}, \hat{y})$, anisotropic spatial covariance $\boldsymbol{\Sigma}_{\text{phys}}$, uncertainty ellipse geometry, and automated abstention decisions.

Every surrogate output is explicitly tagged as `ML_SURROGATE`. All 23 integration and independent parity tests have passed with 100% success, verifying sub-100 ms CPU latencies, robust error sanitization, and strict solid mechanics collision gating.

---

## 2. Frozen API Contract Specification (v1.0.0)

The API contract is frozen under `/api/v1`. The Stage 5 React dashboard can consume these endpoints without implementing Python-side feature engineering, rasterization, or mathematical covariance conversion.

```
+----------------------------------------------------------------------------------------------------+
|                                    SimuStruct API Routes (v1.0.0)                                  |
+------------------------------+--------+------------------------------------------------------------+
| Endpoint Path                | Method | Description & Primary Function                             |
+------------------------------+--------+------------------------------------------------------------+
| /api/v1/health               | GET    | Process liveness, service status, and model health details |
| /api/v1/readiness            | GET    | Lightweight readiness probe (returns 503 if not loaded)    |
| /api/v1/models               | GET    | Model registry catalog listing all certified surrogates    |
| /api/v1/models/{model_id}    | GET    | Granular model metadata, benchmarks, and I/O specifications|
| /api/v1/validate/geometry    | POST   | Solid mechanics non-penetration and clearance validation   |
| /api/v1/predict/forward      | POST   | Forward max stress & Kt prediction via 7-estimator ensemble|
| /api/v1/predict/inverse      | POST   | Inverse defect localization, covariance ellipse & heatmap  |
+------------------------------+--------+------------------------------------------------------------+
```

### 2.1 Forward Prediction Contract (`POST /api/v1/predict/forward`)

#### Request Schema (`ForwardPredictionRequest`)
| Field Name | Type | Required? | Accepted Range | Units | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `plate_length_mm` | `float` | **Yes** | 1000.0 (exact) | mm | — | Longitudinal plate dimension ($L$) |
| `plate_height_mm` | `float` | **Yes** | $[600.0, 1000.0]$ | mm | — | Transverse plate dimension ($H$) |
| `central_hole_diameter_mm` | `float` | **Yes** | $H / 12.0 \pm 0.5$ | mm | — | Canonical central circular hole ($D_c$) |
| `defect_diameter_mm` | `float` | **Yes** | $[5.0, 50.0]$ | mm | — | Defect hole diameter ($d$) |
| `defect_center_x_mm` | `float` | **Yes** | $(0, L)$ | mm | — | Longitudinal defect coordinate ($x$) |
| `defect_center_y_mm` | `float` | **Yes** | $(0, H)$ | mm | — | Transverse defect coordinate ($y$) |
| `applied_traction_psi` | `float` | No | $> 0$ | psi | 100.0 | Remote tensile traction $\sigma_0$ |
| `model_id` | `string` | No | — | — | `null` | Optional explicit model override |
| `framework` | `string` | No | `heterogeneous`, `dnn`, `pca` | — | `heterogeneous` | Stacking meta-learner selection |

#### Response Schema (`ForwardPredictionResponse`)
```json
{
  "success": true,
  "prediction_id": "fwd_7b8a1c92ef01",
  "timestamp": "2026-11-12T14:45:28.847120Z",
  "provenance": "ML_SURROGATE",
  "operation": "FORWARD_MAX_STRESS_PREDICTION",
  "model_id": "forward_ensemble_stage3b",
  "model_version": "3.2.0-frozen",
  "feature_contract_version": "v3B-6features-scaled",
  "predicted_max_stress_psi": 301.85,
  "predicted_Kt_element": 3.0185,
  "ensemble_uncertainty_psi": 4.12,
  "applied_traction_psi": 100.0,
  "base_model_predictions_Kt": {
    "ANN_512": 2.9912,
    "DeepMLP_D1": 3.0125,
    "DeepMLP_D2": 3.0210,
    "ResMLP": 3.0178,
    "RandomForest": 3.0245,
    "ExtraTrees": 3.0198,
    "HistGradientBoosting": 3.0162
  },
  "input_summary": {
    "normalized_coordinates": {
      "defect_x_over_L": 0.3,
      "defect_y_over_H": 0.7
    },
    "aspect_ratio": 1.0,
    "defect_diameter_ratio": 0.24,
    "ligament_clearance_ratio": 2.6216,
    "boundary_clearance_ratio": 0.29,
    "applied_traction_psi": 100.0,
    "fixed_material": {
      "material_name": "Structural Steel (Linear Elastic)",
      "youngs_modulus_GPa": 200.0,
      "poissons_ratio": 0.3,
      "thickness_mm": 1.0
    }
  },
  "execution_time_ms": 63.44,
  "warnings": [],
  "scope_notice": "This forward surrogate predicts scalar maximum element-centroid Von Mises stress. It does not reconstruct complete 2D spatial stress fields or replace numerical FEM convergence analysis."
}
```

---

### 2.2 Inverse Localization Contract (`POST /api/v1/predict/inverse`)

#### Request Schema (`InversePredictionRequest`)
| Field Name | Type | Required? | Accepted Range | Units | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `plate_length_mm` | `float` | **Yes** | 1000.0 | mm | — | Longitudinal plate dimension ($L$) |
| `plate_height_mm` | `float` | **Yes** | $[600.0, 1000.0]$ | mm | — | Transverse plate dimension ($H$) |
| `stress_grid_64x64` | `List[List[float]]` | **Conditional** | $64 \times 64$ matrix | psi | `null` | Primary verified input mode: 2D continuous normalized field |
| `stress_coordinates` | `List[List[float]]` | **Conditional** | $N \times 2$ array | mm | `null` | **Experimental Mode**: Discrete sensor coordinates |
| `stress_values_psi` | `List[float]` | **Conditional** | Length $N$ array | psi | `null` | **Experimental Mode**: Stress values at sensor points |
| `applied_traction_psi`| `float` | No | $> 0$ | psi | 100.0 | Reference far-field traction for normalization |
| `confidence_abstention_threshold_mm` | `float` | No | $> 0$ | mm | 9.07 | Trace dispersion threshold for abstention recommendation |
| `return_heatmap` | `bool` | No | `true`, `false` | — | `true` | Include $64 \times 64$ float probability heatmap |

#### Response Schema (`InversePredictionResponse`)
```json
{
  "success": true,
  "prediction_id": "inv_a4e2190cd561",
  "timestamp": "2026-11-12T14:45:29.810452Z",
  "provenance": "ML_SURROGATE",
  "operation": "INVERSE_SPATIAL_DEFECT_LOCALIZATION",
  "model_id": "inverse_unet_d5_stage3d",
  "model_version": "3.5.0-audited",
  "predicted_defect_x_mm": 507.73,
  "predicted_defect_y_mm": 180.95,
  "normalized_xi": 0.507729,
  "normalized_eta": 0.301587,
  "uncertainty_radius_mm": 1.80,
  "uncertainty_covariance_mm2": [
    [1.92, 0.41],
    [0.41, 1.32]
  ],
  "uncertainty_ellipse": {
    "semi_major_axis_mm": 1.48,
    "semi_minor_axis_mm": 1.05,
    "rotation_angle_deg": 26.4
  },
  "abstention": {
    "abstain": false,
    "status": "ACCEPTED",
    "threshold_used_mm": 9.07,
    "uncertainty_radius_mm": 1.80,
    "advice": "Prediction uncertainty lies within verified high-precision operating envelope (<= 25 mm guarantee)."
  },
  "input_mode": "FULL_FIELD_64x64",
  "heatmap_grid_64x64": [
    [0.0001, 0.0002, "... (64 floats) ..."]
  ],
  "execution_time_ms": 18.69,
  "warnings": [],
  "statistical_disclaimer": "Notice: The spatial uncertainty radius and covariance ellipse represent empirical heatmap dispersion. Due to audited nominal under-coverage in raw deep softmax outputs, they must not be interpreted as statistically calibrated 95% confidence intervals."
}
```

---

### 2.3 Geometry Validation Contract (`POST /api/v1/validate/geometry`)

Evaluates proposed geometry against solid mechanics constraints without loading heavy inference models.

#### Response Status Values:
* `VALID`: All dimensions within domain, clearances exceed safety threshold ($0.012 H = 12\text{ mm}$).
* `VALID_WITH_WARNINGS`: Clearances valid ($> 0$), but within near-interaction boundary ($< 12\text{ mm}$), non-canonical central hole, unanchored intermediate geometry, or defect size near domain edge.
* `INVALID`: Physical collision detected ($c_{\text{lig}} \le 0$), outer boundary penetration ($c_{\text{bnd}} \le 0$), non-positive dimensions, or out-of-domain aspect ratio ($L/H < 1.0$ or $L/H > 1.667$).

---

## 3. OpenAPI, Swagger UI and ReDoc Audit

1. **Specification Access:**
   * Interactive Swagger UI: `http://localhost:8000/docs`
   * ReDoc Reference: `http://localhost:8000/redoc`
   * OpenAPI 3.1.0 JSON: `http://localhost:8000/openapi.json`
2. **Schema Audit:**
   * All Pydantic models contain explicit `example`, `description`, `gt`, and `units` metadata.
   * Every endpoint defines responses for `200`, `400` (`InvalidGeometryError` / `SimuStructAPIException`), `404` (`ModelNotFoundError`), `422` (`RequestValidationError`), and `503` (`ModelNotReadyError`).
   * No internal checkpoint paths, filesystem structures, or server stack traces are leaked in schema definitions or error messages.

---

## 4. End-to-End Integration Testing & Performance Verification

The integrated master test runner (`backend/tests/run_tests.py`) exercises all layers across 23 automated integration tests:

```
==================================================
SIMUSTRUCT ML SURROGATE BACKEND TEST SUITE
==================================================
--- Running Health & Readiness Tests ---
  [PASS] test_root_endpoint (11.5 ms)
  [PASS] test_health_endpoint (5.3 ms)
  [PASS] test_readiness_probe (4.7 ms)
--- Running Model Registry Tests ---
  [PASS] test_list_models (4.3 ms)
  [PASS] test_get_forward_model_detail (5.7 ms)
  [PASS] test_get_inverse_model_detail (4.0 ms)
  [PASS] test_nonexistent_model_returns_404 (2.5 ms)
--- Running Geometry Validation Tests ---
  [PASS] test_valid_canonical_geometry (3.7 ms)
  [PASS] test_collision_with_central_hole_rejected (3.7 ms)
  [PASS] test_outer_boundary_penetration_rejected (7.5 ms)
  [PASS] test_non_canonical_central_hole_produces_warning (3.6 ms)
--- Running Forward Prediction Tests ---
  [PASS] test_forward_prediction_success (1392.7 ms)
  [PASS] test_forward_prediction_numerical_agreement_with_research (611.2 ms)
  [PASS] test_forward_prediction_rejects_collision (5.4 ms)
  [PASS] test_forward_prediction_rejects_boundary_penetration (4.1 ms)
--- Running Inverse Prediction Tests ---
  [PASS] test_inverse_prediction_with_64x64_grid (34.2 ms)
  [PASS] test_inverse_prediction_with_scattered_coordinates (21.9 ms)
  [PASS] test_inverse_prediction_numerical_agreement (60.3 ms)
  [PASS] test_inverse_rejects_missing_inputs (3.4 ms)
--- Running Performance Benchmarks ---
  [PASS] test_performance_benchmarks (905.5 ms)
--- Running Stage 4C Independent Verification Suite ---
  [PASS] run_stage4c_forward_parity_tests (1433.0 ms)
  [PASS] run_stage4c_inverse_parity_tests (255.6 ms)
  [PASS] run_stage4c_negative_and_boundary_tests (37.4 ms)
==================================================
TEST SUMMARY: 23 PASSED, 0 FAILED (TOTAL 23)
==================================================
```

### Performance & Footprint Re-Verification

| Metric | Measured Baseline | Target SLA | Production Compliance |
| :--- | :--- | :--- | :--- |
| **Forward Latency (Warm Avg)** | **63.44 ms** | $< 100.0\text{ ms}$ | **EXCEEDED (Fast)** |
| **Forward Latency (P95)** | **71.74 ms** | $< 150.0\text{ ms}$ | **EXCEEDED (Fast)** |
| **Inverse Latency (Warm Avg)** | **18.69 ms** | $< 100.0\text{ ms}$ | **EXCEEDED (Fast)** |
| **Inverse Latency (P95)** | **19.81 ms** | $< 100.0\text{ ms}$ | **EXCEEDED (Fast)** |
| **Inverse Heatmap Response Payload** | **29.10 KB** | $< 100.0\text{ KB}$ | **PASS** |
| **Process RSS Memory** | **626.61 MB** | $< 1024.0\text{ MB}$ | **PASS** |
| **Model Preloading Stability** | Zero reloads across repeated calls | Zero drift | **VERIFIED** |

---

## 5. Security & Operational Safety Audit

1. **Payload Safety Limits:**
   * Grid dimensions strictly enforced to $64 \times 64$ via NumPy shape verification (`INVALID_STRESS_GRID_SHAPE`).
   * Scattered sensor points capped at `MAX_COORDINATE_POINTS = 25000` to prevent memory exhaustion DoS (`PAYLOAD_TOO_LARGE`).
   * Non-finite values (`NaN`, `Inf`) strictly caught and rejected with HTTP 400 (`NON_FINITE_STRESS_VALUES`).
2. **Path & Checkpoint Protection:**
   * Checkpoint files are loaded only from configured internal directories (`settings.STAGE3B_DIR`, `settings.STAGE3D_DIR`).
   * User requests cannot supply arbitrary file paths, model paths, or deserialization instructions.
3. **CORS & Environment Configuration:**
   * Development origins (`http://localhost:3000`, `http://localhost:5173`, `http://127.0.0.1:3000`) enabled for local React prototyping.
   * Production deployment requires setting `ENVIRONMENT=production` and restricting `BACKEND_CORS_ORIGINS` to the production domain.

---

## 6. Stage 5 React Integration Recommendations

The Stage 5 React Engineering Dashboard can consume the backend following this integration pattern:

1. **Debounced Geometry Verification:**
   * Wire plate dimensions and defect position inputs to `POST /api/v1/validate/geometry` with a 150 ms debounce.
   * Disable the "Predict Forward Stress" button if `is_valid === false` and highlight colliding edges in red.
2. **Surrogate Stress Display:**
   * Display returned `predicted_max_stress_psi` and `predicted_Kt_element` alongside an explicit badge: `[ML_SURROGATE]`.
   * Display the ensemble dispersion standard deviation (`ensemble_uncertainty_psi`) as model epistemic disagreement.
3. **Heatmap & Uncertainty Ellipse Canvas:**
   * Render the $64 \times 64$ probability heatmap using an HTML5 `<canvas>` or WebGL texture with a colormap (e.g. Viridis / Turbo).
   * Overlay an SVG `<ellipse>` centered at `(predicted_defect_x_mm, predicted_defect_y_mm)` with axes `semi_major_axis_mm`, `semi_minor_axis_mm` rotated by `rotation_angle_deg`.
   * Label the ellipse as **"Heatmap Spatial Dispersion"** (do NOT label as "95% Confidence Interval").
4. **Abstention Banner Handling:**
   * If `abstention.abstain === true`, display an amber caution banner:
     `WARNING: Prediction flagged for review (uncertainty 171.4 mm > threshold 9.07 mm). Extreme ligament/boundary proximity detected; numerical FEM verification recommended.`

---

## 7. Release Gate Sign-Off

* **Backend Status:** Formally Frozen (Version `1.0.0`).
* **Test Gate:** 23 of 23 automated tests passed (100%).
* **Numerical Parity:** Confirmed across 10 forward specimens and 6 inverse specimens.
* **Release Gate Decision:** **APPROVED FOR STAGE 5 REACT DASHBOARD INTEGRATION**.
