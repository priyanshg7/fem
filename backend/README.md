# SimuStruct High-Performance FEM ML Backend (Stage 4A–4D Frozen Release v1.0.0)

Welcome to the production backend service for the **Static Analysis of a Plate with Defect Using Machine Learning** digital twin platform. This service encapsulates frozen research surrogate models, physics-based geometric collision validation, forward stress concentration prediction, and inverse spatial defect localization with spatial uncertainty dispersion and automated abstention.

**Current Release Status:** **FROZEN (v1.0.0) — Certified for Stage 5 React Engineering Dashboard Integration**

---

## 1. System Overview & Architecture

The backend bridges computational solid mechanics and client-facing digital twins (e.g., React / Streamlit applications). It exposes verified, reproducible machine-learning surrogates tagged explicitly as `ML_SURROGATE` to ensure strict separation from true numerical finite element analysis (FEM).

```
                      +-----------------------------------+
                      |   Client Application (React/UI)   |
                      +-----------------+-----------------+
                                        |  REST / JSON
                                        v
+-----------------------------------------------------------------------------------+
|                              FastAPI Service (:8000)                              |
|                                                                                   |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | /api/v1/health     |  | /api/v1/models     |  | /api/v1/validate/geometry   |  |
|  | Readiness & Status |  | Metadata & Schemas |  | Physics Collision Checks    |  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|                                                                                   |
|  +--------------------------------------------+  +-----------------------------+  |
|  | /api/v1/predict/forward                    |  | /api/v1/predict/inverse     |  |
|  | Forward Multi-Level Ensemble (Stage 3B)    |  | Model D5 U-Net (Stage 3D)   |  |
|  +--------------------------------------------+  +-----------------------------+  |
+---------------------------------------+-------------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+--------------------------------------+  +--------------------------------------+
| Forward Ensemble Predictor           |  | Spatial Defect Localizer (U-Net)     |
| - 7 Heterogeneous Estimators         |  | - 64x64 Normalized Stress Field In   |
| - Meta-Ridge Stacking                |  | - Heatmap Spatial Density Out        |
| - Latency: ~63 ms (CPU)              |  | - Soft-Argmax Localization           |
| - Accuracy: Test R² = 0.9634         |  | - Anisotropic Covariance & Abstention|
+--------------------------------------+  +--------------------------------------+
```

---

## 2. Directory Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── routes/
│   │       │   ├── health.py        # /health and /readiness checks
│   │       │   ├── models.py        # Registry inspection and metadata
│   │       │   ├── validation.py    # Physics-based geometry verification
│   │       │   ├── forward.py       # Forward stress prediction endpoint
│   │       │   └── inverse.py       # Inverse defect localization endpoint
│   │       └── api.py               # V1 API router aggregation
│   ├── core/
│   │   ├── config.py                # Environment configuration and geometric limits
│   │   └── errors.py                # Centralized exception types & sanitized handlers
│   ├── schemas/
│   │   ├── health.py                # Health & readiness Pydantic models
│   │   ├── models.py                # Model registry schemas
│   │   ├── validation.py            # Geometry validation schemas
│   │   ├── forward.py               # Forward prediction request/response contracts
│   │   └── inverse.py               # Inverse localization request/response contracts
│   ├── services/
│   │   ├── registry.py              # Model lifecycle manager & checkpoint loader
│   │   ├── validation_service.py    # Collision detection & domain boundaries
│   │   ├── forward_service.py       # Feature construction & ensemble execution
│   │   └── inverse_service.py       # Tensor normalization, U-Net, soft-argmax, abstention
│   └── main.py                      # FastAPI app entrypoint, CORS, lifespan handler
├── tests/
│   ├── test_health.py               # Tests for health & readiness
│   ├── test_models.py               # Tests for model registry
│   ├── test_validation.py           # Geometry edge case & clearance tests
│   ├── test_forward.py              # Forward prediction & parity tests
│   ├── test_inverse.py              # Inverse localization & abstention tests
│   ├── test_performance.py          # Latency, memory, and payload footprint benchmarks
│   ├── test_stage4c_verification.py # Independent specimen parity & error tests
│   └── run_tests.py                 # Master test runner (23/23 tests passed)
├── requirements.txt                 # Pinned dependencies
├── run_server.py                    # Server launcher
└── README.md                        # Authoritative documentation
```

---

## 3. Installation and Setup

### 3.1 Prerequisites
- Python 3.10+ (tested with Python 3.10 / 3.11 / 3.12 / 3.14)
- CPU or NVIDIA CUDA GPU (defaults to CPU inference)

### 3.2 Environment Setup
```powershell
# From project root:
cd backend

# Create virtual environment (optional if using global environment)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3.3 Launching the Server
To run the server locally on port 8000:
```powershell
python run_server.py
```
Or directly with Uvicorn:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started:
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`
- OpenAPI JSON Schema: `http://localhost:8000/openapi.json`

---

## 4. API Endpoints and Contracts

### 4.1 Health and Readiness

#### `GET /api/v1/health`
Returns liveness status, API version, active model IDs, and model health details.
```bash
curl -X GET http://localhost:8000/api/v1/health
```

#### `GET /api/v1/readiness`
Returns operational readiness. Returns HTTP 200 when all primary models are ready; returns HTTP 503 if any required model failed to load.

---

### 4.2 Model Registry

#### `GET /api/v1/models`
Lists all available surrogate models with metadata, parameters, and version IDs.

#### `GET /api/v1/models/{model_id}`
Returns granular metadata, operating boundaries, training metrics, and input/output contracts for a specific model (`forward_ensemble_stage3b` or `inverse_unet_d5_stage3d`).

---

### 4.3 Geometry Validation

#### `POST /api/v1/validate/geometry`
Validates proposed plate geometry and defect placement against solid mechanics non-penetration criteria before running computationally expensive surrogates.

**Request Schema:**
```json
{
  "plate_length_mm": 1000.0,
  "plate_height_mm": 1000.0,
  "central_hole_diameter_mm": 83.333,
  "defect_diameter_mm": 20.0,
  "defect_center_x_mm": 300.0,
  "defect_center_y_mm": 700.0
}
```

**Physics Checks Performed:**
1. **Aspect Ratio:** $1.0 \le L/H \le 1.667$ ($H \in [600, 1000]\text{ mm}$ for $L=1000\text{ mm}$).
2. **Canonical Central Hole:** $D_c = H / 12$ centered at $(L/2, H/2)$.
3. **Ligament Collision:** Clearance $c_{\text{lig}} = \sqrt{(x - L/2)^2 + (y - H/2)^2} - \frac{D_c + d}{2} > 0$.
4. **Outer Boundary Penetration:** $c_{\text{bnd}} = \min(x - d/2, L - x - d/2, y - d/2, H - y - d/2) > 0$.
5. **Clearance Margin Warnings:** Flags configurations closer than $0.012 H$ ($12\text{ mm}$) to hole or boundary.
6. **Intermediate Geometry Tracking:** Flags plate heights between verified discrete anchors ($600, 700, 800, 900, 1000\text{ mm}$).

---

### 4.4 Forward Stress Concentration Prediction

#### `POST /api/v1/predict/forward`
Predicts maximum element-centroid Von Mises stress $\sigma_{\max}$ and stress concentration factor $K_t$ using the Stage 3B Forward Heterogeneous Meta-Stacking Ensemble.

**Request Schema:**
```json
{
  "plate_length_mm": 1000.0,
  "plate_height_mm": 1000.0,
  "central_hole_diameter_mm": 83.333,
  "defect_diameter_mm": 20.0,
  "defect_center_x_mm": 300.0,
  "defect_center_y_mm": 700.0,
  "applied_traction_psi": 100.0,
  "framework": "heterogeneous"
}
```

**Response Schema:**
```json
{
  "success": true,
  "prediction_id": "fwd_7b8a1c92ef01",
  "provenance": "ML_SURROGATE",
  "operation": "FORWARD_MAX_STRESS_PREDICTION",
  "model_id": "forward_ensemble_stage3b",
  "model_version": "3.2.0-frozen",
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
  "execution_time_ms": 63.44,
  "warnings": [],
  "scope_notice": "This forward surrogate predicts scalar maximum element-centroid Von Mises stress. It does not reconstruct complete 2D spatial stress fields or replace numerical FEM convergence analysis."
}
```

---

### 4.5 Inverse Defect Localization

#### `POST /api/v1/predict/inverse`
Localizes an unknown defect from a normalized 2D Von Mises stress field using Model D5 `DefectHeatmapUNet`. Computes anisotropic spatial covariance and applies automated abstention.

**Input Modes Supported:**
1. `stress_grid_64x64`: Pre-gridded $64 \times 64$ float matrix of Von Mises stress (**Audited & Certified Mode**).
2. `stress_coordinates` + `stress_values_psi`: Scattered sensor measurements (**Experimental Mode**; emits advisory warning).

**Request Schema:**
```json
{
  "plate_length_mm": 1000.0,
  "plate_height_mm": 600.0,
  "stress_grid_64x64": [
    [100.0, 102.0, "... (64 floats) ..."],
    ["... 64 rows ..."]
  ],
  "applied_traction_psi": 100.0,
  "return_heatmap": true
}
```

**Response Schema:**
```json
{
  "success": true,
  "prediction_id": "inv_a4e2190cd561",
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

## 5. Automated Verification & Testing

The master test suite covers 23 automated tests across unit, integration, parity, and performance benchmark suites.

### Running the Test Suite
```powershell
python backend/tests/run_tests.py
```

### Benchmark Results
| Metric | Benchmark Result | Target Limit | Pass Status |
| :--- | :--- | :--- | :--- |
| **Forward Latency (Warm Avg)** | **63.44 ms** | $< 100.0\text{ ms}$ | **PASS** |
| **Inverse Latency (Warm Avg)** | **18.69 ms** | $< 100.0\text{ ms}$ | **PASS** |
| **Inverse Response Size** | **29.10 KB** | $< 100.0\text{ KB}$ | **PASS** |
| **Process RSS Memory** | **626.61 MB** | $< 1024.0\text{ MB}$ | **PASS** |
| **Forward Numerical Parity ($K_t$)** | **$< 4.1 \times 10^{-5}$** | Exact Parity | **PASS** |
| **Inverse Coordinate Parity** | **$< 0.006\text{ mm}$** | Exact Parity | **PASS** |
| **Automated Test Coverage** | **23 / 23 Passed** | 100% | **PASS** |

---

## 6. Stage 5 React Dashboard Integration Guidelines

1. **Geometry Validation Debounce:** Call `POST /api/v1/validate/geometry` on form input changes with a 150 ms debounce.
2. **Surrogate Transparency:** Display the badge `[ML_SURROGATE]` alongside all predictions.
3. **Uncertainty Ellipse SVG:** Render an SVG `<ellipse>` centered at `(predicted_defect_x_mm, predicted_defect_y_mm)` with radiuses `semi_major_axis_mm` and `semi_minor_axis_mm` rotated by `rotation_angle_deg`. Label as **"Heatmap Spatial Dispersion"**.
4. **Abstention Advisory:** When `abstention.abstain === true`, display an amber banner: `FLAGGED FOR REVIEW: High spatial uncertainty. Confirm with high-fidelity FEA.`
