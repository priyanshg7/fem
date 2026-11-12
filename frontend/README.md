# SimuStruct Digital Twin Platform — Frontend Application

Modern React 18 + TypeScript 5 engineering dashboard for real-time stress analysis, spatial defect localization, and traceable engineering case reporting. Built on Vite 5 with a dark technical engineering console design system.

---

## 1. Overview & Architecture

The frontend serves as the interactive engineering portal for the **SimuStruct Digital Twin Platform**, interfacing with the versioned FastAPI ML surrogate service (`/api/v1`).

### Primary Workspaces
1. **Forward Stress Analysis (Stage 5B):**
   - Parametric plate geometry selector (5 canonical aspect ratios $\alpha \in [1.0, 1.67]$, $L = 1000\text{ mm}$, $H \in [600, 1000]\text{ mm}$, $D_c = H/12$).
   - Interactive SVG canvas with real-time pointer click-and-drag defect repositioning, boundary clamping, and dynamic ligament lines.
   - 180 ms debounced geometry validation (`POST /api/v1/validate/geometry`) with physical non-penetration gating.
   - High-speed Stage 3B Stacking Ensemble prediction (`POST /api/v1/predict/forward`) displaying $\sigma_{\max}\text{ psi}$, $K_t$, ensemble uncertainty ($\pm\text{std}$), sub-100ms execution latency, and expandable base-model breakdown.
   - Stale input notice tracking parameter adjustments after prediction.
2. **Inverse Defect Inspection (Stage 5C):**
   - Blind spatial inverse localizer consuming continuous $64 \times 64$ Von Mises stress fields (`POST /api/v1/predict/inverse`).
   - File upload (JSON, CSV), manual clipboard paste, and built-in confirmatory demonstration dataset (`SPEC-I02`, $1000\times 700\text{ mm}$).
   - Matrix diagnostics: Validates $64 \times 64$ dimensions, checks for NaN/Inf non-finite values, and reports stress statistics ($\sigma_{\min}, \sigma_{\max}, \sigma_{\text{mean}}$).
   - 2D Spatial Heatmap Visualizer: Pixelated Viridis probability rendering, color scale legend, physical plate axes, central hole overlay ($D_c = H/12$), predicted defect reticle $(\hat{x}, \hat{y})$, and empirical spatial dispersion ellipse ($a, b, \theta^\circ$).
   - Verified vertical orientation: Row 0 maps to physical $y = 0\text{ mm}$ (bottom of plate), Row 63 maps to $y = H\text{ mm}$ (top of plate).
   - Backend abstention check: Retention threshold $\rho \le 9.07\text{ mm}$ (`ACCEPTED` vs `FLAGGED_FOR_REVIEW`) with clear engineering disclaimers distinguishing empirical softmax sharpness from calibrated confidence intervals.
   - Experimental sparse sensor mode kept behind a gated tab with distinct warnings.
3. **Engineering Case History & Reporting (Stage 5D):**
   - Immutable client-side case history (`localStorage`) logging unique case IDs, UTC timestamps, request payloads, response payloads, and cryptographic SHA-256 content hashes.
   - Search, filter (All, Forward, Inverse), and side-by-side case comparison modal.
   - Formal engineering report viewer (`ReportViewer`) matching computational audit standards.
   - Multi-format exports: Print-to-PDF (`window.print()` with clean `@media print` styles), machine-readable JSON (`{case_id}.json`), tabular summary CSV, and $64 \times 64$ matrix CSVs.

---

## 2. Directory Structure

```
frontend/
├── dist/                      # Production build output
├── node_modules/              # Pinned dependencies
├── public/                    # Static assets (favicons, SVG glyphs)
├── src/
│   ├── api/
│   │   ├── client.ts          # Centralized fetch API client with timeout & error typing
│   │   └── types.ts           # Strictly-typed Stage 4D API schema mirror & CaseRecord types
│   ├── components/
│   │   ├── common/            # Reusable UI primitives (Alert, Badge, Button, Card)
│   │   ├── forward/           # Forward workspace components:
│   │   │   ├── ClearancePanel.tsx     # Ligament & boundary clearance indicator
│   │   │   ├── GeometrySelector.tsx   # 5 canonical plate geometry presets
│   │   │   ├── ParameterControls.tsx  # Sliders, numeric inputs, verification presets
│   │   │   ├── PlateViewer.tsx        # Interactive SVG canvas with click & drag
│   │   │   ├── ResultsPanel.tsx       # Stress metrics, uncertainty, estimator breakdown
│   │   │   └── StaleNotice.tsx        # Alert for modified inputs post-inference
│   │   ├── inverse/           # Inverse workspace components:
│   │   │   ├── HeatmapViewer.tsx      # 2D Viridis probability heatmap & dispersion ellipse
│   │   │   ├── InverseResultsPanel.tsx # Predicted coordinates, dispersion, abstention advice
│   │   │   ├── InverseWorkspace.tsx   # Workspace container & layout coordinator
│   │   │   └── StressFieldInput.tsx   # Matrix import, paste, diagnostics, demo loader
│   │   ├── history/           # Case history components:
│   │   │   └── CaseHistoryWorkspace.tsx # Audit log table, filters, side-by-side comparison
│   │   ├── reporting/         # Engineering report modal:
│   │   │   └── ReportViewer.tsx       # Audit report layout with print CSS & export actions
│   │   └── layout/            # Shell structure (Header, Navigation, Footer)
│   ├── data/
│   │   └── sample_stress_field.json # Verified 64x64 confirmatory demo dataset (SPEC-I02)
│   ├── hooks/
│   │   ├── useBackendStatus.ts  # Background health & readiness polling
│   │   ├── useDebounce.ts       # Input debounce coordinator (180ms)
│   │   ├── useForwardAnalysis.ts # Forward workspace state, validation & prediction
│   │   └── useInverseAnalysis.ts # Inverse workspace state, validation & prediction
│   ├── services/
│   │   └── caseHistory.ts     # Immutable localStorage repository & SHA-256 hashing
│   ├── styles/
│   │   └── index.css          # Technical console tokens, engineering cards, print styles
│   ├── utils/
│   │   ├── colormap.ts        # Perceptually uniform Viridis colormap stops & RGB mapping
│   │   ├── exportUtils.ts     # JSON download, CSV matrix export, tabular summary CSV
│   │   └── matrixValidator.ts # 64x64 dimension validation, non-finite check, CSV/JSON parser
│   ├── App.tsx                # Master dashboard router & modal manager
│   ├── main.tsx               # DOM mount point
│   └── vite-env.d.ts          # Vite client environment typing
├── tests/
│   ├── api_client.test.ts             # API client serialization & typing unit tests
│   ├── geometry_rules.test.ts         # Canonical geometries & coordinate clamp unit tests
│   ├── inverse_and_reporting.test.ts  # Matrix validation, hashing, live D5 inverse tests
│   ├── live_backend_integration.test.ts # End-to-end integration tests against FastAPI
│   └── run_tests.ts                   # Master standalone test runner (npx tsx)
├── index.html                 # HTML shell
├── package.json               # Dependencies & scripts
├── tsconfig.json              # TypeScript 5 strict configuration
└── vite.config.ts             # Vite 5 dev server & proxy configuration
```

---

## 3. Getting Started

### Prerequisites
- Node.js >= 18.0.0 (tested on Node.js v22.12.0)
- Python 3.10+ running the FastAPI backend service on port 8000 (`python backend/run_server.py`)

### Installation
From the `frontend` directory:
```bash
npm install
```

### Development Server
Start the Vite development server with proxy forwarding:
```bash
npm run dev
```
The application will be accessible at `http://localhost:3000`. API requests to `/api/v1/*` are automatically proxied to `http://127.0.0.1:8000`.

### Production Build & Preview
To build optimized static assets:
```bash
npm run build
```
To serve the production build locally:
```bash
npm run preview -- --port 3000
```

---

## 4. Testing

The frontend includes an automated TypeScript test suite verifying geometry rules, matrix validation, cryptographic hashing, and live end-to-end integration against the running FastAPI backend.

Run the test suite:
```bash
npm test
```

### Test Results (12/12 Passing):
```
====================================================
SIMUSTRUCT REACT ENGINEERING DASHBOARD TEST SUITE
====================================================
  [TEST] Canonical Geometry Families (5 Plates)
[PASS] Canonical Geometry Families (0.4 ms)
  [TEST] Screen to Physical Coordinate Mapping Bounds
[PASS] Coordinate Clamping Bounds (0.1 ms)
  [TEST] ApiError Class Construction & Serialization
[PASS] ApiError Serialization (0.3 ms)
  [TEST] Forward Prediction Request Payload Typing
[PASS] Request Payload Typing (0.1 ms)
  [TEST] Live Backend Health and Readiness Probes
[PASS] Live Backend Health & Readiness (41.7 ms)
  [TEST] Live Model Registry Metadata Reflection
[PASS] Live Model Registry Reflection (6.1 ms)
  [TEST] Live Geometry Validation Workflow
[PASS] Live Geometry Validation (6.9 ms)
  [TEST] Live Forward Surrogate Prediction (Stage 3B Ensemble)
[PASS] Live Forward Surrogate Prediction (130.1 ms)
  [TEST] Matrix Diagnostics & Dimensional Validation
[PASS] Matrix Diagnostics & Validation (3.0 ms)
  [TEST] Colormap & Content Hashing
[PASS] Colormap & Content Hashing (2.0 ms)
  [TEST] Live Backend Inverse Localization (Stage 3D Model D5)
[PASS] Live Backend Inverse Localization (D5) (20.6 ms)
  [TEST] Live Backend Abstention Trigger & Error Handlers
[PASS] Live Backend Abstention & Error Handling (29.5 ms)

====================================================
FRONTEND TEST SUMMARY: 12 PASSED, 0 FAILED (TOTAL 12)
====================================================
```

---

## 5. Provenance & Engineering Standards

- **Explicit Provenance:** All predicted outputs are stamped with the `ML_SURROGATE` badge. The application explicitly discloses that predictions are generated by frozen statistical machine learning surrogates (Stage 3B Stacking Ensemble or Stage 3D Model D5 U-Net) rather than direct numerical solves of the continuum governing equations (FEM).
- **Physical Safety Gating:** The Forward Workspace automatically disables inference execution upon geometric boundary collision or central hole overlap.
- **Empirical Dispersion Semantics:** The spatial covariance ellipse and uncertainty radius returned by Model D5 represent empirical heatmap sharpness and must **not** be interpreted as statistically calibrated 95% confidence intervals.
- **Abstention Disclosure:** In accordance with the Stage 5C scientific audit on `SPEC-I01`, the application explicitly clarifies that an `ACCEPTED` status reflects a passed dispersion filter ($\rho \le 9.07\text{ mm}$), not an unconditional guarantee of ground-truth localization correctness.
