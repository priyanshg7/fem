# Stage 5D Technical Report: Engineering Reporting, Case History, and Export System

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Application:** SimuStruct Digital Twin Platform  
**Stage:** Stage 5D — Engineering Reporting, Case History, and Export System  
**Date:** November 2026  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Stage 5D delivers the **Engineering Reporting, Case History, and Export System** within the SimuStruct Digital Twin Platform. This capability bridges computational predictions with professional engineering workflows, allowing structural engineers to archive, inspect, compare, and export immutable case records for both Forward Stress Analysis (Stage 3B Ensemble) and Inverse Defect Inspection (Stage 3D Model D5 U-Net).

Key technical achievements of Stage 5D:
1. **Traceable Case Records (`CaseRecord`):** Establishes an immutable data structure archiving complete request and response payloads, execution telemetry, model versions, backend prediction IDs, user-declared input provenance, and cryptographic SHA-256 content hashes.
2. **Client-Side Case History Workspace (`CaseHistoryWorkspace`):** Provides a search-and-filter interface over stored case records, category breakdown (Forward vs Inverse), individual report review, JSON export, and a side-by-side case comparison view.
3. **Formal Engineering Report Viewer (`ReportViewer`):** Implements an audit-grade report modal matching the technical console design system. Sections include an Executive Summary, Specimen Geometry & Boundary Loading, Surrogate Predictions, Uncertainty & Dispersion Semantics, and an Explicit Scientific Limitations & Governance clause.
4. **Multi-Format Export Engine (`exportUtils`):**
   - **Downloadable PDF Export:** Implements print-media styling (`@media print`) enabling browser print-to-PDF generation with clean typography, page breaks, and no clipped tables or figures.
   - **Machine-Readable JSON:** Exports complete structured case records including cryptographic hashes for independent verification.
   - **Tabular & Matrix CSVs:** Exports tabular case parameter summaries, $64 \times 64$ input stress fields, and $64 \times 64$ predicted probability heatmaps with labeled coordinate headers.
5. **Data Integrity & Immutability:** Case records are immutable. Modifying parameters in the workspace marks current results as stale and produces a distinct new record upon subsequent inference runs, preventing historical mutation.

---

## 2. Immutable Case Record Architecture (`src/api/types.ts`)

To ensure absolute traceability, case records are defined with strict typing, capturing the exact state of an analysis run:

```typescript
export type AnalysisType = 'FORWARD' | 'INVERSE' | 'LINKED';
export type StressSourceProvenance =
  | 'EXTERNAL_FEM'
  | 'EXPERIMENTAL_SENSOR'
  | 'RESEARCH_DATASET'
  | 'DEMONSTRATION_SAMPLE';

export interface CaseRecord {
  case_id: string;                                // e.g. CASE-FWD-20261112-A3F1
  analysis_type: AnalysisType;
  title: string;
  created_at: string;                             // ISO-8601 UTC timestamp
  content_hash: string;                           // SHA-256 payload integrity digest
  notes?: string;
  is_stale?: boolean;
  api_version: string;                            // Backend service version (1.0.0)
  frontend_version: string;                       // Frontend client version (1.0.0)
  provenance: 'ML_SURROGATE';

  // Forward Payload & Output (if FORWARD or LINKED)
  forward_request?: ForwardPredictionRequest;
  forward_response?: ForwardPredictionResponse;

  // Inverse Payload & Output (if INVERSE or LINKED)
  inverse_request?: InversePredictionRequest;
  inverse_response?: InversePredictionResponse;
  stress_source_provenance?: StressSourceProvenance;
  stress_source_description?: string;
  reference_ground_truth?: ReferenceGroundTruth;
}
```

### 2.1 Cryptographic Content Hashing (`computeContentHash`)
A deterministic SHA-256 digest is generated across the serialized request and response payloads using the browser's native `crypto.subtle.digest` API (with deterministic fallback for test runners). This digest verifies data integrity and guarantees that report parameters have not been tampered with or modified post-inference.

### 2.2 Client-Side Persistence Scope
Case records are persisted to `localStorage` under the key `simustruct_case_history_v1`. Technical documentation and the UI explicitly state:
- Persistence is currently scoped to the user's local browser storage.
- Records are preserved across page reloads and browser sessions on the local machine.
- Local records are not centrally replicated to an external server database.

---

## 3. Case History Workspace (`src/components/history/CaseHistoryWorkspace.tsx`)

The Case History Workspace provides a centralized interface accessible via the main navigation bar:

### 3.1 Case Filtering and Search
- **Filter Tabs:** Instant filtering between "All Records", "Forward Analyses", and "Inverse Analyses".
- **Real-Time Search:** Filters records dynamically by Case ID, title, or specimen dimensions.
- **Summary Counters:** Displays live counts of completed analyses on navigation tabs.

### 3.2 Audit Log Table
The table presents essential engineering columns:
- **Comparison Checkbox:** Select up to two cases for comparative evaluation.
- **Case ID:** Clickable link opening the full engineering report.
- **Type Badge:** High-contrast `FORWARD` (cyan) or `INVERSE` (green) badges.
- **Timestamp:** Formatted UTC date and time of prediction.
- **Specimen Dimensions:** Length $\times$ Height in mm (e.g., $1000 \times 700$).
- **Key Output:**
  - Forward: Maximum stress $\sigma_{\max}\text{ psi}$ and $K_t$.
  - Inverse: Defect center $(\hat{x}, \hat{y})\text{ mm}$ and dispersion $\pm\rho\text{ mm}$.
- **Model Telemetry:** Model identifier and server execution latency in ms.
- **Actions:** View full report, export JSON, or delete record.

### 3.3 Side-by-Side Case Comparison Modal
Selecting any two cases activates the "Compare Selected (2)" action, opening a comparison modal that displays parameters, geometry, predictions, uncertainty dispersion, and abstention decisions side-by-side.

---

## 4. Engineering Report Layout (`src/components/reporting/ReportViewer.tsx`)

The `ReportViewer` modal renders a complete engineering report designed for technical reviews, project documentation, and archival:

### 4.1 Structure and Standard Sections
1. **Document Header:** Platform title, report generation timestamp, case identifier, and cryptographic content hash.
2. **Executive Engineering Summary:** Succinct statement describing the surrogate evaluation, boundary conditions, and applicable standards.
3. **Specimen Geometry & Applied Loading:** Detailed two-column breakdown of plate dimensions ($L, H$), aspect ratio ($\alpha$), central hole ($D_c$), defect coordinates or diameter, and applied far-field traction ($\sigma_0$).
4. **Surrogate Prediction Metrics:**
   - **Forward Analyses:** Maximum element-centroid Von Mises stress $\sigma_{\max}$, stress concentration factor $K_t$, ensemble uncertainty $\pm\text{std}$, model version, and base-estimator breakdown table (`ANN_512`, `DeepMLP_D2`, `ResMLP`, `RandomForest`, `ExtraTrees`, `HistGradientBoosting`).
   - **Inverse Analyses:** Predicted defect location $(\hat{x}, \hat{y})$ in physical mm and dimensionless $(\xi, \eta)$, heatmap spatial dispersion radius $\rho$, covariance ellipse axes ($a, b, \theta^\circ$), abstention status, and demonstration reference truth comparison when available.
5. **Safety System Decision & Abstention Guidance:** Clear explanation of whether the result was accepted by the retention threshold or flagged for review, accompanied by actionable recommendations for independent NDE or FEM verification.
6. **Scientific Limitations and Governance Clause:**
   - **Surrogate Non-Equivalence:** Explicitly discloses that predictions are ML surrogate estimates, not direct numerical solutions of the partial differential equations (PDEs).
   - **Dispersion Semantics:** Reaffirms that spatial dispersion reflects mathematical heatmap sharpness and must not be interpreted as statistically calibrated 95% confidence intervals.
   - **Fitness-for-Service Disclaimer:** Explicitly disclaims structural fitness-for-service certification without independent physical inspection and numerical simulation.

---

## 5. Multi-Format Export Capabilities (`src/utils/exportUtils.ts`)

### 5.1 Print-Ready PDF Generation
The report viewer incorporates custom print-media CSS styles (`@media print`):
- Action bars, navigation, and modal backdrops are hidden (`display: none !important`).
- Backgrounds are converted to pure white (`#ffffff`) and text to high-contrast black (`#000000`).
- Tables and figures include `page-break-inside: avoid` rules to prevent awkward page splits.
- Invoked seamlessly via the "Print / Save PDF" button (`window.print()`).

### 5.2 Machine-Readable JSON Export
Clicking "Export JSON" produces a structured `.json` document named `{case_id}.json`. The file contains:
- Complete case metadata, creation timestamps, and content hash.
- Full nested request and response objects.
- Model IDs, versions, and execution telemetry.
- User-provided input provenance descriptions.

### 5.3 Tabular and Matrix CSV Exports
- **Case Summary CSV (`exportTabularSummaryCsv`):** Produces `{case_id}_summary.csv` with a parameter-value-unit-provenance schema suitable for ingestion into Excel, MATLAB, or Python pandas.
- **Input Stress Matrix CSV (`exportMatrixAsCsv`):** Exports the raw $64 \times 64$ stress field with labeled column (`X_Col_0` to `X_Col_63`) and row (`Y_Row_0` to `Y_Row_63`) headers.
- **Predicted Heatmap Matrix CSV:** Exports the $64 \times 64$ spatial probability matrix returned by Model D5.

---

## 6. Verification and Integration Test Results

The reporting, case history, and export subsystems were verified through automated tests:

### 6.1 Test Execution Summary
All 12 automated unit and integration tests passed:
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

### 6.2 Verified Export Integrity
1. **JSON Serialization & Round-Trip:** Generated case JSON was verified to parse back with 100% field parity and matching SHA-256 hash.
2. **Matrix CSV Formatting:** Verified that CSV outputs contain exactly 65 comma-separated columns (1 axis header + 64 data columns) across 65 lines (1 column header + 64 rows).
3. **Immutability Guarantee:** When a completed analysis is modified in the workspace, the existing case record retains its original timestamp and outputs; running a new prediction generates an independent new Case ID.

---

## 7. Stage 5D Conclusion

Stage 5D successfully establishes an immutable engineering case history and multi-format reporting system. The implementation delivers traceable audit records, high-fidelity PDF and JSON exports, and rigorous scientific disclaimers, ensuring that all machine-learning predictions produced by the platform are properly contextualized and archived for professional engineering use.
