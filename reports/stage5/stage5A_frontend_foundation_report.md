# Stage 5A Technical Report: React Frontend Foundation, TypeScript Architecture, and Design System

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Application:** SimuStruct Digital Twin Platform  
**Stage:** Stage 5A — React Frontend Foundation, TypeScript, API Client, Navigation, Shared State, and Technical Design System  
**Date:** November 2026  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Stage 5A establishes the frontend foundation for the **SimuStruct Digital Twin Platform**, transitioning the validated research models (Stages 3A–3E) and production FastAPI backend (Stages 4A–4D) into an interactive, high-performance web dashboard.

The frontend is implemented as a single-page React 18 application in TypeScript 5, powered by Vite 5. It features a technical engineering console design system tailored for structural engineering analysis, a centralized strictly-typed API client mirroring the Stage 4D backend contract, background backend health and model readiness tracking, and an extensible dual-workspace navigation architecture.

All foundational components, types, and client routines have been verified via automated unit and integration tests, and the production build compiles with zero TypeScript warnings or bundling errors (`190 kB` gzip-optimized bundle).

---

## 2. Architectural Architecture & Stack Selection

| Component | Technology | Rationale |
|---|---|---|
| **Build & Bundler** | **Vite 5.4** | Near-instant HMR, ES module native architecture, efficient tree-shaking, and built-in reverse proxy forwarding. |
| **UI Framework** | **React 18.3** | Component modularity, functional hooks (`useCallback`, `useMemo`, `useRef`), and robust state lifecycle management. |
| **Language** | **TypeScript 5.6** | Full type safety, strict null checking, and exact structural mirroring of Stage 4D Pydantic schemas. |
| **Icons** | **Lucide React** | Clean, lightweight SVG glyphs for technical indicators, warnings, geometries, and operational controls. |
| **Testing** | **Node.js TSX Runner** | Rapid, zero-overhead TypeScript test execution without heavy testing framework baggage. |

---

## 3. Technical Design System ("Engineering Console")

A purpose-built CSS design system was constructed in `src/styles/index.css` adhering to professional computational software standards (e.g., Abaqus/ANSYS workbenches, Grafana, modern CAD consoles):

1. **Color Palette:**
   - **Background:** Slate Black (`#0a0d14`) providing high visual contrast and reduced fatigue during extended engineering sessions.
   - **Surfaces & Cards:** Dark Navy Slate (`#111622`) with subtle border elevation (`#1f293d`).
   - **Brand Accent:** Precision Cyan (`#00b4d8` / `#0096c7`) representing structural tension and digital twin instrumentation.
   - **Semantic Statuses:**
     - Success / Healthy: Emerald Green (`#10b981`)
     - Warning / Approaching Limit: Amber Gold (`#f59e0b`)
     - Danger / Collision: Crimson Red (`#ef4444`)
     - Informational: Electric Cyan (`#06b6d4`)
2. **Typography:**
   - Primary: `Inter`, sans-serif (clean labels, descriptions, and UI controls).
   - Numerical / Coordinates: `JetBrains Mono` / monospace (tabular alignment of coordinates, stresses, and clearances).
3. **Reusable Component Primitives (`src/components/common/`):**
   - `Card.tsx`: Tactile container supporting optional header titles, badges, and action bars.
   - `Badge.tsx`: Semantic status pill supporting `cyan`, `green`, `amber`, `red`, and `gray` variants.
   - `Button.tsx`: High-contrast engineering buttons with interactive hover/active states, loading spinners, and disabled gating.
   - `Alert.tsx`: Dismissible and persistent alert banners with contextual icons for collision warnings and stale result notices.

---

## 4. Centralized TypeScript API Client

The API client (`src/api/client.ts`) serves as the single source of truth for network communication with the backend service (`/api/v1`):

### 4.1 Strict Schema Mirroring (`src/api/types.ts`)
The TypeScript interfaces directly match the Stage 4D OpenAPI / Pydantic definitions:
- `HealthResponse` & `ReadinessResponse`: Backend status, model inventory, and hardware execution device.
- `ModelDetail` & `ModelRegistryResponse`: Metadata, versioning, checkpoints, and aspect-ratio domain limits.
- `GeometryValidationRequest` & `GeometryValidationResponse`: 6-parameter plate definition, clearance metrics, and validation codes.
- `ForwardPredictionRequest` & `ForwardPredictionResponse`: Traction, geometry, selected framework, predicted stress, uncertainty, and base model breakdown.
- `InversePredictionResponse`: Defect coordinate predictions, localization error, confidence, and heatmap telemetry.

### 4.2 Environmental Adaptability
The client handles both browser runtime (via Vite development proxy `/api/v1`) and headless Node.js testing:
```typescript
const API_BASE_URL =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL) ||
  (typeof globalThis !== 'undefined' && (globalThis as any).process?.env?.VITE_API_URL) ||
  '/api/v1';
```

### 4.3 Error Normalization & Timeout Handling
Network requests include an automated 10-second `AbortController` timeout and parse backend HTTP 400/422/500 JSON error structures into a structured `ApiError` instance, capturing status codes, machine-readable error codes, and field-level validation details.

---

## 5. Layout, Shared State, and Navigation

### 5.1 Shell Layout
- **Header (`src/components/layout/Header.tsx`):** Displays platform title, system status, active forward/inverse model badges, and live backend connection pulse indicator (`Live Backend Connected` vs `Connecting...`).
- **Navigation Bar (`src/components/layout/Navigation.tsx`):**
  - **Forward Stress Analysis:** Active workspace for Stage 5A/5B.
  - **Inverse Defect Inspection:** Planned workspace for Stage 5C, displaying a `Stage 5C Planned` badge to clearly delineate implementation phases.
- **Footer (`src/components/layout/Footer.tsx`):** Displays backend version (`1.0.0`), service identity, and explicit disclaimer that predictions are machine-learning surrogates.

### 5.2 Background Status Hook (`src/hooks/useBackendStatus.ts`)
A dedicated React hook polls `/api/v1/health` and `/api/v1/readiness` on mount and every 30 seconds. It exposes reactive state flags (`isOnline`, `isReady`, `healthData`, `models`) ensuring that the user is immediately alerted if the surrogate service becomes unreachable.

---

## 6. Verification and Validation Results

The foundation was verified through automated unit tests in `frontend/tests/`:

1. **`tests/geometry_rules.test.ts`:**
   - Validated canonical dimensions and aspect ratios for all 5 plate configurations ($1000\times 1000$ to $1000\times 600$).
   - Validated physical coordinate transformation and bounding clamping functions.
2. **`tests/api_client.test.ts`:**
   - Validated `ApiError` class creation, message preservation, and JSON error envelope parsing.
   - Validated `ForwardPredictionRequest` JSON serialization and payload typing.
3. **Build Pipeline:**
   - Production bundle compiled with `tsc && vite build`:
     - `dist/index.html`: `1.10 kB` (gzip: `0.62 kB`)
     - `dist/assets/index-CHo9fU0X.css`: `4.56 kB` (gzip: `1.56 kB`)
     - `dist/assets/index-D5ZhNJK2.js`: `190.83 kB` (gzip: `57.75 kB`)
     - Build duration: `2.51s`
     - Status: Zero TypeScript compilation errors, zero warnings.

---

## 7. Stage 5A Conclusion

Stage 5A successfully establishes the technical engineering frontend foundation. All navigation, API communication, TypeScript definitions, and styling conventions are in place, creating a solid base for the Forward Engineering Analysis Workspace (Stage 5B).

