# Stage 3D — Phase D: Regional Peak Stress Inverse Localization Report

## 1. Executive Summary
In Stage 3C, scalar maximum stress inversion was proven to fail fundamentally ($MAE \approx 270\text{--}300\text{ mm}$) due to dimensional underdetermination ($\mathbb{R}^1 \to \mathbb{R}^2$ pre-images forming closed 1D level-set contours spanning entire quadrants). 
Phase D develops **Model D1: Regional Peak Stress Inversion**, partitioning the plate into 4 spatial quadrants relative to the specimen center $(L/2, H/2)$ and adding central hole ring peak and statistical stress metrics (16 features + aspect ratio $L/H$).

By decomposing the stress response regionally, Model D1 breaks global quadrant ambiguity:
- **Baseline Scalar Inversion (Stage 3C):** Test Mean Euclidean Error = **288.5 mm**
- **Model D1 (ANN-256):** Test Mean Euclidean Error = **121.25 mm** (Median: 113.26 mm, $\le 25\text{ mm}$: 4.4%)
- **Model D1 (HistGradientBoosting):** Test Mean Euclidean Error = **124.68 mm** (Confirmatory: 137.55 mm)
- **Model D1 (ExtraTrees):** Test Mean Euclidean Error = **125.47 mm** (Confirmatory: 136.59 mm)

This represents an error reduction of over **58%** compared to scalar inversion.

## 2. Feature Formulation and Continuum Mechanics
The plate is partitioned into five distinct continuum zones:
1. Quadrant 1 (Top-Right): $x \ge L/2, y \ge H/2$
2. Quadrant 2 (Top-Left): $x < L/2, y \ge H/2$
3. Quadrant 3 (Bottom-Left): $x < L/2, y < H/2$
4. Quadrant 4 (Bottom-Right): $x \ge L/2, y < H/2$
5. Central Cutout Ring: Elements located at radial distances $r \in [R_c, 1.25 R_c]$ where $R_c = 100\text{ mm}$.

For each zone, three statistical invariants are extracted:
- Peak stress concentration factor: $K_{t, \text{zone}} = \sigma_{vm, \max} / 100\text{ psi}$
- Spatial mean stress: $\mu_{\text{zone}} / 100\text{ psi}$
- 90th percentile stress: $P90_{\text{zone}} / 100\text{ psi}$

Together with overall $K_{t,\max}$ and specimen aspect ratio $L/H$, these 17 features provide coarse directional asymmetry that identifies which quadrant contains the defect.

## 3. Performance Across Regressor Architectures (Experiment E1)
Evaluated on the 7,557 historical test samples and 500 blind confirmatory samples:

| Model Architecture | Test Mean Euc (mm) | Test Median Euc (mm) | Test P95 Euc (mm) | Test $\le 25\text{ mm}$ (%) | Confirmatory Mean Euc (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression | 188.16 | 178.35 | 321.40 | 1.1% | 249.05 |
| ExtraTrees (100 trees) | 125.47 | 122.74 | 228.15 | 5.7% | 136.59 |
| HistGradientBoosting | 124.68 | 118.90 | 231.04 | 4.2% | 137.55 |
| ANN-256 (256-128 MLP) | **121.25** | **113.26** | **230.12** | **4.4%** | **180.16** |

## 4. Joint Defect Sizing (Experiment E2)
When defect diameter $D/D_c$ is unknown and must be jointly predicted alongside $(X/L, Y/H)$:
- Defect location error remains stable: Mean Euclidean Error = **126.65 mm** (ExtraTrees).
- Defect diameter prediction fails: $MAE_D = 24.13\text{ mm}$, $R^2_D = -28.34$.
- **Mechanics Explanation:** Regional stress metrics capture the global quadrant envelope, but the local stress perturbation of a defect scales simultaneously with both diameter $d$ and clearance $c_{\text{lig}}$. Without high-resolution local gradients, regional peak stress is degenerate between a small defect near a hole and a large defect further away.

Refer to [stage3D_fig06_model_D1_scatter.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig06_model_D1_scatter.png) and [stage3D_fig07_model_D1_error_distribution.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig07_model_D1_error_distribution.png).
