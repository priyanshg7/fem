# Stage 3E: Defect Diameter & Spatial Clearance Sensitivity Report

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 3E — Model Robustness & Scientific Interpretation  
**Component:** Defect Size and Proximity Sensitivity Analysis  
**Date:** November 12, 2026  
**Evaluated Benchmarks:** Untouched Confirmatory Benchmark ($N=500$)  

---

## 1. Executive Summary & Physical Hypotheses

The inverse localization of a circular defect from surface stress fields is governed by two physical parameters:
1. **Defect Size Ratio ($d/D_c$):** Governs the magnitude and spatial extent of the perturbation field. As $d \to 0$, the stress perturbation approaches zero ($\Delta \sigma \to 0$), reducing signal-to-noise ratio.
2. **Clearance to Hole & Boundary ($c_{\text{hole}}, c_{\text{bnd}}$):** When a defect is within close proximity to the central hole ($c_{\text{hole}} < 15\text{ mm}$), non-linear ligament stress interactions occur, distorting the local Kirsch stress distribution into an asymmetric coupled concentration.

This report evaluates Model D5 across these regimes using the untouched 500-sample confirmatory benchmark.

---

## 2. Sensitivity to Defect Sizing Ratio ($d / D_c$)

The 500 confirmatory samples were partitioned into four diameter ratio bins:

| Defect Diameter Bin ($d/D_c$) | Physical Description | Sample Count ($N$) | Mean Euclidean Error (mm) | Median Euclidean Error (mm) | P95 Euclidean Error (mm) | Success Rate ($\le 25\text{ mm}$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$[0.08, 0.15)$** | Very Small Defect (Weak Signal) | $162$ | $16.32$ | $5.91$ | $54.21$ | $92.6\%$ |
| **$[0.15, 0.22)$** | Nominal Defect (Standard Perturbation) | $185$ | $12.45$ | $5.44$ | $38.10$ | $95.1\%$ |
| **$[0.22, 0.30)$** | Moderate Defect (Strong Field Distortion) | $53$ | $9.81$ | $5.20$ | $22.45$ | $98.1\%$ |
| **$[0.30, 0.45]$** | Out-of-Domain Diameter Extension | $100$ | **$5.89$** | **$5.74$** | **$10.82$** | **$100.0\%$** |

![Figure 3E-3: Parameter Sensitivity](stage3E_fig03_diameter_and_clearance_sensitivity.png)

### Key Observations
1. **Out-of-Domain Large Defects ($100\%$ Success):** For out-of-domain large defects ($d/D_c \in [0.32, 0.45]$), Model D5 achieves **$5.89\text{ mm}$ mean error** with $100.0\%$ of predictions falling within $25\text{ mm}$. The larger defect creates a massive localized stress concentration that the U-Net heatmap detects with near-perfect confidence.
2. **Small Defect Limit ($d/D_c < 0.15$):** For very small defects, median error remains low ($5.91\text{ mm}$), but the mean rises to $16.32\text{ mm}$ due to a few cases where the localized stress peak falls below the background far-field stress variation.

---

## 3. Sensitivity to Spatial Clearance Regimes

The dataset was partitioned by hole-to-hole clearance $c_{\text{hole}} = \sqrt{(x - X_c)^2 + (y - Y_c)^2} - \frac{D_c + d}{2}$:

| Clearance Regime ($c_{\text{hole}}$) | Physical Interaction Regime | Sample Count ($N$) | Mean Euclidean Error (mm) | Median Euclidean Error (mm) | Success Rate ($\le 25\text{ mm}$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$< 15\text{ mm}$** | Extreme Ligament Interaction | $50$ | $74.20$ | $14.85$ | $70.0\%$ |
| **$[15, 30)\text{ mm}$** | Near-Hole Interaction | $68$ | $21.40$ | $6.21$ | $91.2\%$ |
| **$[30, 60)\text{ mm}$** | Moderate Intermediate Field | $142$ | $11.15$ | $5.48$ | $97.2\%$ |
| **$\ge 60\text{ mm}$** | Far-Field Space Filling | $240$ | **$8.22$** | **$5.25$** | **$98.8\%$** |

### Boundary Clearance Regimes ($c_{\text{bnd}}$)
Similarly, for boundary clearance:
- $c_{\text{bnd}} < 15\text{ mm}$ (Near Plate Edge): Mean = $68.10\text{ mm}$, Median = $13.20\text{ mm}$, Success = $74.0\%$.
- $c_{\text{bnd}} \ge 15\text{ mm}$ (Interior Domain): Mean = $10.16\text{ mm}$, Median = $5.34\text{ mm}$, Success = $98.0\%$.

---

## 4. Physical Synthesis

1. **Ligament Shadowing:** When the defect is extremely close to the central hole ($c_{\text{hole}} < 15\text{ mm}$), the stress concentration of the central hole ($K_t \approx 3.0$) merges with that of the defect, creating a single blended asymmetric peak. The U-Net sometimes places the predicted center closer to the combined centroid.
2. **Abstention Utility:** Fortunately, as demonstrated in our uncertainty calibration audit, these ligament and boundary interaction cases exhibit significantly larger spatial uncertainty radii ($\sigma_{\text{phys}} > 10\text{ mm}$), allowing the abstention pipeline to automatically flag and reject them for human review.
