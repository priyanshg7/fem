# Stage 3E: Geometry & Leave-One-Geometry-Out (LOGO) Generalization Report

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 3E — Model Robustness & Scientific Interpretation  
**Component:** Multi-Geometry Scaling and LOGO Cross-Validation  
**Date:** November 12, 2026  
**Plate Aspect Ratios:** $L/H \in \{1.000, 1.111, 1.250, 1.429, 1.667\}$  

---

## 1. Executive Summary & Problem Formulation

In structural mechanics, changing the aspect ratio of a finite perforated plate under uniaxial tension alters the boundary stiffness, Saint-Venant decay distance, and transverse Poisson contractions. An inverse defect localizer must demonstrate that its spatial reasoning is not brittle to aspect ratio variations.

This report evaluates Model D5 under two rigorous regimes:
1. **In-Domain & Confirmatory Evaluation ($N=500$ untouched):** Testing across all 5 geometry families where the training distribution spanned the aspect ratios.
2. **Leave-One-Geometry-Out Cross-Validation (LOGO):** Rigorously training Model D5 on 4 geometries and testing on the 5th entirely held-out geometry family, testing extrapolation to unseen aspect ratios.

---

## 2. Multi-Geometry Confirmatory Benchmark Results ($N=500$)

The untouched 500-sample confirmatory benchmark (100 samples per geometry) yielded the following breakdown:

| Geometry Family | Specimen Dimensions ($L \times H\text{ mm}$) | Aspect Ratio ($L/H$) | Canonical $D_c$ (mm) | Confirmatory Mean Euc (mm) | Confirmatory Median Euc (mm) | Success Rate ($\le 25\text{ mm}$) | Success Rate ($\le 10\text{ mm}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `square_1000x1000` | $1000 \times 1000$ | $1.000$ | $83.33$ | $27.14$ | $7.07$ | $94.0\%$ | $84.0\%$ |
| `rectangular_1000x900` | $1000 \times 900$ | $1.111$ | $75.00$ | $17.88$ | $5.86$ | $95.0\%$ | $88.0\%$ |
| `rectangular_1000x800` | $1000 \times 800$ | $1.250$ | $66.67$ | $23.82$ | $5.70$ | $93.0\%$ | $86.0\%$ |
| `rectangular_1000x700` | $1000 \times 700$ | $1.429$ | $58.33$ | $19.32$ | $5.73$ | $92.0\%$ | $87.0\%$ |
| `rectangular_1000x600` | $1000 \times 600$ | $1.667$ | $50.00$ | $26.76$ | $5.78$ | $92.0\%$ | $86.0\%$ |
| **All Geometries** | — | — | — | **$22.99$** | **$6.00$** | **$93.2\%$** | **$86.2\%$** |

> [!NOTE]
> Notice that the median localization error across all rectangular geometries is extraordinarily tight: **$5.70\text{ mm}$ to $5.86\text{ mm}$**! The higher mean error ($17\text{--}27\text{ mm}$) is driven by the 20 physically challenging boundary and ligament configurations included in each geometry slice.

---

## 3. Leave-One-Geometry-Out (LOGO) Extrapolation Performance

When an entire plate aspect ratio is held out during training, Model D5 achieves the following performance on the unseen geometry:

| Held-Out Geometry Family | Held-Out Aspect Ratio ($L/H$) | LOGO Mean Euclidean Error (mm) | LOGO Median Euclidean Error (mm) | LOGO P95 Euclidean Error (mm) | LOGO Success Rate ($\le 25\text{ mm}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `square_1000x1000` | $1.000$ | $14.31$ | $12.11$ | $24.01$ | $96.5\%$ |
| `rectangular_1000x900` | $1.111$ | $5.75$ | $4.97$ | $12.05$ | $99.0\%$ |
| `rectangular_1000x800` | $1.250$ | $9.37$ | $5.04$ | $11.38$ | $97.5\%$ |
| `rectangular_1000x700` | $1.429$ | $6.29$ | $5.21$ | $11.22$ | $98.0\%$ |
| `rectangular_1000x600` | $1.667$ | $7.19$ | $4.27$ | $10.08$ | $99.0\%$ |
| **LOGO Overall Average** | — | **$8.58$** | **$6.32$** | **$13.75$** | **$98.0\%$** |

![Figure 3E-2: Geometry & LOGO Generalization](stage3E_fig02_geometry_and_logo_generalization.png)

---

## 4. Mechanical Interpretation & Aspect Ratio Invariance

1. **Why Model D5 Excels on Unseen Geometries:** In standard fully connected networks or tree ensembles (e.g. Model D1), changing the aspect ratio changes the relative positions of regional quadrants, creating severe feature shifts that caused Model D1 to fail ($165.58\text{ mm}$ error). In contrast, Model D5 maps continuous normalized stress gradients directly into a 2D spatial heatmap. The local stress disturbance surrounding a defect is governed by elasticity theory ($\sim 1/r^2$ decay), which remains invariant under normalized coordinate transformations!
2. **Post-Processing Un-Normalization:** Because Model D5 predicts dimensionless coordinates $(\xi, \eta) \in [0, 1]^2$, physical coordinates are scaled via $(X, Y) = (L \xi, H \eta)$. The network learns the dimensionless topology of stress concentrations, allowing seamless transfer across plate aspect ratios.
