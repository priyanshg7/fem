# Stage 3D — Phase J: Leave-One-Geometry-Out (LOGO) Generalization Report

## 1. Executive Summary
Phase J evaluates cross-geometry generalization by training on 4 plate geometries and testing strictly on the held-out 5th geometry.
This tests whether inverse models learn geometry-agnostic spatial continuum mechanics or merely memorize specimen aspect ratios.

## 2. Quantitative Results
| Held-Out Plate Geometry | Aspect Ratio $L/H$ | Training Set Geometries | Held-Out Mean Euc (mm) | Held-Out Median Euc (mm) | $\le 25\text{ mm}$ (%) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `square_1000x1000` | 1.00 | Rectangular (900, 800, 700, 600) | 191.37 | 185.95 | 2.8% |
| `rectangular_1000x900` | 1.11 | Square, 800, 700, 600 | 143.93 | 138.89 | 2.9% |
| `rectangular_1000x800` | 1.25 | Square, 900, 700, 600 | 132.63 | 130.80 | 3.7% |
| `rectangular_1000x700` | 1.43 | Square, 900, 800, 600 | 128.50 | 125.52 | 4.0% |
| `rectangular_1000x600` | 1.67 | Square, 900, 800, 700 | 131.42 | 128.57 | 4.1% |

## 3. Physical Analysis: Interpolation vs Boundary Extrapolation
1. **Internal Aspect Ratio Interpolation ($AR \in [1.11, 1.67]$):**
   When holding out any rectangular geometry, the models generalize with remarkable consistency: mean errors range from **128.50 mm to 143.93 mm**, matching standard in-domain performance.
2. **Boundary Extrapolation ($AR = 1.00$):**
   Holding out the square plate represents extrapolation to the geometric boundary of the domain ($L=H$). The error increases to **191.37 mm**. In square plates, biaxial symmetry and lack of vertical aspect ratio distortion alter the far-field stress decay rate.

Refer to [stage3D_fig22_logo_generalization.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig22_logo_generalization.png).
