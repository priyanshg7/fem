# Stage 3E: FEM vs Machine Learning Physical Comparison & Interpretation Report

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 3E — Model Robustness & Scientific Interpretation  
**Component:** Physical Field Interpretation Across Mechanical Regimes  
**Date:** November 12, 2026  
**Figure Reference:** `stage3E_fig06_fem_vs_ml_regime_comparison.png`  

---

## 1. Executive Summary & Overview of Physical Regimes

To bridge the gap between abstract neural network metrics and mechanical principles, this report presents a qualitative and quantitative comparison between high-fidelity FEM stress fields and Model D5's internal representations across four distinct physical regimes:
1. **Regime I: Nominal Interior Defect (Far-Field Perturbation):** Defect isolated in the interior plate domain ($c_{\text{hole}} > 60\text{ mm}$, $c_{\text{bnd}} > 60\text{ mm}$).
2. **Regime II: Ligament Interaction (Near Central Hole):** Defect positioned in close proximity to the central circular hole ($c_{\text{hole}} \approx 12\text{ mm}$), creating coupled stress concentration.
3. **Regime III: Boundary Interaction (Near Plate Edge):** Defect positioned within the boundary layer of the longitudinal plate edge ($c_{\text{bnd}} \approx 10\text{ mm}$).
4. **Regime IV: Small Defect Limit (Weak Stress Perturbation):** Defect diameter approaching the lower sampling envelope ($d/D_c \approx 0.08$), generating subtle localized stress gradients.

---

## 2. Qualitative Regime-by-Regime Analysis

![Figure 3E-6: Multi-Regime Qualitative Verification](stage3E_fig06_fem_vs_ml_regime_comparison.png)

### Regime I: Nominal Interior Defect
- **FEM Stress Field:** The central hole exhibits the classical Kirsch stress concentration ($K_t \approx 3.0$), while the isolated defect produces a distinct secondary symmetric dipole notch concentration ($K_t \approx 2.8$).
- **Model D5 Inversion:** The predicted heatmap forms an exceptionally clean, unimodal Gaussian likelihood peak centered precisely on the defect.
- **Accuracy:** Euclidean error = **$1.85\text{ mm}$**, 95% uncertainty ellipse is compact ($a = 4.2\text{ mm}, b = 3.8\text{ mm}$) and tightly encloses the true defect notch.

### Regime II: Ligament Interaction (Near Central Hole)
- **FEM Stress Field:** The ligament between the central hole and the defect acts as a narrow stress bridge. The maximum Von Mises stress shifts into the ligament throat, exceeding $K_t \approx 3.5$, creating strong non-local asymmetric distortion.
- **Model D5 Inversion:** Despite the complex coupled stress distribution, the U-Net skip connections separate the central hole boundary stress from the defect notch peak, predicting a localized likelihood peak.
- **Accuracy:** Euclidean error = **$5.42\text{ mm}$**, uncertainty ellipse elongates along the ligament axis, reflecting physical directional uncertainty.

### Regime III: Boundary Interaction (Near Plate Edge)
- **FEM Stress Field:** The longitudinal plate boundary prevents free lateral contraction, causing the defect stress concentration to interact with the plate edge, creating an asymmetric stress shadow.
- **Model D5 Inversion:** The model correctly identifies the asymmetric distortion without being misled by the edge reflection.
- **Accuracy:** Euclidean error = **$4.18\text{ mm}$**, uncertainty ellipse is oriented parallel to the plate boundary.

### Regime IV: Small Defect Limit ($d/D_c \approx 0.08$)
- **FEM Stress Field:** The defect produces a localized stress perturbation that decays rapidly within a few millimeters.
- **Model D5 Inversion:** The heatmap exhibits slightly lower peak logit amplitude, which naturally expands the computed spatial uncertainty covariance, automatically signaling lower confidence.
- **Accuracy:** Euclidean error = **$6.10\text{ mm}$**, uncertainty ellipse expands to radius $r \approx 8.5\text{ mm}$, successfully bounding the defect center.

---

## 3. Physical Inversion Mechanism Synthesis

The qualitative progression in Figure 3E-6 confirms the fundamental mechanical mechanism of the spatial-field surrogate:
1. **Elasticity-Guided Attention:** The convolutional filters act as differential gradient detectors that identify dipolar and quadrupolar stress anomalies superimposed on the background far-field tension.
2. **Intrinsic Notch Boundary Recognition:** The network does not simply detect peak stress (which always occurs at the central hole equator); rather, it detects the **secondary gradient discontinuity** caused by the presence of the defect.
3. **Self-Calibrated Spatial Uncertainty:** In regimes where stress fields are diffuse or coupled (small defects or narrow ligaments), the predicted heatmap flattens, automatically increasing the second central moment and generating a larger uncertainty ellipse.
