# Stage 4A — Model Artifact Freeze and Inference Contract Manifest

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 4A — Model Artifact Freeze and Backend Architecture  
**Author:** Computational Mechanics & Surrogate Modeling Research Group  
**Status:** Frozen & Audited for Production API Integration  
**Date:** November 2026  

---

## 1. Executive Summary

This manifest documents the formal freezing of the machine learning surrogate models for the **SimuStruct Digital Twin Platform**. Following the extensive numerical simulations in Stages 2–2V (51,625 FEM simulations), forward stacked ensemble development in Stage 3B, inverse spatial localization experiments in Stage 3D, and the close-out audit / robustness evaluation in Stage 3E, two primary surrogate architectures are selected, audited, and locked for production deployment:

1. **Primary Forward Surrogate:** `forward_ensemble_stage3b` — A heterogeneous meta-stacking ensemble combining four deep neural networks (`ANN-512`, `DeepMLP-D1`, `DeepMLP-D2`, `ResMLP`) and three gradient-boosted / tree ensembles (`RandomForest`, `ExtraTrees`, `HistGradientBoosting`) with a constrained Ridge meta-regressor.
2. **Primary Inverse Surrogate:** `inverse_unet_stage3d` (Model D5) — A fully convolutional U-Net with soft-argmax spatial density extraction, delivering continuous coordinate localization, anisotropic spatial covariance estimation, and physics-informed abstention filtering.

Every inference pipeline explicitly tags its outputs as `ML_SURROGATE` to prevent confusing statistical surrogate approximations with certified finite element analysis.

---

## 2. Checkpoint Inventory and Cryptographic Integrity

All checkpoints have been verified for integrity and are stored under version-controlled repository paths:

### 2.1 Forward Model Checkpoints (`models/stage3B_forward_stacking/`)

| File Name | Role | SHA-256 Hash Prefix | Size (KB) | Framework |
| :--- | :--- | :--- | :--- | :--- |
| `meta_framework4_heterogeneous.pkl` | Ridge Meta-Learner | `C5091D1C11C7C9E2C...` | 1.8 KB | Scikit-Learn 1.6+ |
| `stage3B_feature_scaler.pkl` | Standard Feature Scaler | `9C81426E447C2F023...` | 1.2 KB | Scikit-Learn 1.6+ |
| `baseline_ANN_512.pt` | Base Estimator: ANN-512 | `740CEFB9A444C12E3...` | 542 KB | PyTorch 2.6+ |
| `baseline_DeepMLP_D1.pt` | Base Estimator: DeepMLP-D1 | `0705761576D324825...` | 845 KB | PyTorch 2.6+ |
| `baseline_DeepMLP_D2.pt` | Base Estimator: DeepMLP-D2 | `F18C7BD2E46A45DD8...` | 1,220 KB | PyTorch 2.6+ |
| `baseline_ResMLP.pt` | Base Estimator: ResMLP | `0BFC1CB4467710B5C...` | 680 KB | PyTorch 2.6+ |
| `baseline_RandomForest.pkl` | Base Estimator: Random Forest | `738A1E27F71E8445E...` | 125,400 KB | Scikit-Learn 1.6+ |
| `baseline_ExtraTrees.pkl` | Base Estimator: Extra Trees | `71137C41FC65CF679...` | 118,200 KB | Scikit-Learn 1.6+ |
| `baseline_HistGradientBoosting.pkl`| Base Estimator: HistGB | `D0717ACDB9A1F187E...` | 4,320 KB | Scikit-Learn 1.6+ |

### 2.2 Inverse Model Checkpoints (`models/stage3D_spatial_inverse/`)

| File Name | Role | SHA-256 Hash Prefix | Size (KB) | Framework |
| :--- | :--- | :--- | :--- | :--- |
| `model_D5_unet.pt` | Model D5 U-Net Checkpoint | `510DB96EABED4C92D...` | 1,940 KB | PyTorch 2.6+ |
| `model_D5_metrics.json` | Audited Metrics & Thresholds | `F6B4DADEA3CBCB523...` | 4.8 KB | JSON Metadata |

---

## 3. Forward Surrogate Inference Contract (`forward_ensemble_stage3b`)

### 3.1 Input Feature Construction

The forward model accepts physical plate and defect dimensions and builds a 6-dimensional dimensionless feature vector:

$$\mathbf{x} = \begin{bmatrix} \tilde{x} \\ \tilde{y} \\ \tilde{d} \\ \alpha \\ \tilde{c}_{\text{lig}} \\ \tilde{c}_{\text{bnd}} \end{bmatrix} = \begin{bmatrix} x / L \\ y / H \\ d / D_c \\ L / H \\ c_{\text{lig}} / D_c \\ \min(c_{\text{bnd}}) / H \end{bmatrix} \in \mathbb{R}^6$$

Where:
- $L = 1000\text{ mm}$ (constant reference length).
- $H \in [600, 1000]\text{ mm}$ (variable plate height).
- $D_c = H / 12$ (canonical central circular hole diameter).
- $c_{\text{lig}} = \sqrt{(x - L/2)^2 + (y - H/2)^2} - \frac{D_c + d}{2}$ (ligament clearance).
- $c_{\text{bnd}} = \min(x - d/2, L - x - d/2, y - d/2, H - y - d/2)$ (boundary clearance).

### 3.2 Standardization and Ensemble Execution

1. **Standardization:**
   $$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
   Parameters loaded from `stage3B_feature_scaler.pkl`.
2. **Base Estimator Inference:**
   $$\hat{y}_i = f_i(\mathbf{z}), \quad i \in \{1, \dots, 7\}$$
   Where $i \in \{\text{ANN-512}, \text{DeepMLP-D1}, \text{DeepMLP-D2}, \text{ResMLP}, \text{RF}, \text{ET}, \text{HGB}\}$.
3. **Meta-Learner Aggregation:**
   $$\hat{K}_t = w_0 + \sum_{i=1}^7 w_i \hat{y}_i$$
   Coefficients loaded from `meta_framework4_heterogeneous.pkl`.
4. **Stress Dimensionalization:**
   $$\sigma_{vm,\text{max}} = \hat{K}_t \times \sigma_{\text{nom}}$$
   Nominal far-field traction is fixed at $\sigma_{\text{nom}} = 100.0\text{ psi}$ for standard comparisons.

### 3.3 Operating Domain and Physics Boundaries

| Parameter | Minimum | Maximum | Units | Enforcement Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Plate Length ($L$)** | 1000.0 | 1000.0 | mm | Strict Equality ($L=1000$) |
| **Plate Height ($H$)** | 600.0 | 1000.0 | mm | Domain Range Error if outside |
| **Aspect Ratio ($L/H$)** | 1.000 | 1.667 | — | Reject if outside $[1.0, 1.667]$ |
| **Defect Diameter ($d$)** | 5.0 | 50.0 | mm | Reject if non-positive |
| **Ligament Clearance ($c_{\text{lig}}$)** | $> 0.0$ | — | mm | Reject on collision ($\le 0$); warn if $< 12\text{ mm}$ |
| **Boundary Clearance ($c_{\text{bnd}}$)** | $> 0.0$ | — | mm | Reject on penetration ($\le 0$); warn if $< 12\text{ mm}$ |

### 3.4 Historical and Confirmatory Validation Performance

- **Original Test Dataset (51,625 samples):**
  - $R^2 = 0.9340$
  - Pearson correlation $r = 0.9665$
  - $\text{RMSE} = 2.65\text{ psi}$
- **New Blind Confirmatory Dataset (5,000 samples):**
  - $R^2 = 0.9634$
  - Pearson correlation $r = 0.9818$
  - $\text{RMSE} = 4.69\text{ psi}$
  - Mean Absolute Error $\text{MAE} = 2.45\text{ psi}$

---

## 4. Inverse Surrogate Inference Contract (`inverse_unet_stage3d` / Model D5)

### 4.1 Input Tensor Specification

The inverse model maps a full 2D Von Mises stress field directly to defect localization probabilities:

$$\mathbf{S} \in \mathbb{R}^{B \times 1 \times 64 \times 64}$$

- **Channel 1:** Normalized Von Mises stress $\bar{\sigma}_{vm} = \sigma_{vm} / \sigma_{\text{nom}}$.
- **Grid Resolution:** $64 \times 64$ uniform grid spanning $(x, y) \in [0, L] \times [0, H]$.
- **Anti-Leakage Boundary:** No geometric defect masks, void indicator channels, or defect coordinates are permitted in the input tensor.

### 4.2 Network Architecture and Soft-Argmax Coordinate Extraction

Model D5 uses an encoder-decoder U-Net with skip connections:
- **Encoder:** 4 downsampling blocks (DoubleConv + MaxPool2d), channels: $1 \to 32 \to 64 \to 128 \to 256$.
- **Bottleneck:** DoubleConv, channels: $256 \to 512$.
- **Decoder:** 4 upsampling blocks (ConvTranspose2d + Cat + DoubleConv), channels: $512 \to 256 \to 128 \to 64 \to 32$.
- **Output Head:** $1 \times 1$ Conv yielding logits $\mathbf{Z} \in \mathbb{R}^{1 \times 64 \times 64}$.
- **Spatial Softmax (Heatmap):**
  $$P_{ij} = \frac{\exp(Z_{ij} / \tau)}{\sum_{m=1}^{64} \sum_{n=1}^{64} \exp(Z_{mn} / \tau)}, \quad \tau = 1.0$$
- **Soft-Argmax Centroid:**
  $$\tilde{\mu}_x = \sum_{i,j} X_{ij} P_{ij}, \quad \tilde{\mu}_y = \sum_{i,j} Y_{ij} P_{ij}$$
  Continuous physical coordinates:
  $$\hat{x} = \tilde{\mu}_x \times L, \quad \hat{y} = \tilde{\mu}_y \times H$$

### 4.3 Anisotropic Spatial Covariance and Uncertainty Ellipse

The spatial dispersion of the predicted heatmap is computed via second central moments:

$$\boldsymbol{\Sigma}_{\text{norm}} = \begin{bmatrix} \sum_{i,j} (X_{ij} - \tilde{\mu}_x)^2 P_{ij} & \sum_{i,j} (X_{ij} - \tilde{\mu}_x)(Y_{ij} - \tilde{\mu}_y) P_{ij} \\ \sum_{i,j} (X_{ij} - \tilde{\mu}_x)(Y_{ij} - \tilde{\mu}_y) P_{ij} & \sum_{i,j} (Y_{ij} - \tilde{\mu}_y)^2 P_{ij} \end{bmatrix}$$

Physical covariance scaled by plate dimensions:

$$\boldsymbol{\Sigma}_{\text{phys}} = \begin{bmatrix} L^2 & 0 \\ 0 & H^2 \end{bmatrix} \boldsymbol{\Sigma}_{\text{norm}} = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{bmatrix}$$

Uncertainty ellipse parameters:
- **Eigenvalues:** $\lambda_1 \ge \lambda_2$ of $\boldsymbol{\Sigma}_{\text{phys}}$.
- **Semi-major axis:** $a = 2 \sqrt{\lambda_1}$.
- **Semi-minor axis:** $b = 2 \sqrt{\lambda_2}$.
- **Orientation:** $\theta = \frac{1}{2} \text{atan2}(2 \sigma_{xy}, \sigma_{xx} - \sigma_{yy})$.
- **Trace Dispersion Metric:**
  $$\rho = \sqrt{\text{Tr}(\boldsymbol{\Sigma}_{\text{phys}})} = \sqrt{\sigma_{xx} + \sigma_{yy}}$$

### 4.4 Automated Abstention Policy

To protect engineering decisions against hallucination or high-ambiguity fields:
- **Abstention Metric:** Trace dispersion $\rho$.
- **Optimal Decision Threshold:** $\theta_{\text{abst}} = 9.07\text{ mm}$.
- **Decision Logic:**
  $$\text{Status} = \begin{cases} \text{ACCEPTED} & \text{if } \rho \le 9.07\text{ mm} \\ \text{FLAGGED\_FOR\_REVIEW} & \text{if } \rho > 9.07\text{ mm} \end{cases}$$
- **Impact:** Filters $15\%$ of highest-uncertainty predictions (mostly severe hole/boundary ligament interactions), improving mean localization accuracy on accepted cases from $10.02\text{ mm}$ down to $6.41\text{ mm}$ ($36\%$ error reduction).

---

## 5. Artifact Audit Verification Sign-Off

The checkpoints and contracts listed in this manifest have been verified through automated integration tests (`backend/tests/run_tests.py`), yielding:
- $100\%$ numerical equivalence between API endpoints and standalone research scripts ($|K_{t,\text{API}} - K_{t,\text{standalone}}| < 10^{-4}$).
- 20 of 20 automated tests passing without warning.
- Checkpoint paths and weights frozen against further modification.
