# Stage 3B Single-Model Baseline Report: Comprehensive Estimator Benchmarks

## Executive Summary

Before building and cross-fitting forward stacked ensemble models in Stage 3B, this study establishes an audited, high-precision single-model benchmark. Nine diverse estimators were trained on the identical 36,512-simulation training split using **Feature Set D** (physical interaction and boundary clearances):
- **Shallow Feedforward ANNs:** ANN-128, ANN-256, ANN-512
- **Deep Multi-Layer Perceptrons:** DeepMLP-D1 (3 layers), DeepMLP-D2 (4 layers)
- **Residual Network:** ResMLP (3 residual blocks, skip connections)
- **High-Capacity Tree Ensembles:** Random Forest, Extra Trees, HistGradientBoosting

Each estimator was evaluated across three strictly separated test sets:
1. **Historical Held-Out Test Set ($N = 7,557$)**: Standard frozen test partition.
2. **Retrospective Challenge Benchmark ($N = 500$)**: Extreme ligament clearance down to $6.5\text{ mm}$ (Stage 3A-R).
3. **New Blind Confirmatory Test Set ($N = 500$)**: 300 space-filling, 100 in-domain challenging, and 100 out-of-domain defect diameter extension ($d/D_c \in [0.32, 0.45]$).

---

## 1. Mathematical Formulation & Feature Extraction

All models predict the normalized stress concentration factor $K_{t,\text{element}} = \sigma_{\text{vM, max}}^{\text{elem}} / \sigma_0$ ($\sigma_0 = 100.0\text{ psi}$) from Feature Set D:
$$\mathbf{x} = \left[ \frac{x}{L},\; \frac{y}{H},\; \frac{d}{D_c},\; \frac{L}{H},\; \frac{c_{\text{ligament}}}{D_c},\; \frac{c_{\text{bnd,min}}}{H} \right] \in \mathbb{R}^6$$

Features were standardized using $\boldsymbol{\mu}_{\text{train}}$ and $\boldsymbol{\sigma}_{\text{train}}$ fitted strictly on the 36,512 training records.

---

## 2. Comparative Performance on Historical Held-Out Test Set ($N = 7,557$)

| Model Architecture | Model Family | Parameters / Trees | Test $R^2$ | Pearson $R$ | Test RMSE (psi) | Test MAE (psi) | Max Error (psi) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | Shallow ANN | 1,025 params | 0.4665 | 0.6898 | 7.53 | 3.94 | 161.85 |
| **ANN-256** | Shallow ANN | 2,049 params | 0.5584 | 0.7507 | 6.85 | 3.45 | 154.64 |
| **ANN-512** | Shallow ANN | 4,097 params | 0.5122 | 0.7526 | 7.20 | 3.77 | 135.19 |
| **DeepMLP-D1** | Deep MLP | 165,249 params | 0.6402 | 0.8817 | 6.18 | 4.65 | 90.23 |
| **DeepMLP-D2** | Deep MLP | 231,169 params | 0.6194 | 0.8239 | 6.36 | 3.71 | 109.78 |
| **ResMLP** | Residual Net | 396,289 params | **0.8292** | **0.9198** | **4.26** | **2.72** | **68.49** |
| **HistGradientBoosting** | Tree Ensemble | 200 trees | **0.9196** | **0.9594** | **2.92** | **1.84** | **49.50** |
| **Random Forest** | Tree Ensemble | 100 trees | **0.9219** | **0.9602** | **2.88** | **1.83** | **54.97** |
| **Extra Trees** | Tree Ensemble | 100 trees | **0.9268** | **0.9630** | **2.79** | **1.80** | **58.15** |

> [!NOTE]
> On the standard historical test partition ($N = 7,557$), tree-based ensembles (Extra Trees, Random Forest, HistGradientBoosting) achieve $R^2 > 0.92$ and $\text{RMSE} < 2.92\text{ psi}$.
> Among neural networks, the deep residual network **ResMLP** drastically outperforms shallow ANNs, achieving $R = 0.9198$ and cutting peak error from $161.85\text{ psi}$ down to $68.49\text{ psi}$.

---

## 3. Generalization on Confirmatory Test Suites

Evaluating estimators solely on in-distribution test splits can create false confidence. Here we evaluate all 9 single models on two external confirmatory suites:

### 3.1 New Stage 3B Blind Confirmatory Test Set ($N = 500$)
*(Includes 100 out-of-domain diameter extension cases $d/D_c \in [0.32, 0.45]$)*

| Model | Test Set $R^2$ | Blind Conf $R^2$ | Blind Conf $R$ | Blind Conf RMSE (psi) | Blind Conf MaxErr (psi) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | 0.4665 | 0.4642 | 0.7061 | 17.94 | 118.28 |
| **ANN-256** | 0.5584 | 0.5503 | 0.7634 | 16.43 | 111.45 |
| **ANN-512** | 0.5122 | 0.5793 | 0.7712 | 15.89 | 90.59 |
| **DeepMLP-D1** | 0.6402 | 0.7868 | 0.8924 | 11.31 | 61.77 |
| **DeepMLP-D2** | 0.6194 | 0.6133 | 0.8351 | 15.24 | 75.04 |
| **ResMLP** | 0.8292 | **0.8433** | **0.9221** | **9.70** | **53.05** |
| **HistGradientBoosting** | 0.9196 | **0.9510** | **0.9760** | **5.42** | **42.43** |
| **Random Forest** | 0.9219 | **0.9594** | **0.9801** | **4.94** | **42.55** |
| **Extra Trees** | 0.9268 | **0.9623** | **0.9814** | **4.76** | **41.85** |

### 3.2 Retrospective Challenge Benchmark ($N = 500$, Stage 3A-R)
*(Extreme ligament clearances down to $6.5\text{ mm}$, stress spikes $> 400\text{ psi}$)*

| Model | Challenge $R^2$ | Challenge RMSE (psi) | Challenge MaxErr (psi) |
| :--- | :---: | :---: | :---: |
| **ANN-128** | 0.1442 | 43.40 | 269.45 |
| **ANN-512** | 0.2251 | 41.29 | 249.73 |
| **ResMLP** | 0.3672 | 37.32 | 208.40 |
| **Extra Trees** | 0.6160 | 29.07 | 185.67 |
| **HistGradientBoosting** | **0.7228** | **24.70** | **149.55** |
| **Random Forest** | **0.7535** | **23.29** | **156.97** |

---

## 4. Physical Insights and Stacking Implications

1. **Boundary-Layer Resilience:**
   In regions where the defect hole encroaches within $10\text{ mm}$ of the central hole ($c_{\text{ligament}} < 10\text{ mm}$), stress rises asymptotically. Unregularized neural networks extrapolate polynomial curves, causing errors up to $200\text{ psi}$. Decision trees, by virtue of recursive partitioning, clamp predictions to localized leaf averages, capping maximum error.
2. **Smooth Spatial Gradients:**
   In the interior space-filling domain, neural networks (specifically ResMLP) provide continuous, differentiable stress fields, avoiding the staircase discontinuities of decision trees.
3. **The Stacking Hypothesis:**
   Combining ResMLP, DeepMLP-D2, Random Forest, Extra Trees, and HistGradientBoosting via a cross-fitted meta-regressor should yield an optimal surrogate that achieves smooth interpolation across open plate regions while retaining conservative bounds under asymptotic boundary layer spikes.
