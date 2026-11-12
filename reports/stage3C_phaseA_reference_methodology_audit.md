# Stage 3C Phase A Audit Report: Reference Paper Inverse Methodology

## Executive Summary

This report establishes the baseline scientific audit of the reference research paper:
> **Singh, A., & Pamnani, R. (2025). "Static Analysis of a Plate with Defect Using Machine Learning."**

The audit is conducted directly from the original paper PDF (`media_1794446466119_9a1f9093.pdf`), focusing specifically on the **inverse machine learning problem** for defect localization:
$$\sigma_{vM,\max} \longrightarrow (X, Y)$$

---

## 1. Physical Setup and Experimental Parameters

The paper's experimental setup is detailed in Section 2.4 and Table 1 (Page 6):

| Physical Parameter | Symbol | Paper Specification | Normalized Dimension | Notes / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Plate Length** | $l$ | 1.0 (normalized) | 1.0 | Square plate geometry ($l = b$) |
| **Plate Width / Height** | $b$ | 1.0 (normalized) | 1.0 | Height in load direction ($y$) |
| **Central Hole Diameter** | $a$ | $1/12 \approx 0.08333$ | $0.08333$ | Centered at $(0.5, 0.5)$ |
| **Defect Hole Diameter** | $c$ | $1/80 = 0.01250$ | $0.01250$ | **Strictly CONSTANT / FIXED diameter** |
| **Diameter Ratio** | $c/a$ | $(1/80)/(1/12) = 0.150$ | $0.150$ | Constant ratio across all simulations |
| **Applied Traction** | $P$ | $100\text{ psi}$ | $1.0\text{ (scaled)}$ | Uniaxial tensile load in $y$-direction |
| **Elastic Modulus** | $E$ | $10^7\text{ psi}$ | — | Linear elastic isotropic |
| **Poisson's Ratio** | $\mu$ | $0.3$ | — | Standard structural steel/aluminum |
| **Total Simulation Runs** | $N$ | **10,075 runs** | — | Generated in Abaqus via Python script |

> [!IMPORTANT]
> **Audit Finding on Geometry and Defect Size:**
> 1. The reference paper evaluated **only a single square plate ($1 \times 1$)**. It did not investigate rectangular aspect ratios.
> 2. The defect diameter was **strictly fixed at $c = 1/80 = 0.0125$ ($c/a = 0.15$)**. The paper did NOT vary defect diameter.
> 3. Therefore, in the reference paper, the only degrees of freedom were the two spatial coordinates of the defect: $(X, Y) \in (0, 1) \times (0, 1)$.

---

## 2. Inverse Problem Definition in Reference Paper

The paper states (Section 2.5, Table 3, Page 6):
> *"Conversely, the inverse model performs an inverse task where the observed stress values act as input parameter to estimate the defect's probable location (X, Y). Inverse prediction is crucial for defect localization in real-world application such as non-destructive testing and structural health monitoring, where stress data is often easier to measure than internal defect."*

- **Input Variable:** $\mathbf{x} = [\sigma_{vM,\max}]$ (a single scalar: maximum Von Mises stress in psi or normalized stress concentration factor $K_t$).
- **Target Variables:** $\mathbf{y} = [X, Y]$ (normalized Cartesian spatial coordinates of the defect center).

---

## 3. Reference Paper Inverse Results

### 3.1 Standalone Inverse Artificial Neural Networks (ANN)
The paper trained feedforward neural networks with a single hidden layer of 128, 256, and 512 neurons (Section 3.2, Pages 10–14):

| Inverse Model | Hidden Neurons | $R_X$ (Pearson) | $R_Y$ (Pearson) | $\text{MSE}_X$ | $\text{MSE}_Y$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | 128 | 0.4786 | 0.4943 | 0.0421 | 0.0398 |
| **ANN-256** | 256 | 0.4804 | 0.4957 | 0.0417 | 0.0410 |
| **ANN-512** | 512 | 0.4821 | 0.5014 | 0.0408 | 0.0392 |

#### Authors' Self-Reported Limitations on Standalone ANNs:
On Page 11, the authors explicitly state:
> *"The resulting correlation coefficients (R-values) were found to be 0.4786, 0.4804, and 0.4821 for X and 0.494, 0.495, 0.501 for Y... These values are significantly low, indicating a weak correlation between the predicted and actual inputs. Furthermore, the regression plots for these models exhibit a high degree of scatter, reflecting poor mapping capability and inconsistency in prediction. This suggests that the inverse ANN model in its current configuration is not able to effectively learn the inverse relationship between stress and geometric parameters..."*

### 3.2 Stacked Ensemble Inverse Models in Reference Paper
To improve upon the poor ANN results, the authors developed three stacked ensemble models (SEM-1, SEM-2, SEM-3):

| Inverse Stacked Model | Architecture Description | $R_X$ | $R_Y$ | $\text{MSE}_X$ | $\text{MSE}_Y$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SEM-1 (DNN-Stacked)** | Multi-ANN Level 0 $\to$ Meta Regressor | 0.8203 | 0.8213 | 0.0182 | 0.0171 |
| **SEM-2 (PCA-Stacked)** | PCA projection of base predictions $\to$ Meta | 0.9173 | 0.9141 | 0.0089 | 0.0091 |
| **SEM-3 (Meta-Stacked)**| Polynomial features + bagging/boosting $\to$ Multi-meta | **0.9389** | **0.9368** | **0.0067** | **0.0066** |

The authors report that SEM-3 achieves $R \approx 0.938$ and $\text{MSE} \approx 0.0067$ for both $X$ and $Y$ coordinates.

---

## 4. Critical Scientific Audit & Methodological Appraisal

### 4.1 The Fundamental Identifiability Paradox
A central question that the paper does not rigorously explore mathematically is **scalar non-identifiability**:
1. **Dimension Mismatch:**
   Mapping a 1D scalar $\sigma_{vM,\max} \in \mathbb{R}^1$ to a 2D position $(X, Y) \in \mathbb{R}^2$ is inherently underdetermined. The pre-image of a scalar value is a 1-dimensional level set (contour curve), not an isolated point!
2. **Geometric Quadrant Symmetry:**
   Under uniaxial tension along $y$, an isotropic plate containing a centered circular hole has fourfold symmetry. Four distinct coordinates:
   $$(X, Y), \quad (1-X, Y), \quad (X, 1-Y), \quad (1-X, 1-Y)$$
   produce virtually indistinguishable maximum Von Mises stresses.
3. **Implication for Supervised Training:**
   Any deterministic regression model minimizing mean squared coordinate error ($\text{MSE}$) under multi-modal data is mathematically proven to output the **conditional expectation**:
   $$\hat{\mathbf{y}}(\sigma) = \mathbb{E}[(X, Y) \mid \sigma_{\max} = \sigma]$$
   Due to quadrant symmetry, the conditional mean collapses toward $(0.5, 0.5)$! This explains why standalone ANNs achieved $R \approx 0.48 - 0.50$.
4. **How did SEM-3 achieve $R \approx 0.938$?**
   The paper notes that SEM-3 employed high-order polynomial expansion on the scalar stress input coupled with complex non-linear boosting/bagging meta-learners. In Stage 3C, we must examine whether SEM-3 learned an artificial partition or if it genuinely localized defects, and test its performance when tested on independent, non-leaked out-of-fold and blind validation splits.

---

## 5. Blueprint for Stage 3C Formulations

To provide a rigorous scientific investigation, Stage 3C will split the inverse problem into three clear, decoupled experiments:

1. **Experiment C1 (Reference-Faithful):**
   - Single square plate ($1000 \times 1000$ mm, $L/H = 1.00$).
   - Fixed defect diameter matching the paper ratio ($c/a = 0.15$, $d = 12.5$ mm).
   - Input: $\mathbf{x} = [K_{t,\max}]$.
   - Targets: $\mathbf{y} = [X/L, Y/H]$.
2. **Experiment C2 (Extended Scalar-Only Multi-Geometry):**
   - Five plate geometries ($L/H \in [1.00, 1.67]$) with continuously varying defect diameter ($d/D_c \in [0.075, 0.300]$).
   - Input: $\mathbf{x} = [K_{t,\max}]$.
   - Targets: $\mathbf{y} = [X/L, Y/H]$.
3. **Experiment C3 (Geometry- and Diameter-Conditioned):**
   - Multi-geometry dataset with known plate aspect ratio and known defect size.
   - Input: $\mathbf{x} = [K_{t,\max}, L/H, D/D_c]$.
   - Targets: $\mathbf{y} = [X/L, Y/H]$.

This guarantees both an exact reproduction of the published paper and a realistic assessment of scalar stress inversion.

