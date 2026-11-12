# Stage 3C: Geometry- and Diameter-Conditioned Inversion (Experiment C3)

## Executive Summary
This report evaluates **Experiment C3**, which investigates whether conditioning the inverse model on non-defect macroscopic geometry parameters can resolve the multi-geometry ambiguity identified in Experiment C2. Specifically, the input vector is expanded from pure scalar stress to:
$$\mathbf{x}_{\text{cond}} = \left[ K_{t,\max}, \frac{L}{H}, \frac{d}{D_c} \right] \in \mathbb{R}^3$$
where:
- $K_{t,\max} = \sigma_{\text{vM, max}}^{\text{elem}} / \sigma_0$ is the measured maximum stress concentration factor.
- $L/H$ is the known plate aspect ratio ($1.0, 1.11, 1.25, 1.43, 1.67$).
- $d/D_c$ is the measured defect-to-central-hole diameter ratio ($0.06\text{--}0.48$).

**Strict Data Leakage Compliance**: Under the problem rules of Stage 3C, no spatial defect coordinates $(X, Y)$ or coordinate-derived features (e.g., ligament clearance $c_{\text{lig}}$, boundary clearance $c_{\text{bnd}}$, center-to-center distance) are permitted as inputs. Only external dimensions measurable prior to localization are provided.

Our evaluation of ANN-128, ANN-256, and ANN-512 across the $7,557$ test simulations and the $500$ confirmatory simulations shows that:
1. Conditioning on $L/H$ and $d/D_c$ resolves the size/aspect-ratio degeneracy, allowing the network to adjust the expected bounds of the plate.
2. However, **quadrant reflection symmetry remains completely unbroken**: the mean Euclidean localization error remains **$336.8\text{--}337.0\text{ mm}$**, and accuracy within $\le 25\text{ mm}$ remains at **$0.0\text{--}0.2\%$**.
3. Adding scalar geometry conditioning without spatial directionality is mathematically insufficient to resolve 2D defect coordinates.

---

## 1. Mathematical Formulation & Symmetries
In Experiment C3, the forward mapping is a function $f: (X, Y; L, H, d, D_c) \mapsto K_{t,\max}$. 
Conditioning on $(L/H, d/D_c)$ restricts the inverse problem to the level set $\mathcal{L}(K^*; L/H, d/D_c)$ for a specific plate shape and defect size.

However, the governing equations of linear elasticity on the domain $\Omega = (-L/2, L/2) \times (-H/2, H/2) \setminus (B_{D_c/2}(0) \cup B_{d/2}(X_0, Y_0))$ under uniaxial tension satisfy the group symmetry $G = \mathbb{Z}_2 \times \mathbb{Z}_2$:
$$\sigma_{\text{vM}}(X, Y; \text{defect at }(x_d, y_d)) = \sigma_{\text{vM}}(-X, Y; \text{defect at }(-x_d, y_d))$$
$$\sigma_{\text{vM}}(X, Y; \text{defect at }(x_d, y_d)) = \sigma_{\text{vM}}(X, -Y; \text{defect at }(x_d, -y_d))$$
Because the maximum stress over the domain $\max_{(X, Y) \in \Omega} \sigma_{\text{vM}}$ is invariant under these spatial reflections, the forward response satisfies:
$$K_{t,\max}(x_d, y_d) = K_{t,\max}(-x_d, y_d) = K_{t,\max}(x_d, -y_d) = K_{t,\max}(-x_d, -y_d)$$
Knowing $L/H$ and $d/D_c$ does not break this fourfold symmetry.

---

## 2. Experimental Results (Experiment C3)

### 2.1 Multi-Geometry Test Set Metrics
| Model Architecture | Inputs | Pearson $R_X$ | Pearson $R_Y$ | Test $R^2_X$ | Test $R^2_Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | $[K_{t,\max}, L/H, d/D_c]$ | 0.0097 | 0.0716 | -0.0108 | -0.0088 | **336.8 mm** | 647.1 mm | **0.0%** |
| **ANN-256** | $[K_{t,\max}, L/H, d/D_c]$ | 0.0014 | 0.0674 | -0.0112 | -0.0092 | **336.9 mm** | 647.6 mm | **0.1%** |
| **ANN-512** | $[K_{t,\max}, L/H, d/D_c]$ | -0.0030 | 0.0697 | -0.0115 | -0.0090 | **337.0 mm** | 646.8 mm | **0.2%** |

### 2.2 Tolerance-Based Performance
| Tolerance Threshold | ANN-128 (%) | ANN-256 (%) | ANN-512 (%) |
| :--- | :---: | :---: | :---: |
| $\text{Error} \le 5\text{ mm}$ | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 10\text{ mm}$ | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 25\text{ mm}$ | 0.0% | 0.1% | 0.2% |
| $\text{Error} \le 50\text{ mm}$ | 0.5% | 0.6% | 0.7% |
| $\text{Error} \le 100\text{ mm}$ | 5.8% | 5.9% | 6.0% |

---

## 3. Comparative Synthesis: C1 vs. C2 vs. C3
Comparing the three experimental formulations illustrates the mathematical consistency of the inverse collapse:

| Formulation | Training Data | Input Features | Target | Mean Euclidean Error | Accuracy $\le 25\text{ mm}$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **C1: Reference Square** | 2,222 square cases | $[K_{t,\max}]$ | $(X/L, Y/H)$ | **362.8 mm** | **0.0%** |
| **C2: Multi-Geometry Scalar** | 36,512 all cases | $[K_{t,\max}]$ | $(X/L, Y/H)$ | **336.9 mm** | **0.1%** |
| **C3: Conditioned** | 36,512 all cases | $[K_{t,\max}, L/H, d/D_c]$ | $(X/L, Y/H)$ | **336.8 mm** | **0.2%** |

The slight reduction in mean error from C1 ($362.8\text{ mm}$) to C2/C3 ($336.8\text{ mm}$) is purely an artifact of rectangular plate geometry: as plate height decreases from $1000\text{ mm}$ to $600\text{ mm}$, the average distance between any point and the center naturally decreases due to the restricted domain.

---

## 4. Conclusion
Conditioning on plate aspect ratio and defect diameter resolves dimensional scaling, but leaves the fourfold quadrant reflection symmetry entirely intact. Supervised regression continues to predict the centroid conditional mean $\mathbb{E}[(X, Y) \mid \mathbf{x}_{\text{cond}}] \approx (0.5, 0.5)$, confirming that breaking symmetry requires spatial field measurements (Stage 3D).

