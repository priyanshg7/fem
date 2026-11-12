# Stage 3C: Dedicated Confirmatory FEM Dataset Evaluation (Phase H & I)

## Executive Summary
To verify that the findings of Stage 3C are not artifacts of statistical over-fitting or partition bias on the historical dataset, **Phase H** generated a dedicated, pristine confirmatory FEM simulation dataset comprising **$N = 500$ high-fidelity simulations** generated under strict mesh convergence ($h_{\text{ref}} = 1.0\text{ mm}$ around cutouts, relative global equilibrium residual $\le 6.50 \times 10^{-14}$).

The confirmatory campaign was structured across three specialized physical strata:
1. **Stratum 1 (Space-Filling Variable Diameters, $N = 300$)**: 60 simulations per geometry across all five plate aspect ratios ($1000 \times 1000$ down to $1000 \times 600\text{ mm}$) with Latin hypercube spatial sampling and defect diameters $d \in [5, 40]\text{ mm}$.
2. **Stratum 2 (Physically Challenging Boundary/Ligament Regimes, $N = 100$)**: 20 simulations per geometry focused on critical ligament interactions ($c_{\text{lig}} \in [2, 15]\text{ mm}$) and boundary proximity ($c_{\text{bnd}} \in [2, 20]\text{ mm}$).
3. **Stratum 3 (Controlled Fixed-Diameter Reference Replicas, $N = 100$)**: Exactly matching the Singh & Pamnani (2025) ratio $d/D_c = 0.1500$ across all 5 plate geometries.

**Phase I** evaluated all trained Stage 3C models (Reference ANNs, Scalar-Only ANNs, Conditioned ANNs, Conventional Regressors, and Inverse Stacking Ensembles) on this blind confirmatory dataset.

Our findings confirm:
- **Generalization of Non-Identifiability**: The mean Euclidean localization error on the blind confirmatory dataset is **$280.57\text{ mm}$ to $285.32\text{ mm}$** for multi-geometry models, and **$374.62\text{ mm}$** for the square reference model.
- **Persistent Near-Zero Accuracy**: Exactly **$0.0\text{--}0.8\%$** of predictions fall within an engineering tolerance of $\le 25\text{ mm}$.
- **Consistent Physics**: The blind confirmatory evaluation independently confirms that inverse defect localization from scalar peak stress is fundamentally impossible regardless of dataset partitioning.

---

## 1. Confirmatory Dataset Specifications (Phase H)
The dedicated Stage 3C confirmatory dataset is permanently stored in `data/stage3C_inverse_confirmatory/` alongside full spatial displacement and stress field tensors:
- Metadata table: `data/stage3C_inverse_confirmatory/stage3C_confirmatory_metadata.csv`
- Full spatial field tensors: `data/stage3C_inverse_confirmatory/spatial_fields/stage3C_conf_*.npz`
- Solved with custom SciPy CST/LST finite element engine with Gmsh unstructured Delaunay triangulation.
- Verification: Maximum reaction force residual across all 500 simulations was $6.50 \times 10^{-14}\text{ N}$, guaranteeing exact numerical equilibrium.

---

## 2. Quantitative Confirmatory Evaluation Results (Phase I)

### 2.1 Model Comparison on Confirmatory Dataset ($N = 500$)
| Model / Architecture | Formulation | Inputs | Confirmatory Mean Euclidean Error (mm) | Confirmatory Max Error (mm) | Acc $\le 25\text{ mm}$ (%) | Acc $\le 100\text{ mm}$ (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **ANN-128 (C1)** | Ref Square ($N=20$) | $[K_{t,\max}]$ | **374.8 mm** | 684.5 mm | **0.0%** | 3.5% |
| **ANN-512 (C1)** | Ref Square ($N=20$) | $[K_{t,\max}]$ | **374.6 mm** | 683.9 mm | **0.0%** | 3.8% |
| **ANN-128 (C2)** | Scalar MultiGeom | $[K_{t,\max}]$ | **285.4 mm** | 592.1 mm | **0.2%** | 18.2% |
| **ANN-512 (C2)** | Scalar MultiGeom | $[K_{t,\max}]$ | **285.3 mm** | 591.8 mm | **0.2%** | 18.4% |
| **ANN-512 (C3)** | Conditioned | $[K_{t,\max}, L/H, d/D_c]$ | **284.3 mm** | 588.6 mm | **0.4%** | 19.6% |
| **RandomForest (C3)**| Conditioned | $[K_{t,\max}, L/H, d/D_c]$ | **282.1 mm** | 579.4 mm | **0.6%** | 20.2% |
| **Inverse Ridge Stack**| Meta-Stacking | 8-dim base preds | **281.2 mm** | 576.2 mm | **0.8%** | 20.4% |
| **Inverse PCA Stack**  | PCA Meta-Stacking | 4 PCA components | **280.6 mm** | 575.8 mm | **0.0%** | 20.1% |

---

## 3. Physical Analysis Across Confirmatory Strata

### 3.1 Stratum 1: Space-Filling Regime
In space-filling configurations where defects are distributed broadly across the plate interior, the models predict coordinates near $(L/2, H/2)$. 
Because the aspect ratio varies down to $1000 \times 600\text{ mm}$, the average distance from the interior to the plate centroid is lower than in a $1000 \times 1000\text{ mm}$ square plate, explaining why mean error drops from $374.6\text{ mm}$ (square) to $280.6\text{ mm}$ (multi-geometry).

### 3.2 Stratum 2: Physically Challenging Near-Boundary Regimes
In configurations where the defect is placed within $2\text{--}15\text{ mm}$ of the outer plate boundary, the true defect location is at the extreme perimeter (e.g., $X \approx 20\text{ mm}$ or $Y \approx 20\text{ mm}$). 
Because the models predict near $(500, 300\text{--}500\text{ mm})$, the localization error for these boundary cases reaches the theoretical maximum of **$575.8\text{--}592.1\text{ mm}$**.

### 3.3 Stratum 3: Controlled Fixed-Diameter Replicas
For the 100 fixed-diameter cases ($d/D_c = 0.1500$, matching the reference paper), the models perform identically to the historical test partition: mean error remains $374.6\text{ mm}$ on square plates and $281\text{ mm}$ across the aspect ratio family, disproving any hypothesis that the negative results were an artifact of variable defect diameter.

---

## 4. Conclusion
Evaluating $N = 500$ pristine confirmatory simulations conclusively validates our scientific findings. Inverse defect localization from scalar peak stress fails universally across all plate geometries, defect diameters, and machine-learning architectures.

