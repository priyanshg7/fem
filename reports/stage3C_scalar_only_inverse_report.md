# Stage 3C: Extended Scalar-Only Inversion Across Multi-Geometry Dataset (Experiment C2)

## Executive Summary
This report presents the scientific evaluation of **Experiment C2**, which extends the scalar-only inverse defect localization formulation from a single square plate to the entire multi-geometry database consisting of 5 plate aspect ratios ($1000 \times 1000$, $1000 \times 900$, $1000 \times 800$, $1000 \times 700$, and $1000 \times 600\text{ mm}$) and continuously varying defect hole diameters ($d \in [5, 40]\text{ mm}$, $d/D_c \in [0.06, 0.48]$).

A common assumption in data-driven surrogate modeling is that expanding the training dataset size and introducing multi-geometry diversity might allow machine-learning algorithms to learn latent inverse representations. We test this hypothesis by training ANN-128, ANN-256, and ANN-512 regressors on $36,512$ FEM training simulations using only the observed scalar maximum Von Mises stress ($K_{t,\max} = \sigma_{\text{vM, max}}^{\text{elem}} / \sigma_0$).

Our experimental findings definitively disprove this hypothesis:
1. **Persistent Physical Failure**: The mean Euclidean localization error across the $7,557$ test simulations is **$336.9\text{ mm}$** (comparable to $33\text{--}42\%$ of the plate dimensions).
2. **Zero Engineering Utility**: Exactly **$0.0\text{--}0.1\%$** of predictions fall within an engineering tolerance of $\le 25\text{ mm}$.
3. **Compound Degeneracy**: In addition to fourfold quadrant reflection symmetry, varying defect diameter and plate aspect ratio introduces **manifold overlapping**: a small defect close to the central hole produces the exact same peak stress as a large defect placed further away or in a plate of different aspect ratio.

---

## 1. Formulation & Dataset Description
- **Input Feature**: Pure scalar stress concentration factor $K_{t,\max} = \sigma_{\text{vM, max}}^{\text{elem}} / 100\text{ psi} \in \mathbb{R}^1$.
- **Output Targets**: Dimensionless defect coordinates $(X/L, Y/H) \in [0, 1] \times [0, 1]$.
- **Data Partitions**:
  - Training Set: $36,512$ simulations (70%).
  - Validation Set: $7,556$ simulations (15%).
  - Historical Test Set: $7,557$ simulations (15%).
  - Dedicated Stage 3C Confirmatory Set: $500$ simulations.
- **Architectures**: Single hidden layer ANNs (128, 256, 512 neurons), trained with Adam optimizer, batch size 512, CosineAnnealing learning rate schedule.

---

## 2. Experimental Results (Experiment C2)

### 2.1 Multi-Geometry Test Set Metrics
| Model Architecture | Pearson $R_X$ | Pearson $R_Y$ | Test $R^2_X$ | Test $R^2_Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | 0.0251 | 0.0609 | -0.0102 | -0.0094 | **336.9 mm** | 647.5 mm | **0.0%** |
| **ANN-256** | 0.0257 | 0.0615 | -0.0098 | -0.0091 | **337.0 mm** | 648.1 mm | **0.0%** |
| **ANN-512** | 0.0285 | 0.0566 | -0.0089 | -0.0085 | **336.9 mm** | 646.9 mm | **0.1%** |

### 2.2 Tolerance-Based Accuracy Across Distance Thresholds
| Tolerance Threshold | ANN-128 (%) | ANN-256 (%) | ANN-512 (%) |
| :--- | :---: | :---: | :---: |
| $\text{Error} \le 5\text{ mm}$ | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 10\text{ mm}$ | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 25\text{ mm}$ | 0.0% | 0.0% | 0.1% |
| $\text{Error} \le 50\text{ mm}$ | 0.4% | 0.4% | 0.5% |
| $\text{Error} \le 100\text{ mm}$ | 5.2% | 5.3% | 5.4% |

Over $94.6\%$ of all predictions incur errors exceeding $100\text{ mm}$.

---

## 3. Physical Analysis: The Two Levels of Degeneracy

### 3.1 Quadrant Symmetry (Level 1 Degeneracy)
In any plate geometry symmetric about the $X = L/2$ and $Y = H/2$ axes, a single scalar stress measurement cannot differentiate between the four quadrants:
$$(X, Y) \sim (L - X, Y) \sim (X, H - Y) \sim (L - X, H - Y)$$
Under MSE loss, the optimal Bayes estimator outputs the conditional mean $\mathbb{E}[(X, Y) \mid K_{t,\max}]$. Because the distribution of defect locations is symmetric about the center, this conditional mean is pinned near $(0.5, 0.5)$.

### 3.2 Diameter and Aspect Ratio Confounding (Level 2 Degeneracy)
In Experiment C1, defect diameter and plate geometry were fixed, meaning the pre-image was a 1D curve. In Experiment C2, where plate height $H \in [600, 1000]\text{ mm}$ and defect diameter $d \in [5, 40]\text{ mm}$ vary, the pre-image of a single scalar stress $K_{t,\max}$ expands from a 1D curve into a **3D volume in parameter space**:
$$\mathcal{S}(K^*) = \{(X, Y, d, H) \mid K_{t,\max}(X, Y, d, H) = K^*\}$$
For instance:
- A defect with $d = 35\text{ mm}$ placed far from the hole ($c_{\text{lig}} = 200\text{ mm}$) in a $1000 \times 600\text{ mm}$ plate produces $K_{t,\max} \approx 3.45$.
- A defect with $d = 10\text{ mm}$ placed moderately close to the hole ($c_{\text{lig}} = 35\text{ mm}$) in a $1000 \times 1000\text{ mm}$ plate produces identical $K_{t,\max} \approx 3.45$.
- Without knowing the defect size or plate aspect ratio, the inverse mapping suffers severe multi-valued ambiguity.

---

## 4. Conclusion
Scaling the dataset from $10,075$ to $51,625$ simulations and training larger neural networks cannot compensate for missing physical information. Scalar maximum Von Mises stress fundamentally lacks the degrees of freedom necessary to resolve 2D spatial defect coordinates.

