# Stage 3C: Reference Inverse ANN Evaluation & Replication Audit (Experiment C1)

## Executive Summary
This report presents the rigorous independent replication and physical evaluation of the inverse defect localization methodology published by **Singh & Pamnani (2025)** ("Static Analysis of a Plate with Defect Using Machine Learning"). In their study, feed-forward Artificial Neural Networks (ANN-128, ANN-256, ANN-512) were trained to estimate the 2D defect coordinates $(X/L, Y/H)$ in a square isotropic plate from a single scalar observation of maximum Von Mises stress concentration ($K_{t,\max}$).

While Singh & Pamnani reported apparent convergence with Pearson correlation coefficients $R_X \approx 0.4786\text{--}0.4821$ and $R_Y \approx 0.4943\text{--}0.5014$, our independent FEM campaign and multi-seed replication demonstrate that **reporting Pearson correlation as localization performance is scientifically misleading**. In physical engineering units, the mean Euclidean localization error is **$362.8\text{ mm}$ to $363.4\text{ mm}$ on a $1000 \times 1000\text{ mm}$ plate** (over $36\%$ of the entire plate dimension), and **$0.0\%$ of predictions fall within acceptable engineering tolerances ($\le 25\text{ mm}$)**.

The models do not learn the true defect coordinates; rather, gradient descent under mean squared error (MSE) converges to the **conditional expectation** $\mathbb{E}[(X, Y) \mid K_{t,\max}]$, which collapses near the plate centroid $(0.5, 0.5)$ due to fourfold quadrant reflection symmetry.

---

## 1. Experimental Setup & Reference Benchmark Parameters
Following the exact dimensional and physical parameters of Singh & Pamnani (2025):
- **Plate Geometry**: Square plate ($1000 \times 1000\text{ mm}$, $L/H = 1.0$).
- **Central Hole**: Fixed circular cutout diameter $D_c = 83.33\text{ mm}$ ($a/L = 1/12$).
- **Defect Hole**: Circular hole with fixed diameter $d = 12.50\text{ mm}$ ($d/D_c = 0.1500$, matching the paper's $c/a = 0.150$).
- **Input Feature**: Pure scalar stress concentration $K_{t,\max} = \sigma_{\text{vM, max}}^{\text{elem}} / \sigma_0$ ($\sigma_0 = 100\text{ psi}$).
- **Output Targets**: Dimensionless defect centroid coordinates $(X/L, Y/H) \in [0, 1] \times [0, 1]$.
- **Architectures**: Single hidden layer feed-forward networks with 128, 256, and 512 neurons, ReLU activation, Sigmoid output layer.
- **Statistical Rigor**: 5 independent training runs per architecture with randomized weight initialization seeds ($42, 101, 202, 303, 404$).

---

## 2. Experimental Results (Experiment C1)

### 2.1 Independent Replication vs. Singh & Pamnani (2025)
The table below contrasts the metrics reported by Singh & Pamnani (2025) with our independent replication evaluated on the historical test partition ($N = 468$) and the blind confirmatory partition ($N = 20$).

| Model Architecture | Source | Pearson $R_X$ | Pearson $R_Y$ | Norm MSE $X$ | Norm MSE $Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN-128** | *Singh & Pamnani (2025)* | 0.4821 | 0.4943 | 0.0402 | 0.0421 | *Not Reported* | *Not Reported* | *Not Reported* |
| **ANN-128** | *Independent Re-eval* | 0.0027 | 0.1065 | 0.0747 | 0.0757 | **363.4 mm** | 682.1 mm | **0.0%** |
| **ANN-256** | *Singh & Pamnani (2025)* | 0.4786 | 0.4987 | 0.0398 | 0.0415 | *Not Reported* | *Not Reported* | *Not Reported* |
| **ANN-256** | *Independent Re-eval* | 0.0235 | 0.1067 | 0.0746 | 0.0757 | **363.0 mm** | 681.4 mm | **0.0%** |
| **ANN-512** | *Singh & Pamnani (2025)* | 0.4802 | 0.5014 | 0.0391 | 0.0412 | *Not Reported* | *Not Reported* | *Not Reported* |
| **ANN-512** | *Independent Re-eval* | 0.0638 | 0.1059 | 0.0744 | 0.0757 | **362.8 mm** | 680.9 mm | **0.0%** |

### 2.2 Tolerance-Based Engineering Accuracy
Evaluating defect localization across engineering tolerance thresholds reveals the total failure of scalar-only regression:

| Tolerance Threshold | ANN-128 (%) | ANN-256 (%) | ANN-512 (%) |
| :--- | :---: | :---: | :---: |
| $\text{Error} \le 5\text{ mm}$ ($\le 0.5\% L$) | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 10\text{ mm}$ ($\le 1.0\% L$) | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 25\text{ mm}$ ($\le 2.5\% L$) | 0.0% | 0.0% | 0.0% |
| $\text{Error} \le 50\text{ mm}$ ($\le 5.0\% L$) | 0.2% | 0.2% | 0.2% |
| $\text{Error} \le 100\text{ mm}$ ($\le 10.0\% L$) | 3.8% | 3.8% | 4.1% |

Over $95\%$ of all test samples have localization errors exceeding $100\text{ mm}$.

---

## 3. Physical & Mathematical Cause of the Error Saturation

### 3.1 Fourfold Bilateral Quadrant Symmetry
Under uniform tensile loading $\sigma_{\infty}$ applied along the vertical edges ($Y$-direction), a square isotropic plate possesses reflectional symmetry across the two orthogonal plate centerlines:
$$X \leftrightarrow L - X, \quad Y \leftrightarrow H - Y$$
Consequently, for any defect placed at $(X_0, Y_0)$, the three mirror configurations:
$$(L - X_0, Y_0), \quad (X_0, H - Y_0), \quad (L - X_0, H - Y_0)$$
produce **identical scalar maximum Von Mises stress** $K_{t,\max}$ to within numerical machine precision. 

When an algorithm is trained to predict $(X, Y)$ from a single scalar $K_{t,\max}$ that has at least 4 mutually distant pre-images, any deterministic regressor minimizing quadratic loss outputs the arithmetic mean of the symmetric candidates:
$$\hat{X} = \frac{X_0 + (L - X_0) + X_0 + (L - X_0)}{4} = \frac{L}{2} = 500\text{ mm}$$
$$\hat{Y} = \frac{Y_0 + Y_0 + (H - Y_0) + (H - Y_0)}{4} = \frac{H}{2} = 500\text{ mm}$$

### 3.2 The Centroid Collapse Theorem
The theoretical expected Euclidean distance from a random point $(X, Y)$ uniformly distributed over $[0, L] \times [0, H]$ to the plate center $(L/2, H/2)$ is given by:
$$\mathbb{E}[D_{\text{center}}] = \int_0^1 \int_0^1 \sqrt{(u - 0.5)^2 + (v - 0.5)^2} \, du \, dv \approx 0.3826 \times 1000\text{ mm} = 382.6\text{ mm}$$
The empirical mean Euclidean error of our trained ANNs ($362.8\text{--}363.4\text{ mm}$) closely matches this theoretical value (slightly lower only because holes cannot be placed inside the central exclusion zone).

### 3.3 Why Pearson $R \approx 0.5$ in the Paper Was Misleading
In statistical regression:
1. Pearson correlation coefficient $R$ measures linear covariance: $R = \frac{\text{Cov}(\hat{Y}, Y)}{\sigma_{\hat{Y}} \sigma_Y}$.
2. If $\hat{Y}$ is squashed into a very narrow range around $0.5$ but has a tiny residual covariance with $Y$, $R$ can remain moderately positive ($\approx 0.48\text{--}0.50$) even when the variance of the prediction $\sigma_{\hat{Y}}^2 \to 0$ and the physical spatial error is massive.
3. Singh & Pamnani did not compute physical Euclidean distance or report the spatial scatter of predictions. Had they plotted predicted $(X, Y)$ coordinates on the plate geometry, they would have observed that all predictions collapse into a tight cluster near $(0.5, 0.5)$.

---

## 4. Key Takeaways & Recommendations
1. **Scalar Stress Cannot Localize Defects**: Single-scalar maximum Von Mises stress lacks all directional and quadrant information. 
2. **Replication Confirms Physics**: Increasing hidden layer width from 128 to 512 neurons yields negligible improvement ($363.4\text{ mm} \to 362.8\text{ mm}$), confirming that model capacity is not the limiting factor.
3. **Necessity of Stage 3D**: To localize defects uniquely and break the fourfold symmetry degeneracy, full-field spatial stress distributions or multi-sensor regional measurements (Stage 3D) are mathematically indispensable.

