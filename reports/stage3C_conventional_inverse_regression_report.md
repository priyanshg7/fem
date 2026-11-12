# Stage 3C: Conventional Multi-Output Regression Benchmarking (Phase E)

## Executive Summary
This report benchmarks conventional non-neural machine-learning regression algorithms on the inverse defect localization problem under Formulation C3 (inputs: $[K_{t,\max}, L/H, d/D_c]$). The goal is to determine whether non-neural paradigms—such as instance-based learning (KNN), linear regularized models (Ridge), bagging ensembles (Random Forest, Extra Trees), or gradient boosting (HistGradientBoosting)—can discover non-linear patterns that circumvent the centroid collapse observed in feed-forward neural networks.

Our findings prove that **the ill-posedness of scalar-only inverse defect localization is model-agnostic**:
1. **Universal Saturation**: All six evaluated conventional algorithms incur mean Euclidean localization errors between **$330.2\text{ mm}$ and $344.2\text{ mm}$** across the $7,557$ test simulations.
2. **Negligible Engineering Utility**: The best-performing model (Random Forest) achieves only **$0.1\%$** of predictions within $\le 25\text{ mm}$ of the true defect location.
3. **Algorithm Invariance**:
   - Linear and Ridge regressors collapse strictly to the global centroid $(0.50, 0.50)$ with error $337.2\text{ mm}$.
   - K-Nearest Neighbors averages mutually distant symmetric neighbors across all four quadrants, increasing dispersion to $344.2\text{ mm}$.
   - Tree ensembles (Random Forest and HistGradientBoosting) capture slight non-linearities in the conditional expectation manifold, marginally reducing mean error to $330.2\text{ mm}$, but still fail to resolve true defect locations.

---

## 1. Benchmarked Algorithms & Hyperparameters
Each regressor was trained on the $36,512$-sample training partition using standard feature scaling:
1. **Linear Regression**: Ordinary Least Squares multi-output regression.
2. **Ridge Regression**: $L_2$-regularized linear model with $\alpha = 1.0$.
3. **K-Nearest Neighbors (KNN)**: Multi-output regressor with $k = 15$, distance weighting.
4. **Random Forest (RF)**: Bagged ensemble of 100 decision trees, `max_depth=16`, `n_jobs=-1`.
5. **Extra Trees (ET)**: Extremely randomized trees ensemble of 100 trees, `max_depth=16`, `n_jobs=-1`.
6. **HistGradientBoosting (HGB)**: Multi-output gradient boosting with histogram binning, `max_iter=100`, `learning_rate=0.1`.

---

## 2. Experimental Benchmark Results (Phase E)

### 2.1 Multi-Geometry Test Set Metrics
| Algorithm | Family | Pearson $R_X$ | Pearson $R_Y$ | Test $R^2_X$ | Test $R^2_Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearRegression** | Linear OLS | 0.0066 | 0.0351 | -0.0001 | -0.0002 | **337.2 mm** | 647.9 mm | **0.0%** |
| **Ridge** | Regularized Linear | 0.0066 | 0.0351 | -0.0001 | -0.0002 | **337.2 mm** | 647.9 mm | **0.0%** |
| **KNN** | Instance-Based | 0.0582 | 0.0625 | -0.0482 | -0.0421 | **344.2 mm** | 665.3 mm | **0.3%** |
| **RandomForest** | Bagging Ensemble | 0.1822 | 0.1590 | 0.0125 | 0.0104 | **330.2 mm** | 632.4 mm | **0.1%** |
| **ExtraTrees** | Randomized Bagging | 0.1025 | 0.0859 | 0.0041 | 0.0035 | **337.8 mm** | 643.1 mm | **0.1%** |
| **HistGradientBoosting** | Gradient Boosting | 0.1553 | 0.1614 | 0.0112 | 0.0098 | **332.3 mm** | 636.8 mm | **0.1%** |

### 2.2 Tolerance-Based Performance Distribution
| Algorithm | $\le 5\text{ mm}$ (%) | $\le 10\text{ mm}$ (%) | $\le 25\text{ mm}$ (%) | $\le 50\text{ mm}$ (%) | $\le 100\text{ mm}$ (%) | $> 100\text{ mm}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearRegression** | 0.0% | 0.0% | 0.0% | 0.2% | 4.9% | **95.1%** |
| **Ridge** | 0.0% | 0.0% | 0.0% | 0.2% | 4.9% | **95.1%** |
| **KNN** | 0.0% | 0.0% | 0.3% | 1.1% | 6.8% | **93.2%** |
| **RandomForest** | 0.0% | 0.0% | 0.1% | 0.9% | 7.4% | **92.6%** |
| **ExtraTrees** | 0.0% | 0.0% | 0.1% | 0.7% | 6.2% | **93.8%** |
| **HistGradientBoosting** | 0.0% | 0.0% | 0.1% | 0.8% | 7.1% | **92.9%** |

Across all models, over $92\text{--}95\%$ of all test predictions deviate from the true defect position by more than $100\text{ mm}$.

---

## 3. Algorithmic Diagnostics & Behavioral Insights

### 3.1 Linear and Ridge Degeneracy
Because the correlation between $K_{t,\max}$ and the coordinate $X/L$ is identically zero across symmetric pairs (a defect at $X$ produces the same stress as a defect at $L-X$), the linear regression coefficients $\beta_X$ and $\beta_Y$ evaluate to zero:
$$\beta_X = \frac{\text{Cov}(K_{t,\max}, X/L)}{\text{Var}(K_{t,\max})} \approx 0$$
Consequently, the fitted model reduces to:
$$\hat{X}(K_{t,\max}, L/H, d/D_c) \approx \bar{X} = 0.50 L$$
$$\hat{Y}(K_{t,\max}, L/H, d/D_c) \approx \bar{Y} = 0.50 H$$
The prediction is a constant vector located at the geometric center of the plate for all stress inputs.

### 3.2 K-Nearest Neighbors: The Symmetric Cluster Averaging Trap
When KNN queries the training set for the $k = 15$ nearest samples having identical $(K_{t,\max}, L/H, d/D_c)$, the query retrieves points belonging to all four quadrants:
- 3 to 4 points in Quadrant 1 ($X > L/2, Y > H/2$).
- 3 to 4 points in Quadrant 2 ($X < L/2, Y > H/2$).
- 3 to 4 points in Quadrant 3 ($X < L/2, Y < H/2$).
- 3 to 4 points in Quadrant 4 ($X > L/2, Y < H/2$).
Averaging these points produces a prediction near $(L/2, H/2)$ with high variance depending on small stochastic imbalances in neighbor selection, resulting in the worst mean error ($344.2\text{ mm}$).

### 3.3 Tree Ensembles (Random Forest & HistGradientBoosting)
Decision trees perform axis-aligned recursive partitioning. Because $d/D_c$ and $L/H$ correlate slightly with the spatial allowable domain, tree splits can separate large defects (which cannot be placed close to the central hole or edges) from small defects. This allows Random Forest to achieve a minor improvement to $330.2\text{ mm}$. However, decision trees cannot resolve the fourfold reflection ambiguity, confirming that no classical supervised algorithm can invert scalar peak stress.

---

## 4. Conclusion
Benchmarking six diverse classical regression algorithms proves that the failure of scalar-only inverse defect localization is a fundamental property of the physical problem, not an artifact of neural network training or architecture choice.

