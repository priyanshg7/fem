# Stage 3C: Inverse Stacking Ensemble Evaluation (Phase F)

## Executive Summary
In Stage 3B of this research project, stacked ensemble architectures achieved unprecedented accuracy in forward stress prediction, driving test $R^2$ from $0.7001$ to $0.9634$ and reducing RMSE to $4.69\text{ psi}$. A natural scientific question is: **Can stacked meta-ensembles similarly resolve the inverse defect localization problem?**

This report evaluates **Phase F: Inverse Stacking Ensembles**, testing two multi-model stacked architectures:
1. **Inverse Ridge Stack**: Multi-output Ridge meta-regression over out-of-fold (OOF) predictions from four diverse base models (KNN, Random Forest, Extra Trees, HistGradientBoosting).
2. **Inverse PCA Stack**: Principal Component Analysis meta-stacking, extracting the top 4 orthogonal latent prediction components followed by regularized Ridge regression.

To guarantee zero data leakage, all base-model predictions were generated using strict 5-fold cross-fitting on the $36,512$ training samples.

Our rigorous evaluation yields a definitive conclusion:
- **Stacking Fails on Inverse Localization**: The Inverse Ridge Stack achieves a test Pearson correlation of $R_X = 0.1991, R_Y = 0.1794$ and a mean Euclidean error of **$330.1\text{ mm}$**. This is virtually identical to standalone Random Forest ($330.2\text{ mm}$).
- **The Information-Theoretic Limit**: While stacking successfully eliminates individual model variance and bias in well-posed forward problems, **ensembles cannot synthesize information that is absent from the input domain**. Combining four degenerate predictions centered at $(0.5, 0.5)$ mathematically results in another degenerate prediction centered at $(0.5, 0.5)$.

---

## 1. Stacking Architecture & Leakage Prevention
The inverse stacking pipeline operates under strict 5-fold cross-fitting:
$$\mathcal{D}_{\text{train}} = \bigcup_{k=1}^5 \mathcal{D}_k$$
For each fold $k \in \{1, \dots, 5\}$:
1. Base models $\mathcal{M}_m$ ($m \in \{\text{KNN, RF, ET, HGB}\}$) are trained on $\mathcal{D}_{\setminus k}$.
2. Out-of-fold predictions $\hat{\mathbf{y}}_k^{(m)} = \mathcal{M}_m(\mathbf{x}_k)$ are generated on the held-out fold $\mathcal{D}_k$.
3. The concatenated level-1 design matrix has dimension $N \times 8$:
$$\mathbf{Z} = \left[ \hat{\mathbf{y}}_{\text{KNN}}, \hat{\mathbf{y}}_{\text{RF}}, \hat{\mathbf{y}}_{\text{ET}}, \hat{\mathbf{y}}_{\text{HGB}} \right] \in \mathbb{R}^{36,512 \times 8}$$
4. Level-2 meta-regressors are fitted on $(\mathbf{Z}, \mathbf{y}_{\text{true}})$.
5. For evaluation, base models trained on the full training set generate meta-features on the test ($N = 7,557$) and dedicated confirmatory ($N = 500$) datasets.

---

## 2. Experimental Stacking Results (Phase F)

### 2.1 Multi-Geometry Test Set Performance
| Architecture | Meta-Learner | Pearson $R_X$ | Pearson $R_Y$ | Test $R^2_X$ | Test $R^2_Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best Single Base (RF)** | None | 0.1822 | 0.1590 | 0.0125 | 0.0104 | **330.2 mm** | 632.4 mm | **0.1%** |
| **Inverse Ridge Stack** | MultiOutput Ridge | 0.1991 | 0.1794 | 0.0142 | 0.0118 | **330.1 mm** | 630.8 mm | **0.1%** |
| **Inverse PCA Stack** | PCA(4) + Ridge | 0.1970 | 0.1702 | 0.0138 | 0.0110 | **330.6 mm** | 631.9 mm | **0.0%** |

### 2.2 Dedicated Stage 3C Confirmatory Evaluation ($N = 500$)
| Architecture | Pearson $R_X$ | Pearson $R_Y$ | Mean Euclidean Error (mm) | Max Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inverse Ridge Stack** | 0.1845 | 0.1652 | **332.4 mm** | 638.2 mm | **0.2%** |
| **Inverse PCA Stack** | 0.1812 | 0.1598 | **333.1 mm** | 640.1 mm | **0.0%** |

Generalization to the blind 500-sample confirmatory dataset confirms that inverse stacking does not overfit, but remains constrained by the exact same physical lower bound ($~332\text{ mm}$).

---

## 3. Comparison with Reference Paper (Singh & Pamnani 2025)
Singh & Pamnani evaluated stacked ensemble models (SEM-1, SEM-2, SEM-3) combining ANNs, reporting:
- SEM-1: $R_X \approx 0.82\text{--}0.86$
- SEM-2: $R_X \approx 0.89\text{--}0.91$
- SEM-3: $R_X \approx 0.938$

Why do our leak-free stacked models achieve $R \approx 0.199$ while the paper claimed $R \approx 0.938$?
1. **Severe In-Fold Training Leakage in the Paper**: In Singh & Pamnani (2025), level-1 models were trained on the entire dataset and their in-sample predictions were fed directly into the level-2 model without out-of-fold cross-fitting. Overfitting on training noise created spurious correlations between base predictions and targets.
2. **Failure to Report Euclidean Distance**: The paper reported only Pearson $R$, which can be artificially inflated by extreme leverage points or in-sample data leakage.
3. **Physical Impossibility**: No mathematical or machine-learning operation can invert an underdetermined system $\mathbb{R}^1 \to \mathbb{R}^2$ having fourfold symmetry without auxiliary measurements.

---

## 4. Conclusion
Inverse stacking cannot resolve defect localization from scalar peak stress. Ensembling provides no escape from the fundamental mathematical ill-posedness of scalar inversion.

