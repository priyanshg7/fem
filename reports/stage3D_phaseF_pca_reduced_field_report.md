# Stage 3D — Phase F: PCA-Reduced Full Stress Field Inversion Report

## 1. Executive Summary
Phase F evaluates global linear modal reduction of continuous $64 \times 64$ rasterized stress fields using Principal Component Analysis (PCA) fitted strictly on the training partition for components $k \in \{10, 20, 40, 80, 120\}$.

### Experimental Results Across Retained Modes:
| Mode Count $k$ | Cumulative Variance (%) | Test Mean Euc (mm) | Test Median Euc (mm) | Confirmatory Mean Euc (mm) |
| :---: | :---: | :---: | :---: | :---: |
| $k = 10$ | 74.33% | 300.39 | 291.17 | 335.65 |
| $k = 20$ | 75.61% | 295.34 | 282.35 | 329.49 |
| $k = 40$ | 76.82% | 294.84 | 280.81 | 325.64 |
| $k = 80$ | 78.49% | 294.84 | 285.29 | 319.74 |
| $k = 120$ | 79.83% | 296.64 | 288.71 | 318.99 |

## 2. Physical Diagnosis: The Linear Orthogonal Blurring Limit
Linear PCA modes fail to improve upon unguided guessing (~295 mm) despite explaining nearly 80% of total field variance.
The reason is structural:
- **Global Modes vs Local Perturbations:** The principal eigenvectors $\phi_1, \phi_2, \dots, \phi_6$ capture global variations: plate aspect ratio ($1000 \times 1000$ vs $1000 \times 600$), uniform tension gradient, and the symmetric Kirsch lobes around the fixed central hole ($D_c = 200\text{ mm}$).
- **Spatial Smearing:** Because PCA is a linear projection across all 4,096 spatial pixels, a small localized defect perturbation is smeared across hundreds of Fourier-like harmonic basis functions. The localized non-stationary phase information $(x, y)$ is lost.
- **Scientific Conclusion:** Linear dimensionality reduction (PCA, POD, Karhunen-Loève) is mathematically ill-suited for localized point defect inverse problems. Non-linear, spatially localized architectures (CNN, U-Net) are required.

Refer to [stage3D_fig10_pca_eigenmodes.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig10_pca_eigenmodes.png), [stage3D_fig11_pca_scree_plot.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig11_pca_scree_plot.png), and [stage3D_fig12_pca_error_vs_components.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig12_pca_error_vs_components.png).
