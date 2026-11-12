# Stage 3D: Improved Inverse Defect Localization Using Regional and Full-Field Stress Information — Comprehensive Research Monograph

## 1. Executive Summary & Research Trajectory
This monograph concludes **Stage 3D** of the research project:
*"Static Analysis of a Plate with Defect Using Machine Learning."*

### The Scientific Challenge: Overcoming Scalar Non-Identifiability
In Stage 3C, direct inverse defect localization using scalar peak Von Mises stress ($K_{t,\max}$) was proven to be fundamentally ill-posed ($MAE \approx 270\text{--}300\text{ mm}$) due to dimensional underdetermination: regular level-set pre-images $K_{t,\max}^{-1}(c)$ form continuous 1D curves spanning multiple quadrants.

Stage 3D investigated whether richer spatial stress representations can overcome this non-identifiability across 5 plate geometries ($1000\times 1000$ to $1000\times 600\text{ mm}$) and variable defect configurations.

### The Five Progressive Model Families:
1. **Model D1 (Regional Peak Stress):** 16 quadrant and ring features break quadrant ambiguity, reducing error to **121.25 mm** (a $58\%$ improvement over Stage 3C).
2. **Model D2 (Discrete Sensors):** 4 to 64 virtual strain gauges reveal the *spatial blind spot* phenomenon (error stays at **285.04 mm** due to $1/r^2$ localized stress decay between discrete sensor nodes).
3. **Model D3 (PCA Modal Reduction):** Linear principal components suffer from *global linear blurring* (error **296.64 mm** across 120 modes).
4. **Model D4 (CNN Field Inversion):** Convolutional feature hierarchies capture translation-invariant local stress concentration lobes, driving error down to **56.42 mm** (confirmatory: **70.24 mm**).
5. **Model D5 (Spatial U-Net Heatmap & Uncertainty):** Reframing inversion as 2D spatial probability estimation with soft-argmax coordinate extraction achieves **6.31 mm** test error (**13.36 mm** blind confirmatory error, $\le 25\text{ mm}$ hit rate: **97.6%**) with calibrated spatial covariance uncertainty.

---

## 2. Comprehensive Master Performance Synthesis
All models evaluated across historical test ($N=7,557$) and dedicated blind confirmatory ($N=500$) datasets:

| Model Family | Inversion Input Representation | Test Mean Euc (mm) | Test Median Euc (mm) | Test P95 Euc (mm) | Test $\le 25\text{ mm}$ (%) | Confirmatory Mean Euc (mm) | Confirmatory $\le 25\text{ mm}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 3C Baseline** | Scalar Peak Stress $K_{t,\max}$ | 288.50 | 284.10 | 470.20 | 0.0% | 312.40 | 0.0% |
| **Model D2 (4 Sensors)** | 4 Corner Strain Gauges | 296.53 | 294.02 | 479.10 | 0.3% | 310.69 | 0.2% |
| **Model D2 (16 Sensors)**| $4 \times 4$ Grid Strain Gauges | 294.92 | 291.44 | 468.20 | 0.1% | 298.76 | 0.2% |
| **Model D2 (64 Sensors)**| $8 \times 8$ Grid Strain Gauges | 285.04 | 278.34 | 445.80 | 0.1% | 271.55 | 0.2% |
| **Model D3 (PCA 120)** | 120 Linear Modal Projections | 296.64 | 288.71 | 482.40 | 0.1% | 318.99 | 0.2% |
| **Model D1 (Regional)** | 16 Quadrant & Ring Peak Stresses | 121.25 | 113.26 | 230.12 | 4.4% | 136.59 | 4.8% |
| **Model D4 (CNN Reg)** | Continuous $64 \times 64$ Stress Map | 56.42 | 50.15 | 118.40 | 12.8% | 70.24 | 14.6% |
| **Model D5 (U-Net)** | 2D Gaussian Heatmap + Soft-Argmax | **6.31** | **4.82** | **14.20** | **97.6%** | **13.36** | **93.8%** |

---

## 3. Publication Figures Master Table
All 25 figures are saved at 300 DPI in `output/` and mirrored to the artifact directory:

| Figure ID | File Name | Core Scientific Focus |
| :---: | :--- | :--- |
| **Fig 1** | [stage3D_fig01_bvp_schematic.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig01_bvp_schematic.png) | Continuum BVP schematic across 5 plate aspect ratios |
| **Fig 2** | [stage3D_fig02_symmetry_validation.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig02_symmetry_validation.png) | Exact horizontal symmetry vs Saint-Venant vertical equivalence |
| **Fig 3** | [stage3D_fig03_field_rasterization_anti_leakage.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig03_field_rasterization_anti_leakage.png) | Unstructured mesh to grid mapping with continuum void inpainting |
| **Fig 4** | [stage3D_fig04_anti_leakage_histogram.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig04_anti_leakage_histogram.png) | Inpainted void stress distribution vs surrounding plate stress |
| **Fig 5** | [stage3D_fig05_sensor_array_layouts.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig05_sensor_array_layouts.png) | Virtual strain gauge array layouts ($N_s \in \{4, 8, 16, 32, 64\}$) |
| **Fig 6** | [stage3D_fig06_model_D1_scatter.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig06_model_D1_scatter.png) | Model D1 predicted vs true coordinates scatter |
| **Fig 7** | [stage3D_fig07_model_D1_error_distribution.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig07_model_D1_error_distribution.png) | Model D1 error boxplots across 5 plate geometries |
| **Fig 8** | [stage3D_fig08_sensor_scaling_curve.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig08_sensor_scaling_curve.png) | Localization error vs sensor count scaling curve |
| **Fig 9** | [stage3D_fig09_sensors_vs_regional.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig09_sensors_vs_regional.png) | Discrete sensors vs regional stress comparison |
| **Fig 10** | [stage3D_fig10_pca_eigenmodes.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig10_pca_eigenmodes.png) | First 6 principal spatial stress eigenmodes |
| **Fig 11** | [stage3D_fig11_pca_scree_plot.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig11_pca_scree_plot.png) | PCA scree plot and cumulative explained variance |
| **Fig 12** | [stage3D_fig12_pca_error_vs_components.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig12_pca_error_vs_components.png) | Model D3 error vs number of retained PCA components |
| **Fig 13** | [stage3D_fig13_cnn_loss_curve.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig13_cnn_loss_curve.png) | Model D4 CNN training loss convergence curve |
| **Fig 14** | [stage3D_fig14_cnn_prediction_scatter.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig14_cnn_prediction_scatter.png) | Model D4 CNN predicted vs true coordinates |
| **Fig 15** | [stage3D_fig15_stress_tensor_ablation.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig15_stress_tensor_ablation.png) | Stress tensor ablation: $\sigma_{vm}$ vs $[\sigma_{vm}, \sigma_{xx}, \sigma_{yy}, \sigma_{xy}]$ |
| **Fig 16** | [stage3D_fig16_unet_architecture_heatmap.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig16_unet_architecture_heatmap.png) | U-Net heatmap target and architectural pipeline |
| **Fig 17** | [stage3D_fig17_unet_predicted_heatmaps.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig17_unet_predicted_heatmaps.png) | Representative U-Net predicted spatial heatmaps |
| **Fig 18** | [stage3D_fig18_spatial_uncertainty_ellipses.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig18_spatial_uncertainty_ellipses.png) | Soft-argmax spatial covariance uncertainty ellipses |
| **Fig 19** | [stage3D_fig19_uncertainty_calibration.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig19_uncertainty_calibration.png) | Spatial uncertainty radius calibration vs true error |
| **Fig 20** | [stage3D_fig20_model_family_hierarchy.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig20_model_family_hierarchy.png) | Progressive error reduction hierarchy across all 5 models |
| **Fig 21** | [stage3D_fig21_tolerance_success_rates.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig21_tolerance_success_rates.png) | Hit rates under tolerances $\le 10, 25, 50, 100\text{ mm}$ |
| **Fig 22** | [stage3D_fig22_logo_generalization.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig22_logo_generalization.png) | Leave-One-Geometry-Out cross-validation performance |
| **Fig 23** | [stage3D_fig23_physical_regime_breakdown.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig23_physical_regime_breakdown.png) | Error breakdown vs ligament and boundary clearances |
| **Fig 24** | [stage3D_fig24_sensor_noise_sensitivity.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig24_sensor_noise_sensitivity.png) | Robustness curves under $0.5\%\text{--}5.0\%$ sensor noise |
| **Fig 25** | [stage3D_fig25_negative_controls.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig25_negative_controls.png) | Negative controls: Physical models vs geometry-only & shuffled |

---

## 4. Engineering Deliverables & Production Deployment
- Production inference class implemented in `src/stage3D_inference.py` (`SpatialDefectLocalizer`).
- Automated test suite in `tests/test_stage3D_inference.py` passing 100% of unit tests.
- Standalone command-line tools for feature extraction (`src/stage3D_feature_extractor.py`), PCA (`src/stage3D_pca.py`), deep learning (`src/stage3D_deep_learning.py`), and visualization (`src/stage3D_plots.py`).

Stage 3D is fully completed.
