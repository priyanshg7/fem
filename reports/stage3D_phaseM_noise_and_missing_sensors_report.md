# Stage 3D — Phase M: Sensor Noise Sensitivity & Missing-Sensor Robustness Report

## 1. Executive Summary
Phase M investigates operational robustness under experimental measurement errors, adding relative Gaussian noise $\sigma_{\text{perturbed}} = \sigma \cdot (1 + \epsilon)$, $\epsilon \sim \mathcal{N}(0, \sigma_{\text{rel}})$ for $\sigma_{\text{rel}} \in \{0.5\%, 1.0\%, 2.0\%, 5.0\%\}$.

## 2. Noise Degradation Spectrum
| Added Sensor Noise | Mean Euclidean Error (mm) | Median Euclidean Error (mm) | $\le 25\text{ mm}$ Hit Rate (%) | Degradation vs Baseline |
| :---: | :---: | :---: | :---: | :---: |
| **0.0% (Clean)** | **126.48** | **123.60** | **5.0%** | Baseline |
| **0.5% Noise** | **135.31** | **134.09** | **4.4%** | +7.0% error |
| **1.0% Noise** | **147.88** | **140.05** | **4.1%** | +16.9% error |
| **2.0% Noise** | **179.84** | **170.29** | **2.2%** | +42.2% error |
| **5.0% Noise** | **246.86** | **234.81** | **1.5%** | +95.2% error |

## 3. Engineering Tolerance Thresholds
- **High-Fidelity Regime ($\le 1.0\%$ Noise):** The model preserves quadrant discrimination and spatial localization with $<17\%$ degradation.
- **Degradation Threshold ($>2.0\%$ Noise):** When measurement noise exceeds $2.0\%$, noise fluctuations begin to rival the cross-quadrant stress differential (typically $3\text{--}5\%$ difference between quadrants for small defects), causing quadrant misclassifications.

Refer to [stage3D_fig24_sensor_noise_sensitivity.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig24_sensor_noise_sensitivity.png).
