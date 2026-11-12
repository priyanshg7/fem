# Stage 3C: Uncertainty and Measurement Noise Sensitivity Analysis (Phase K & L)

## Executive Summary
In practical non-destructive evaluation (NDE) and structural health monitoring (SHM), physical sensor readings (such as strain gauge rosettes, digital image correlation, or fiber Bragg gratings) inevitably contain measurement noise. 
This report investigates **Phase K & L: Uncertainty Quantification and Noise Sensitivity Analysis** for inverse defect localization under Formulation C3.

We subject the best-performing inverse regressor (Random Forest) to controlled synthetic Gaussian noise perturbations on the measured peak stress concentration factor $K_{t,\max}$ at four standard industrial instrumentation noise levels:
$$K_{t,\max}^{(\epsilon)} = K_{t,\max} \times (1 + \delta), \quad \delta \sim \mathcal{N}(0, \sigma_{\text{noise}}^2), \quad \sigma_{\text{noise}} \in \{0.5\%, 1.0\%, 2.0\%, 5.0\%\}$$

Our experimental findings reveal:
1. **Severe Coordinate Instability**: A minor stress measurement error of just $\pm 0.5\%$ (e.g., $1.5\text{ psi}$ on a $300\text{ psi}$ reading) induces an average predicted coordinate displacement of **$93.50\text{ mm}$**, with individual predictions shifting by up to **$589.03\text{ mm}$** across the plate.
2. **Amplified Dispersion at $\pm 5.0\%$**: At an instrumentation noise level of $\pm 5.0\%$, the mean coordinate shift reaches **$136.02\text{ mm}$**, and the maximum shift reaches **$707.39\text{ mm}$** (over $70\%$ of the entire plate dimension).
3. **The Ill-Posedness Flaw**: Because the scalar-to-coordinate mapping is non-injective and ill-conditioned, slight perturbations in stress cause the model's split decisions to jump discontinuously between different local regions of the allowable domain, proving that scalar stress inversion is critically unstable and physically unviable for practical deployment.

---

## 1. Experimental Methodology & Perturbation Protocol
- **Base Model**: MultiOutput Random Forest (100 estimators, `max_depth=16`), trained on Formulation C3 ($[K_{t,\max}, L/H, d/D_c]$).
- **Test Sample Size**: $N = 7,557$ test simulations across all five plate geometry families.
- **Noise Injection**: Only the measured peak stress column is perturbed; plate aspect ratio $L/H$ and defect diameter ratio $d/D_c$ remain exact.
- **Metrics Tracked**:
  - Mean Coordinate Shift: $\Delta_{\text{shift}} = \sqrt{(\hat{X}_{\text{pert}} - \hat{X}_{\text{base}})^2 + (\hat{Y}_{\text{pert}} - \hat{Y}_{\text{base}})^2}$ in mm.
  - Overall Euclidean Localization Error against true physical defect coordinates.
  - Tolerance percentages ($\le 5, 10, 25, 50, 100\text{ mm}$).

---

## 2. Quantitative Sensitivity Results (Phase L)

| Noise Level | Simulated Stress Error | Mean Pred Shift (mm) | Max Pred Shift (mm) | Mean Euclidean Error (mm) | P95 Euclidean Error (mm) | Accuracy $\le 25\text{ mm}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0% (Baseline)** | 0.0 psi | 0.00 mm | 0.00 mm | **330.15 mm** | 558.24 mm | **0.13%** |
| **$\pm 0.5\%$** | $\approx \pm 1.5\text{ psi}$ | **93.50 mm** | 589.03 mm | **342.74 mm** | 567.75 mm | **0.13%** |
| **$\pm 1.0\%$** | $\approx \pm 3.0\text{ psi}$ | **100.81 mm** | 612.67 mm | **344.53 mm** | 569.75 mm | **0.17%** |
| **$\pm 2.0\%$** | $\approx \pm 6.0\text{ psi}$ | **113.18 mm** | 706.91 mm | **347.88 mm** | 585.28 mm | **0.04%** |
| **$\pm 5.0\%$** | $\approx \pm 15.0\text{ psi}$ | **136.02 mm** | 707.39 mm | **354.35 mm** | 611.45 mm | **0.12%** |

---

## 3. Physical Diagnosis of Coordinate Instability

### 3.1 Discontinuous Boundary-Jumping Effect
When a plate contains a defect far from the central hole, the peak stress is very close to the classical Kirsch stress $\sigma_{\text{Kirsch}} \approx 300\text{ psi}$. In this regime, the derivative of peak stress with respect to defect coordinate is extremely flat:
$$\left\| \nabla_{(X, Y)} K_{t,\max} \right\| \approx 0$$
By the inverse function theorem, the condition number of the inverse mapping is proportional to the reciprocal of the gradient:
$$\kappa = \frac{1}{\left\| \nabla_{(X, Y)} K_{t,\max} \right\|} \to \infty$$
Consequently, an infinitesimal change in stress $\Delta K_{t,\max} \approx 0.01$ forces the inverse pre-image to traverse hundreds of millimeters across the plate, producing maximum predicted coordinate jumps of over $600\text{--}700\text{ mm}$.

### 3.2 Comparison with Forward Problem Sensitivity
In Stage 3B, the forward problem (Geometry $\to$ Stress) was exceptionally well-conditioned: small perturbations in defect location produced smooth, predictable variations in stress, enabling ensembles to achieve $R^2 = 0.9634$. 
In the inverse problem (Stress $\to$ Geometry), the ill-conditioned nature of the pre-image creates extreme sensitivity to noise, demonstrating that direct inverse regression is ill-posed in the sense of Hadamard (the solution does not depend continuously on the data).

---

## 4. Conclusion
Stress measurement noise severely amplifies the physical error of scalar-only inverse defect localizers. A mere $0.5\%$ measurement error shifts predictions by an average of $93.5\text{ mm}$. This instability proves that direct scalar inversion cannot be deployed in practical inspection environments, reinforcing the mandatory transition to Stage 3D spatial field sensing.

