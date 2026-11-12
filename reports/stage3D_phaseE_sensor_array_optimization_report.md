# Stage 3D — Phase E: Fixed Sensor Array Optimization & Scaling Report

## 1. Executive Summary
Phase E investigates virtual strain-gauge sensor arrays distributed across the plate surface with sensor counts $N_s \in \{4, 8, 16, 32, 64\}$.
Features consist exclusively of local Von Mises stresses at fixed standardized coordinates $(\xi_i, \eta_i) \in [0, 1]^2$ plus plate aspect ratio $L/H$.

### Key Finding: The Spatial Blind Spot Phenomenon
- $N_s = 4$ (Far-Field Corners): Mean Euclidean Error = **296.53 mm**
- $N_s = 8$ (Corners + Edge Midpoints): Mean Euclidean Error = **296.82 mm**
- $N_s = 16$ ($4 \times 4$ Grid): Mean Euclidean Error = **294.92 mm**
- $N_s = 32$ ($6 \times 6$ Grid): Mean Euclidean Error = **290.75 mm**
- $N_s = 64$ ($8 \times 8$ Grid): Mean Euclidean Error = **285.04 mm** (Confirmatory: 271.55 mm)

## 2. Why Discrete Point Sensors Fail Compared to Regional Peak Stresses
While Model D1 (regional peak stress) achieves **121.25 mm**, discrete point sensors achieve only **285.04 mm**, even with 64 sensors.
This discrepancy is rooted in classical elasticity:
1. By the Kirsch solution, the stress perturbation $\Delta \sigma(r)$ produced by a circular defect decays quadratically with distance:
   $$\Delta \sigma(r) \propto \sigma_0 \left(\frac{a^2}{r^2}\right)$$
2. For small defects ($a \in [7.5, 30]\text{ mm}$), the perturbation decays to background tension ($100\text{ psi}$) within $2\text{--}3$ defect diameters ($20\text{--}60\text{ mm}$).
3. A fixed grid of 64 sensors has an inter-sensor spacing of $\Delta x \approx 120\text{ mm}$.
4. If a defect does not sit directly adjacent to a sensor node, the sensor measures only nominal uniform stress ($100\text{ psi}$) and is blind to the defect's presence!
5. In contrast, Model D1 computes the continuous mathematical supremum $\max_{x \in Q_k} \sigma_{vm}(x)$, which guarantees detection of the perturbation lobe wherever it falls within the quadrant.

Refer to [stage3D_fig05_sensor_array_layouts.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig05_sensor_array_layouts.png), [stage3D_fig08_sensor_scaling_curve.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig08_sensor_scaling_curve.png), and [stage3D_fig09_sensors_vs_regional.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig09_sensors_vs_regional.png).
