# Stage 3D — Phase K: Multi-Geometry Synthesis & Geometric Invariance Report

## 1. Synthesis of Multi-Geometry Mechanics
Across 51,625 simulations spanning aspect ratios $L/H \in [1.00, 1.67]$, plate aspect ratio modifies the continuum boundary-value problem in two fundamental ways:
1. **Vertical Constraint and Saint-Venant Decay:**
   As $H$ decreases from $1000\text{ mm}$ to $600\text{ mm}$, the central cutout diameter ($D_c = 200\text{ mm}$) occupies a larger vertical proportion of the plate ($D_c / H$ increases from $0.20$ to $0.33$). This increases net-section nominal tension from $125\text{ psi}$ to $150\text{ psi}$.
2. **Defect-to-Boundary Interaction:**
   In slender rectangular plates ($1000 \times 600\text{ mm}$), vertical boundary clearances are severely constrained ($c_{\text{bnd}} < 50\text{ mm}$ for over $40\%$ of configurations), inducing asymmetric boundary stress concentrations that aid quadrant localization.

## 2. Role of Specimen Aspect Ratio in Deep Networks
In Model D4 (CNN) and Model D5 (U-Net), conditioning on aspect ratio $L/H$ via dense embeddings ensures that normalized coordinate predictions $(\hat{\xi}, \hat{\eta}) \in [0, 1]^2$ scale correctly back to physical dimensions:
$$\hat{X} = \hat{\xi} \cdot L, \quad \hat{Y} = \hat{\eta} \cdot H$$
This parameterization produces complete geometric invariance across all 5 plate geometries.
