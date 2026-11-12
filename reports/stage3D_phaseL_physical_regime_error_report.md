# Stage 3D — Phase L: Physical Regime Error Diagnosis & Edge Case Analysis

## 1. Error Stratification Across Physical Clearances
To establish where spatial inversion is most reliable, the test dataset was partitioned into four distinct physical regimes based on ligament clearance $c_{\text{lig}}$ and boundary clearance $c_{\text{bnd}}$:

| Physical Regime | Clearance Definition | Sample Count | Model D1 Mean Euc (mm) | Model D4 Mean Euc (mm) | Model D5 Mean Euc (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Severe Ligament Interaction** | $c_{\text{lig}} \le 25\text{ mm}$ | 1,420 | 108.4 mm | 42.1 mm | **4.12 mm** |
| **Severe Boundary Interaction** | $c_{\text{bnd}} \le 25\text{ mm}$ | 1,280 | 114.2 mm | 48.6 mm | **5.25 mm** |
| **Intermediate Clearance** | $25 < c_{\text{lig}}, c_{\text{bnd}} \le 100\text{ mm}$ | 3,150 | 124.8 mm | 58.2 mm | **6.48 mm** |
| **Far-Field Isolated Defect** | $c_{\text{lig}} > 100\text{ mm}, c_{\text{bnd}} > 100\text{ mm}$ | 1,707 | 138.5 mm | 69.4 mm | **8.92 mm** |

## 2. Key Physical Insight: Interaction Enhances Localization
- Paradoxically, **tight clearances yield the highest localization accuracy** ($4.12\text{ mm}$ vs $8.92\text{ mm}$).
- **Mechanics Explanation:** When a defect approaches the central hole ($c_{\text{lig}} \le 25\text{ mm}$), intense stress concentration develops in the narrow ligament ($K_t > 3.8$). This localized spike creates a sharp, unmistakable gradient signature in the 2D stress field that the U-Net pinpoints effortlessly.
- Conversely, for far-field isolated defects, the stress perturbation is subtle ($\Delta \sigma < 8\text{ psi}$ above baseline), creating a flatter heatmap lobe and wider uncertainty ellipse.

Refer to [stage3D_fig23_physical_regime_breakdown.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig23_physical_regime_breakdown.png).
