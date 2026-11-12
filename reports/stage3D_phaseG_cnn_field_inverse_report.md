# Stage 3D — Phase G: CNN Spatial-Field Inverse Model Report

## 1. Executive Summary
Phase G implements **Model D4: Convolutional Neural Network (CNN) Spatial-Field Inversion**.
Model D4 processes the full 2D continuum stress field as an image representation, utilizing translation-invariant $3 \times 3$ convolutional feature hierarchies and global pooling concatenated with specimen geometry metadata ($L/H, D/D_c$).

### Breakout Performance:
- **Test Mean Euclidean Error:** **56.42 mm**
- **Blind Confirmatory Mean Euclidean Error:** **70.24 mm**
- **Compared to Model D1 (Regional Peak):** **53.5% error reduction** (from 121.25 mm down to 56.42 mm)
- **Compared to Stage 3C Scalar Maximum Stress:** **80.5% error reduction** (from 288.5 mm down to 56.42 mm)

## 2. Architecture and Stress Tensor Ablation
Model D4 consists of:
- Input: $(1, 64, 64)$ continuum inpainted Von Mises stress map.
- 4 Conv Blocks (32 $\to$ 64 $\to$ 128 $\to$ 256 channels) with BatchNorm and LeakyReLU(0.1).
- Global Average Pooling (GAP) collapsing spatial dimensions to a 256-D feature vector.
- Metadata conditioning: Specimen aspect ratio $L/H$ and diameter ratio $D/D_c$ concatenated to GAP representation.
- Fully connected regression head (258 $\to$ 128 $\to$ 64 $\to$ 2) with Sigmoid output mapping to $[0, 1]^2$.

### Stress Tensor Ablation:
- Single-channel $\sigma_{vm}$ alone: Mean Error = **56.42 mm**
- Multi-channel $[\sigma_{vm}, \sigma_{xx}, \sigma_{yy}, \sigma_{xy}]$: Mean Error = **54.18 mm**
The full stress tensor provides a minor ($4\%$) improvement, showing that the scalar Von Mises continuum invariant carries the overwhelming majority of spatial defect localization information.

Refer to [stage3D_fig13_cnn_loss_curve.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig13_cnn_loss_curve.png), [stage3D_fig14_cnn_prediction_scatter.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig14_cnn_prediction_scatter.png), and [stage3D_fig15_stress_tensor_ablation.png](file:///C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6/stage3D_fig15_stress_tensor_ablation.png).
