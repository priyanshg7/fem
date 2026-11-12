# Stage 3E: Structured Architectural & Algorithmic Ablation Report

**Project:** Static Analysis of a Plate with Defect Using Machine Learning  
**Stage:** Stage 3E — Model Robustness & Scientific Interpretation  
**Component:** Controlled Ablation Analysis of Model D5  
**Date:** November 12, 2026  

---

## 1. Executive Summary & Ablation Matrix

To isolate the exact causal factors contributing to Model D5's high localization precision, five controlled ablation experiments were conducted:
1. **Architectural Ablation (Skip Connections):** Standard U-Net (with skip connections) vs Fully Convolutional Network (FCN without skip connections).
2. **Soft-Argmax Temperature ($\beta$):** Varying the inverse softmax temperature $\beta \in \{10.0, 20.0, 40.0, 80.0, 160.0\}$.
3. **Inpainting Interpolation Strategy:** Nearest-Neighbor vs Inverse Distance Weighting (IDW, $k=4$) vs Zero-Fill.
4. **Coordinate Conditioning:** Implicit spatial normalization vs Explicit coordinate channels.
5. **Joint Sizing + Localization Multi-Task Network:** Augmenting the U-Net with an auxiliary global average pooling regression head to jointly predict defect diameter $d$.

---

## 2. Experimental Ablation Results

| Ablation Identifier | Experimental Variation | Mean Euclidean Error (mm) | Median Euclidean Error (mm) | Success Rate ($\le 25\text{ mm}$) | Sizing MAE (mm) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline Model D5** | U-Net (with skips), $\beta=40.0$, Nearest Inpainting | **$10.24$** | **$5.66$** | **$97.4\%$** | N/A (Unknown $d$) |
| **Architecture (No Skips)** | FCN Encoder-Decoder (Skips Removed) | **$21.68$** | **$7.26$** | **$91.2\%$** | N/A |
| **Temperature $\beta = 10.0$** | Soft temperature scaling | **$7.72$** | **$4.37$** | **$98.5\%$** | N/A |
| **Temperature $\beta = 20.0$** | Intermediate temperature | **$8.84$** | **$4.81$** | **$98.1\%$** | N/A |
| **Temperature $\beta = 80.0$** | Sharp peak temperature | **$11.48$** | **$6.09$** | **$96.8\%$** | N/A |
| **Temperature $\beta = 160.0$** | Near-hard argmax temperature | **$12.13$** | **$6.34$** | **$96.2\%$** | N/A |
| **Joint Multi-Task Head** | Simultaneous Heatmap + Defect Diameter Head | **$7.42$** | **$5.12$** | **$98.8\%$** | **$3.08\text{ mm}$** ($R^2=0.42$) |
| **IDW Inpainting ($k=4$)** | Inverse distance continuum interpolation | **$9.46$** | **$5.73$** | **$98.5\%$** | N/A |

![Figure 3E-4: Structured Ablation Studies](stage3E_fig04_ablation_comparison.png)

---

## 3. Detailed Scientific Findings

### 3.1 The Critical Necessity of Skip Connections
When the lateral skip connections between encoder and decoder stages were removed, the mean localization error more than doubled from **$10.24\text{ mm} \to 21.68\text{ mm}$** (a $112\%$ error increase!).
- **Mechanical Reason:** Deep convolutions progressively downsample the spatial resolution ($64 \times 64 \to 8 \times 8$). While this captures broad global tension distributions, the high-frequency steep gradients at the defect notch boundary are lost. Skip connections concatenate early high-resolution feature maps directly into the upsampling decoder, preserving sharp local notch coordinates.

### 3.2 Temperature Parameter ($\beta$) Trade-Off
- For large $\beta \ge 80$, the soft-argmax approaches a standard discrete argmax. Grid discretization quantization artifacts emerge, degrading mean error to $11.5\text{--}12.1\text{ mm}$.
- For moderate $\beta \in [10, 20]$, sub-pixel interpolation occurs via Gaussian probability weighting, lowering mean error to **$7.72\text{ mm}$** and median error to **$4.37\text{ mm}$**.

### 3.3 Multi-Task Synergy (Joint Localization and Sizing)
Adding an auxiliary regression head to predict normalized defect diameter ($d/D_c$) from the bottleneck features actually **improved** localization accuracy ($10.24\text{ mm} \to 7.42\text{ mm}$), while simultaneously predicting the defect diameter with a Mean Absolute Error of **$3.08\text{ mm}$** ($R^2 = 0.4212$).
- **Mechanism:** Multi-task regularization forces the bottleneck encoder to learn features that represent both the spatial position and the physical volume of the defect notch.
