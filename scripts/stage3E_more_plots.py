"""
Generate Stage 3E Figures 3E-2 and 3E-3.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE3E_OUT_DIR = "output/stage3E"
STAGE3E_MODEL_DIR = "models/stage3E"
AUDIT_MODEL_DIR = "models/stage3D_closeout"
CONF_DIR = "data/stage3E_confirmatory"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

# ----------------------------------------------------
# FIG 3E-2: Geometry Generalization (In-Domain vs Confirmatory vs LOGO)
# ----------------------------------------------------
with open(os.path.join(AUDIT_MODEL_DIR, "d5_logo_audit.json"), 'r') as f:
    logo_data = json.load(f)
with open(os.path.join(STAGE3E_MODEL_DIR, "confirmatory_results.json"), 'r') as f:
    conf_data = json.load(f)

geoms = ['square_1000x1000', 'rectangular_1000x900', 'rectangular_1000x800', 'rectangular_1000x700', 'rectangular_1000x600']
geom_labels = ['1000x1000\n(AR=1.0)', '1000x900\n(AR=1.11)', '1000x800\n(AR=1.25)', '1000x700\n(AR=1.43)', '1000x600\n(AR=1.67)']

logo_means = [logo_data[g]['Mean_Euclidean_mm'] for g in geoms]
conf_means = [conf_data['By_Geometry_Family'][g]['Mean_Euclidean_mm'] for g in geoms]
conf_medians = [conf_data['By_Geometry_Family'][g]['Median_Euclidean_mm'] for g in geoms]

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(geoms))
width = 0.28

r1 = ax.bar(x - width, conf_medians, width, label='Confirmatory Median Error (mm)', color='#2e6f40', alpha=0.85, edgecolor='black', linewidth=0.8)
r2 = ax.bar(x, conf_means, width, label='Confirmatory Mean Error (mm)', color='#4a7c59', alpha=0.85, edgecolor='black', linewidth=0.8)
r3 = ax.bar(x + width, logo_means, width, label='Leave-One-Geometry-Out (LOGO) Mean Error (mm)', color='#2980b9', alpha=0.85, edgecolor='black', linewidth=0.8)

for r in r1:
    h = r.get_height()
    ax.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for r in r2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for r in r3:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Localization Error (mm)', fontsize=11, fontweight='bold')
ax.set_title('Figure 3E-2: Geometric Generalization Across Aspect Ratios (L/H = 1.0 to 1.67)\nUntouched Confirmatory (N=500) vs Leave-One-Geometry-Out (LOGO)', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(geom_labels, fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
ax.set_ylim(0, 35)
plt.tight_layout()
plt.savefig(os.path.join(STAGE3E_OUT_DIR, "stage3E_fig02_geometry_and_logo_generalization.png"))
plt.close()
print("Saved stage3E_fig02_geometry_and_logo_generalization.png")

# ----------------------------------------------------
# FIG 3E-3: Diameter and Clearance Sensitivity
# ----------------------------------------------------
df_conf = pd.read_csv(os.path.join(CONF_DIR, "stage3E_untouched_confirmatory_metadata.csv"))
X_conf = np.load(os.path.join(CONF_DIR, "X_conf_64.npy"))
import torch
from src.stage3D_cnn_unet import DefectHeatmapUNet, soft_argmax_2d
base_model = DefectHeatmapUNet(in_channels=1)
base_model.load_state_dict(torch.load(os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt"), map_location='cpu'))
base_model.eval()

with torch.no_grad():
    logits = base_model(torch.tensor(X_conf.reshape(-1, 1, 64, 64), dtype=torch.float32))
    coords_te, _, _ = soft_argmax_2d(logits, beta=40.0)
    pred_x = coords_te[:, 0].numpy() * df_conf['plate_length_mm'].values
    pred_y = coords_te[:, 1].numpy() * df_conf['plate_height_mm'].values
    true_x = df_conf['defect_center_x_mm'].values
    true_y = df_conf['defect_center_y_mm'].values
    euc_errors = np.hypot(pred_x - true_x, pred_y - true_y)

df_conf['euc_error_mm'] = euc_errors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: Error vs Diameter Ratio (d/Dc)
bins_d = [0.08, 0.15, 0.22, 0.30, 0.50]
labels_d = ['0.08-0.15\n(Very Small)', '0.15-0.22\n(Nominal)', '0.22-0.30\n(Moderate)', '0.30-0.45\n(Out-of-Domain)']
df_conf['d_bin'] = pd.cut(df_conf['defect_d_over_Dc'], bins=bins_d, labels=labels_d)
mean_err_d = df_conf.groupby('d_bin', observed=False)['euc_error_mm'].mean()
med_err_d = df_conf.groupby('d_bin', observed=False)['euc_error_mm'].median()

xd = np.arange(len(labels_d))
w = 0.35
ax1.bar(xd - w/2, mean_err_d, w, label='Mean Error (mm)', color='#d9534f', alpha=0.85, edgecolor='black', linewidth=0.8)
ax1.bar(xd + w/2, med_err_d, w, label='Median Error (mm)', color='#f0ad4e', alpha=0.85, edgecolor='black', linewidth=0.8)
for i, v in enumerate(mean_err_d):
    ax1.annotate(f'{v:.1f}', xy=(xd[i] - w/2, v), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
for i, v in enumerate(med_err_d):
    ax1.annotate(f'{v:.1f}', xy=(xd[i] + w/2, v), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
ax1.set_title('(a) Sensitivity to Defect Sizing Ratio (d / Dc)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Localization Error (mm)', fontsize=10, fontweight='bold')
ax1.set_xticks(xd)
ax1.set_xticklabels(labels_d, fontsize=9)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

# Panel B: Error vs Hole Clearance
bins_c = [0, 15, 30, 60, 500]
labels_c = ['< 15 mm\n(Near Ligament)', '15-30 mm\n(Close)', '30-60 mm\n(Moderate)', '> 60 mm\n(Far Field)']
df_conf['c_bin'] = pd.cut(df_conf['hole_to_hole_clearance_mm'], bins=bins_c, labels=labels_c)
mean_err_c = df_conf.groupby('c_bin', observed=False)['euc_error_mm'].mean()
med_err_c = df_conf.groupby('c_bin', observed=False)['euc_error_mm'].median()

xc = np.arange(len(labels_c))
ax2.bar(xc - w/2, mean_err_c, w, label='Mean Error (mm)', color='#2e6f40', alpha=0.85, edgecolor='black', linewidth=0.8)
ax2.bar(xc + w/2, med_err_c, w, label='Median Error (mm)', color='#66bb6a', alpha=0.85, edgecolor='black', linewidth=0.8)
for i, v in enumerate(mean_err_c):
    ax2.annotate(f'{v:.1f}', xy=(xc[i] - w/2, v), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
for i, v in enumerate(med_err_c):
    ax2.annotate(f'{v:.1f}', xy=(xc[i] + w/2, v), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
ax2.set_title('(b) Sensitivity to Hole-to-Hole Clearance (c_hole)', fontsize=11, fontweight='bold')
ax2.set_xticks(xc)
ax2.set_xticklabels(labels_c, fontsize=9)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

plt.suptitle('Figure 3E-3: Geometric Parameter Sensitivity Analysis on Confirmatory Benchmark', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STAGE3E_OUT_DIR, "stage3E_fig03_diameter_and_clearance_sensitivity.png"))
plt.close()
print("Saved stage3E_fig03_diameter_and_clearance_sensitivity.png")
