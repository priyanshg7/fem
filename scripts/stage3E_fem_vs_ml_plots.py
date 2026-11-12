"""
Publication-Quality FEM vs ML Physical Comparison:
Generates Figure 3E-6: Multi-Regime Qualitative Verification across:
1. Nominal Interior Defect
2. Ligament Interaction (Central Hole proximity)
3. Boundary Interaction (Plate Edge proximity)
4. Small Defect Limit (d/Dc ~ 0.08)
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from src.stage3D_cnn_unet import DefectHeatmapUNet, soft_argmax_2d
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field
from src.stage3D_feature_extractor import load_coords_and_stress

STAGE3E_OUT_DIR = "output/stage3E"
STAGE3E_MODEL_DIR = "models/stage3E"
CONF_DIR = "data/stage3E_confirmatory"
os.makedirs(STAGE3E_OUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def plot_fem_vs_ml_regimes():
    meta_path = os.path.join(CONF_DIR, "stage3E_untouched_confirmatory_metadata.csv")
    if not os.path.exists(meta_path):
        print("Confirmatory metadata not found.")
        return
    df_conf = pd.read_csv(meta_path)
    
    # Select 4 representative samples
    # 1. Nominal Interior: space filling, c_hole > 50, c_bnd > 50
    s_nom = df_conf[(df_conf['sample_type'] == 'space_filling_in_domain') & 
                    (df_conf['hole_to_hole_clearance_mm'] > 60) & 
                    (df_conf['boundary_clearance_mm'] > 60)].iloc[0]
                    
    # 2. Ligament interaction: sub_type == 'ligament_interaction'
    s_lig = df_conf[df_conf.get('sub_type', '') == 'ligament_interaction'].iloc[0]
    
    # 3. Boundary interaction: sub_type == 'boundary_interaction'
    s_bnd = df_conf[df_conf.get('sub_type', '') == 'boundary_interaction'].iloc[0]
    
    # 4. Small defect limit: defect_d_over_Dc < 0.10
    s_sml = df_conf.sort_values('defect_d_over_Dc').iloc[0]
    
    regimes = [
        ("Regime I: Nominal Interior Defect", s_nom),
        ("Regime II: Ligament Interaction (Near Central Hole)", s_lig),
        ("Regime III: Boundary Interaction (Near Plate Edge)", s_bnd),
        ("Regime IV: Small Defect Limit (Weak Stress Perturbation)", s_sml)
    ]
    
    # Load Model D5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DefectHeatmapUNet(in_channels=1).to(device)
    model_path = os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt")
    if not os.path.exists(model_path):
        model_path = "models/stage3D_spatial_inverse/model_D5_unet.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 15), dpi=300)
    
    col_titles = [
        "(a) True FEM Mesh & Von Mises Field",
        "(b) Inpainted Continuum Model Input",
        "(c) Model D5 Predicted Heatmap",
        "(d) Inversion & Uncertainty Ellipse"
    ]
    for col_idx, ctitle in enumerate(col_titles):
        axes[0, col_idx].set_title(ctitle, fontsize=12, fontweight='bold', pad=10)
        
    for r_idx, (r_name, row) in enumerate(regimes):
        L = row['plate_length_mm']
        H = row['plate_height_mm']
        Dc = row['central_hole_diameter_mm']
        d = row['defect_diameter_mm']
        xt = row['defect_center_x_mm']
        yt = row['defect_center_y_mm']
        
        fpath = row['field_file']
        data = np.load(fpath)
        nodes = data['nodes'] * 1000.0 # mm
        elements = data['elements']
        elem_vm = data['element_von_mises_psi']
        
        # Col 1: True FEM Stress Field (tripcolor or scatter)
        ax1 = axes[r_idx, 0]
        # Element centroid scatter
        elem_cents = nodes[elements].mean(axis=1)
        sc1 = ax1.scatter(elem_cents[:, 0], elem_cents[:, 1], c=elem_vm, s=0.8, cmap='viridis', vmin=0, vmax=350)
        # Draw holes
        ax1.add_patch(patches.Circle((L/2, H/2), Dc/2, facecolor='white', edgecolor='black', linewidth=1.2))
        ax1.add_patch(patches.Circle((xt, yt), d/2, facecolor='red', edgecolor='darkred', linewidth=1.2))
        ax1.set_xlim(0, L)
        ax1.set_ylim(0, H)
        ax1.set_aspect('equal')
        ax1.set_ylabel(f"{r_name}\nHeight (mm)", fontsize=10, fontweight='bold')
        if r_idx == 3: ax1.set_xlabel("Length (mm)", fontsize=10)
        
        # Col 2: Rasterized Model Input
        cnorm = normalize_coordinates(elem_cents, L, H)
        grid = rasterize_stress_field(cnorm, elem_vm, grid_res=64, method="nearest") / 100.0
        ax2 = axes[r_idx, 1]
        im2 = ax2.imshow(grid, cmap='viridis', origin='lower', extent=[0, L, 0, H], vmin=0, vmax=3.5)
        ax2.set_xlim(0, L)
        ax2.set_ylim(0, H)
        ax2.set_aspect('equal')
        if r_idx == 3: ax2.set_xlabel("Length (mm)", fontsize=10)
        
        # Col 3: Model D5 Output Heatmap
        inp_t = torch.tensor(grid.reshape(1, 1, 64, 64), dtype=torch.float32)
        with torch.no_grad():
            logits = model(inp_t)
            coords_norm, unc_norm, cov_norm = soft_argmax_2d(logits, beta=40.0)
            prob_map = torch.sigmoid(logits[0, 0]).cpu().numpy()
            
        ax3 = axes[r_idx, 2]
        im3 = ax3.imshow(prob_map, cmap='magma', origin='lower', extent=[0, L, 0, H])
        ax3.set_xlim(0, L)
        ax3.set_ylim(0, H)
        ax3.set_aspect('equal')
        if r_idx == 3: ax3.set_xlabel("Length (mm)", fontsize=10)
        
        # Col 4: Defect Comparison Overlay
        xp = coords_norm[0, 0].item() * L
        yp = coords_norm[0, 1].item() * H
        euc_err = np.hypot(xp - xt, yp - yt)
        
        c_norm = cov_norm[0].cpu().numpy()
        cov_phys = np.array([
            [c_norm[0, 0] * (L**2), c_norm[0, 1] * (L * H)],
            [c_norm[1, 0] * (L * H), c_norm[1, 1] * (H**2)]
        ])
        eigvals, eigvecs = np.linalg.eigh(cov_phys)
        # 95% confidence scaling: chi2(2) for 95% is ~5.991, sqrt is 2.447
        scale_95 = 2.447
        semi_a = np.sqrt(max(eigvals[1], 1e-4)) * scale_95
        semi_b = np.sqrt(max(eigvals[0], 1e-4)) * scale_95
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        
        ax4 = axes[r_idx, 3]
        ax4.imshow(prob_map, cmap='gray_r', origin='lower', extent=[0, L, 0, H], alpha=0.35)
        # True defect
        ax4.plot(xt, yt, 'ro', markersize=7, markeredgecolor='black', label=f'True Defect ({xt:.1f}, {yt:.1f})')
        # Pred defect
        ax4.plot(xp, yp, 'g+', markersize=10, markeredgewidth=2.0, label=f'Predicted ({xp:.1f}, {yp:.1f})\nError = {euc_err:.2f} mm')
        # Uncertainty ellipse
        ell = Ellipse((xp, yp), width=2*semi_a, height=2*semi_b, angle=angle, edgecolor='#1f77b4', facecolor='none', linestyle='--', linewidth=1.8, label='95% Conf. Ellipse')
        ax4.add_patch(ell)
        ax4.set_xlim(0, L)
        ax4.set_ylim(0, H)
        ax4.set_aspect('equal')
        ax4.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor='white', framealpha=0.9)
        if r_idx == 3: ax4.set_xlabel("Length (mm)", fontsize=10)
        
    plt.tight_layout()
    out_fig = os.path.join(STAGE3E_OUT_DIR, "stage3E_fig06_fem_vs_ml_regime_comparison.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved {out_fig}")

if __name__ == '__main__':
    plot_fem_vs_ml_regimes()

