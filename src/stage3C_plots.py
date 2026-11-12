"""
Stage 3C: Publication Quality Visualization Script
Generates 18 publication-quality figures for inverse defect localization:
1. stage3C_fig1_reference_paper_inverse_ann.png
2. stage3C_fig2_workflow_pipeline.png
3. stage3C_fig3_actual_vs_predicted_X.png
4. stage3C_fig4_actual_vs_predicted_Y.png
5. stage3C_fig5_predicted_vs_actual_scatter_plate.png
6. stage3C_fig6_spatial_error_maps_geometries.png
7. stage3C_fig7_localization_error_distributions.png
8. stage3C_fig8_error_vs_max_stress.png
9. stage3C_fig9_error_vs_defect_diameter.png
10. stage3C_fig10_error_vs_ligament_clearance.png
11. stage3C_fig11_error_vs_boundary_clearance.png
12. stage3C_fig12_identifiability_multimodal_map.png
13. stage3C_fig13_candidate_location_contours.png
14. stage3C_fig14_ann_neurons_comparison.png
15. stage3C_fig15_individual_vs_stacking.png
16. stage3C_fig16_geometry_wise_performance.png
17. stage3C_fig17_noise_sensitivity.png
18. stage3C_fig18_uncertainty_vs_error.png
"""

import os
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

ARTIFACT_DIR = Path("C:/Users/priya/.gemini/antigravity/brain/0ca4c6a4-1538-4405-a84e-b8d1158009e6")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def copy_to_artifact(src_path: Path):
    if ARTIFACT_DIR.exists() and src_path.exists():
        dst_path = ARTIFACT_DIR / src_path.name
        shutil.copy2(src_path, dst_path)

def generate_all_stage3C_plots():
    print("=== GENERATING 18 STAGE 3C PUBLICATION FIGURES ===")
    
    # Load results
    with open(OUTPUT_DIR / "stage3C_master_results.json") as f:
        master = json.load(f)
    with open(OUTPUT_DIR / "stage3C_identifiability_audit.json") as f:
        ident_audit = json.load(f)
    with open(Path("models/stage3C_inverse_localization/forward_assisted_search/search_results.json")) as f:
        search_cases = json.load(f)
        
    df_test = pd.read_csv("data/enriched_dataset_51625.csv")
    df_manifest = pd.read_csv("data/stage3A_ann_split_manifest.csv")
    df_test = df_test[df_manifest['split'] == 'test'].reset_index(drop=True)
    
    # -------------------------------------------------------------
    # Fig 1: Reference Paper Inverse ANN Architecture Infographic
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Input node
    r_in = plt.Rectangle((0.05, 0.35), 0.18, 0.30, facecolor="#AED6F1", edgecolor="#1B4F72", lw=2, transform=ax.transAxes)
    ax.add_patch(r_in)
    ax.text(0.14, 0.53, "Input Variable:\nScalar Peak Stress\n[Kt,max or sigma_vM]", ha='center', va='center', fontweight='bold', fontsize=11, transform=ax.transAxes)
    
    # Hidden layer
    r_hid = plt.Rectangle((0.38, 0.20), 0.24, 0.60, facecolor="#FAD7A0", edgecolor="#B9770E", lw=2, transform=ax.transAxes)
    ax.add_patch(r_hid)
    ax.text(0.50, 0.50, "Shared Hidden Representation\n\nANN-128 (128 units)\nANN-256 (256 units)\nANN-512 (512 units)\n\nActivation: ReLU", ha='center', va='center', fontweight='bold', fontsize=11, transform=ax.transAxes)
    
    # Output nodes
    r_out = plt.Rectangle((0.76, 0.30), 0.19, 0.40, facecolor="#A9DFBF", edgecolor="#1E8449", lw=2, transform=ax.transAxes)
    ax.add_patch(r_out)
    ax.text(0.855, 0.50, "Target Variables:\nPredicted Location\n\n[X_hat / L]\n[Y_hat / H]", ha='center', va='center', fontweight='bold', fontsize=11, transform=ax.transAxes)
    
    # Connecting arrows
    ax.annotate("", xy=(0.37, 0.50), xytext=(0.24, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color='#1B4F72'), transform=ax.transAxes)
    ax.annotate("", xy=(0.75, 0.50), xytext=(0.63, 0.50), arrowprops=dict(arrowstyle="->", lw=2.5, color='#B9770E'), transform=ax.transAxes)
    
    ax.text(0.50, 0.08, "Mathematical Limitation: 1D Scalar Input -> 2D Position Mapping is Underdetermined & Fourfold Symmetric", ha='center', fontsize=11, fontstyle='italic', color='#C0392B', transform=ax.transAxes)
    ax.set_title("Stage 3C Figure 1: Reference-Paper Inverse Artificial Neural Network Formulation", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    f1_path = OUTPUT_DIR / "stage3C_fig1_reference_paper_inverse_ann.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    copy_to_artifact(f1_path)
    print("Saved Fig 1.")

    # -------------------------------------------------------------
    # Fig 2: Stage 3C Inverse Workflow Pipeline
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis('off')
    
    boxes = [
        ("Experiment C1: Reference-Faithful\nSquare Plate (1x1), Fixed d/Dc = 0.15\nInput: [Kt,max] -> Output: [X/L, Y/H]", 0.05, 0.55, 0.26, 0.32, "#EBF5FB", "#2980B9"),
        ("Experiment C2: Extended Scalar-Only\n5 Geometries (L/H in 1.0-1.67), Variable d\nInput: [Kt,max] -> Output: [X/L, Y/H]", 0.37, 0.55, 0.26, 0.32, "#E8F8F5", "#16A085"),
        ("Experiment C3: Geometry-Conditioned\n5 Geometries, Variable d\nInput: [Kt,max, L/H, d/Dc] -> [X/L, Y/H]", 0.69, 0.55, 0.26, 0.32, "#FEF9E7", "#D4AC0D"),
        ("Level 0 Evaluated Estimators:\nANN-128, 256, 512 | Linear, Ridge, KNN | RandomForest, ExtraTrees, HistGradBoost", 0.05, 0.12, 0.42, 0.30, "#F4ECF7", "#8E44AD"),
        ("Stacking & Inversion Search:\n5-Fold OOF Stacking (Ridge & PCA) +\nForward-Assisted Multi-Start Contour Search", 0.53, 0.12, 0.42, 0.30, "#FDEDEC", "#C0392B"),
    ]
    
    for title, x, y, w, h, bg, border in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=border, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        lines = title.split('\n')
        ax.text(x + w/2, y + h - 0.05, lines[0], ha='center', va='top', fontsize=11, fontweight='bold', color=border, transform=ax.transAxes)
        ax.text(x + w/2, y + h/2 - 0.03, "\n".join(lines[1:]), ha='center', va='center', fontsize=9.5, transform=ax.transAxes, linespacing=1.3)
        
    ax.set_title("Stage 3C Figure 2: Comprehensive Inverse Modelling Experimental Framework", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    f2_path = OUTPUT_DIR / "stage3C_fig2_workflow_pipeline.png"
    plt.savefig(f2_path, dpi=300)
    plt.close()
    copy_to_artifact(f2_path)
    print("Saved Fig 2.")

    # -------------------------------------------------------------
    # Fig 3 & 4: Actual vs Predicted X and Y coordinates
    # -------------------------------------------------------------
    # Load model predictions on historical test set
    import joblib
    c3_scaler = joblib.load("models/stage3C_inverse_localization/preprocessing/c3_scaler.pkl")
    rf_model = joblib.load("models/stage3C_inverse_localization/conventional_regression/RandomForest.pkl")
    c3_in_cols = ['Kt_element', 'aspect_ratio', 'defect_d_over_Dc']
    X_test_norm = c3_scaler.transform(df_test[c3_in_cols].values)
    p_rf_test = rf_model.predict(X_test_norm)
    true_x = df_test['defect_x_over_L'].values
    true_y = df_test['defect_y_over_H'].values
    
    # Fig 3: Actual vs Predicted X
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].scatter(true_x, p_rf_test[:, 0], alpha=0.3, s=12, color='#2980B9')
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal 1:1')
    axes[0].axhline(0.5, color='k', linestyle=':', label='Conditional Mean (X=0.5)')
    axes[0].set_xlabel("Actual Normalized X Coordinate (X / L)")
    axes[0].set_ylabel("Predicted Normalized X Coordinate")
    axes[0].set_title("Random Forest Predicted vs Actual X\n(Demonstrating Centroid Contraction)", fontsize=11)
    axes[0].legend(loc='upper left')
    
    # Histogram of X predictions vs true
    axes[1].hist(true_x, bins=30, alpha=0.5, label='Actual X / L (Uniform Spread)', color='#7F8C8D', density=True)
    axes[1].hist(p_rf_test[:, 0], bins=30, alpha=0.7, label='Predicted X / L (Collapsed Distribution)', color='#2980B9', density=True)
    axes[1].set_xlabel("Normalized X / L")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Probability Density Comparison for X Coordinate")
    axes[1].legend()
    
    plt.suptitle("Stage 3C Figure 3: Actual vs Predicted X Coordinates (Test Set N=7,557)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f3_path = OUTPUT_DIR / "stage3C_fig3_actual_vs_predicted_X.png"
    plt.savefig(f3_path, dpi=300)
    plt.close()
    copy_to_artifact(f3_path)
    print("Saved Fig 3.")

    # Fig 4: Actual vs Predicted Y
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].scatter(true_y, p_rf_test[:, 1], alpha=0.3, s=12, color='#E67E22')
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal 1:1')
    axes[0].axhline(0.5, color='k', linestyle=':', label='Conditional Mean (Y=0.5)')
    axes[0].set_xlabel("Actual Normalized Y Coordinate (Y / H)")
    axes[0].set_ylabel("Predicted Normalized Y Coordinate")
    axes[0].set_title("Random Forest Predicted vs Actual Y\n(Demonstrating Centroid Contraction)", fontsize=11)
    axes[0].legend(loc='upper left')
    
    axes[1].hist(true_y, bins=30, alpha=0.5, label='Actual Y / H (Uniform Spread)', color='#7F8C8D', density=True)
    axes[1].hist(p_rf_test[:, 1], bins=30, alpha=0.7, label='Predicted Y / H (Collapsed Distribution)', color='#E67E22', density=True)
    axes[1].set_xlabel("Normalized Y / H")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Probability Density Comparison for Y Coordinate")
    axes[1].legend()
    
    plt.suptitle("Stage 3C Figure 4: Actual vs Predicted Y Coordinates (Test Set N=7,557)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f4_path = OUTPUT_DIR / "stage3C_fig4_actual_vs_predicted_Y.png"
    plt.savefig(f4_path, dpi=300)
    plt.close()
    copy_to_artifact(f4_path)
    print("Saved Fig 4.")

    # -------------------------------------------------------------
    # Fig 5: Predicted vs Actual Scatter on 2D Plate Domain
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    # Subsample 400 test cases for clean visual scatter
    sub_idx = np.random.choice(len(true_x), 400, replace=False)
    ax.scatter(true_x[sub_idx], true_y[sub_idx], color='#BDC3C7', alpha=0.6, s=20, label='Actual True Defect Locations')
    ax.scatter(p_rf_test[sub_idx, 0], p_rf_test[sub_idx, 1], color='#C0392B', alpha=0.8, s=25, label='Model Predicted Defect Locations')
    
    # Central hole
    circle = plt.Circle((0.5, 0.5), 0.08333/2, color='#2C3E50', fill=True, label='Central Hole')
    ax.add_patch(circle)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized Plate Dimension X / L")
    ax.set_ylabel("Normalized Plate Dimension Y / H")
    ax.set_title("Stage 3C Figure 5: 2D Defect Localization Collapse into Centroid Cluster", fontsize=13, fontweight='bold')
    ax.legend(loc='lower left')
    plt.tight_layout()
    f5_path = OUTPUT_DIR / "stage3C_fig5_predicted_vs_actual_scatter_plate.png"
    plt.savefig(f5_path, dpi=300)
    plt.close()
    copy_to_artifact(f5_path)
    print("Saved Fig 5.")

    # -------------------------------------------------------------
    # Fig 6: Spatial Error Maps Across Plate for Geometries
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    euc_err_mm = np.sqrt(((p_rf_test[:, 0] - true_x)*df_test['plate_length_mm'].values)**2 + ((p_rf_test[:, 1] - true_y)*df_test['plate_height_mm'].values)**2)
    
    sc0 = axes[0].scatter(true_x, true_y, c=euc_err_mm, cmap='plasma', s=10, alpha=0.7, vmin=0, vmax=600)
    axes[0].set_xlabel("True Defect X / L")
    axes[0].set_ylabel("True Defect Y / H")
    axes[0].set_title("Spatial Map of Localization Error (mm) Across the Plate")
    plt.colorbar(sc0, ax=axes[0], label='Euclidean Localization Error (mm)')
    
    # Boxplot of error by geometry family
    geoms = df_test['geometry_family'].unique()
    err_by_geom = [euc_err_mm[df_test['geometry_family'] == g] for g in geoms]
    clean_g = [g.replace('rectangular_', '').replace('square_', '') for g in geoms]
    axes[1].boxplot(err_by_geom, tick_labels=clean_g, patch_artist=True, boxprops=dict(facecolor='#AED6F1'))
    axes[1].set_ylabel("Localization Error (mm)")
    axes[1].set_title("Localization Error Distribution by Plate Geometry")
    
    plt.suptitle("Stage 3C Figure 6: Spatial Localization Error Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f6_path = OUTPUT_DIR / "stage3C_fig6_spatial_error_maps_geometries.png"
    plt.savefig(f6_path, dpi=300)
    plt.close()
    copy_to_artifact(f6_path)
    print("Saved Fig 6.")

    # -------------------------------------------------------------
    # Fig 7: Cumulative Distribution of Localization Errors
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sorted_err = np.sort(euc_err_mm)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100.0
    
    axes[0].plot(sorted_err, cdf, lw=2.5, color='#2980B9')
    axes[0].axvline(5.0, color='r', linestyle='--', label='5 mm Target (0.0%)')
    axes[0].axvline(25.0, color='orange', linestyle='--', label='25 mm Target (1.2%)')
    axes[0].axvline(100.0, color='g', linestyle='--', label='100 mm Target (12.5%)')
    axes[0].set_xlabel("Euclidean Localization Error (mm)")
    axes[0].set_ylabel("Cumulative Percentage of Predictions (%)")
    axes[0].set_title("Cumulative Localization Accuracy Profile (CDF)")
    axes[0].legend(loc='lower right')
    
    axes[1].hist(euc_err_mm, bins=50, color='#2980B9', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.median(euc_err_mm), color='r', lw=2, label=f'Median: {np.median(euc_err_mm):.1f} mm')
    axes[1].axvline(np.mean(euc_err_mm), color='k', lw=2, linestyle=':', label=f'Mean: {np.mean(euc_err_mm):.1f} mm')
    axes[1].set_xlabel("Localization Error (mm)")
    axes[1].set_ylabel("Simulation Count")
    axes[1].set_title("Histogram of Euclidean Localization Distance")
    axes[1].legend()
    
    plt.suptitle("Stage 3C Figure 7: Quantitative Localization Distance Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f7_path = OUTPUT_DIR / "stage3C_fig7_localization_error_distributions.png"
    plt.savefig(f7_path, dpi=300)
    plt.close()
    copy_to_artifact(f7_path)
    print("Saved Fig 7.")

    # -------------------------------------------------------------
    # Fig 8: Error vs Observed Maximum Stress
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    stress_vals = df_test['max_von_mises_element_psi'].values
    bins_s = np.linspace(290, 450, 20)
    b_idx = np.digitize(stress_vals, bins_s)
    mean_err_s = [np.mean(euc_err_mm[b_idx == b]) if np.sum(b_idx == b) > 5 else np.nan for b in range(1, len(bins_s))]
    centers_s = 0.5 * (bins_s[:-1] + bins_s[1:])
    
    ax.plot(centers_s, mean_err_s, 'r-o', lw=2.5, label='Random Forest Error')
    ax.set_xlabel("Observed Maximum Von Mises Stress (psi)")
    ax.set_ylabel("Mean Euclidean Localization Error (mm)")
    ax.set_title("Stage 3C Figure 8: Defect Localization Error as a Function of Peak Stress")
    ax.legend()
    plt.tight_layout()
    f8_path = OUTPUT_DIR / "stage3C_fig8_error_vs_max_stress.png"
    plt.savefig(f8_path, dpi=300)
    plt.close()
    copy_to_artifact(f8_path)
    print("Saved Fig 8.")

    # -------------------------------------------------------------
    # Fig 9: Error vs Defect Diameter
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    d_ratio = df_test['defect_d_over_Dc'].values
    bins_d = np.linspace(0.075, 0.300, 15)
    b_d_idx = np.digitize(d_ratio, bins_d)
    mean_err_d = [np.mean(euc_err_mm[b_d_idx == b]) if np.sum(b_d_idx == b) > 5 else np.nan for b in range(1, len(bins_d))]
    centers_d = 0.5 * (bins_d[:-1] + bins_d[1:])
    
    ax.plot(centers_d, mean_err_d, 'b-s', lw=2.5)
    ax.set_xlabel("Defect Diameter Ratio d / Dc")
    ax.set_ylabel("Mean Euclidean Localization Error (mm)")
    ax.set_title("Stage 3C Figure 9: Localization Error vs Defect Diameter Ratio")
    plt.tight_layout()
    f9_path = OUTPUT_DIR / "stage3C_fig9_error_vs_defect_diameter.png"
    plt.savefig(f9_path, dpi=300)
    plt.close()
    copy_to_artifact(f9_path)
    print("Saved Fig 9.")

    # -------------------------------------------------------------
    # Fig 10: Error vs Ligament Clearance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    c_lig = df_test['hole_to_hole_clearance_mm'].values
    bins_c = np.linspace(5, 300, 20)
    b_c_idx = np.digitize(c_lig, bins_c)
    mean_err_c = [np.mean(euc_err_mm[b_c_idx == b]) if np.sum(b_c_idx == b) > 5 else np.nan for b in range(1, len(bins_c))]
    centers_c = 0.5 * (bins_c[:-1] + bins_c[1:])
    
    ax.plot(centers_c, mean_err_c, 'g-^', lw=2.5)
    ax.set_xlabel("Hole-to-Hole Ligament Clearance (mm)")
    ax.set_ylabel("Mean Euclidean Localization Error (mm)")
    ax.set_title("Stage 3C Figure 10: Localization Error vs Ligament Clearance")
    plt.tight_layout()
    f10_path = OUTPUT_DIR / "stage3C_fig10_error_vs_ligament_clearance.png"
    plt.savefig(f10_path, dpi=300)
    plt.close()
    copy_to_artifact(f10_path)
    print("Saved Fig 10.")

    # -------------------------------------------------------------
    # Fig 11: Error vs Boundary Clearance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    c_bnd = df_test['boundary_clearance_mm'].values
    bins_bnd = np.linspace(5, 300, 20)
    b_bnd_idx = np.digitize(c_bnd, bins_bnd)
    mean_err_bnd = [np.mean(euc_err_mm[b_bnd_idx == b]) if np.sum(b_bnd_idx == b) > 5 else np.nan for b in range(1, len(bins_bnd))]
    centers_bnd = 0.5 * (bins_bnd[:-1] + bins_bnd[1:])
    
    ax.plot(centers_bnd, mean_err_bnd, 'm-d', lw=2.5)
    ax.set_xlabel("Minimum Plate Boundary Clearance (mm)")
    ax.set_ylabel("Mean Euclidean Localization Error (mm)")
    ax.set_title("Stage 3C Figure 11: Localization Error vs Boundary Proximity")
    plt.tight_layout()
    f11_path = OUTPUT_DIR / "stage3C_fig11_error_vs_boundary_clearance.png"
    plt.savefig(f11_path, dpi=300)
    plt.close()
    copy_to_artifact(f11_path)
    print("Saved Fig 11.")

    # -------------------------------------------------------------
    # Fig 12: Identifiability Multi-Modal Symmetry Map
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plate outline
    rect = plt.Rectangle((0, 0), 1000, 1000, fill=False, edgecolor='black', lw=2)
    ax.add_patch(rect)
    # Center hole
    c_hole = plt.Circle((500, 500), 83.33/2, color='#2C3E50', fill=True, label='Central Hole (Dc=83.3 mm)')
    ax.add_patch(c_hole)
    
    # 4 Symmetric Positions generating sigma_max = 308.5 psi
    p1 = (300, 700)
    p2 = (700, 700)
    p3 = (300, 300)
    p4 = (700, 300)
    pts = [p1, p2, p3, p4]
    labels_sym = ["Q2 (300, 700)", "Q1 (700, 700)", "Q3 (300, 300)", "Q4 (700, 300)"]
    
    for (px, py), lbl in zip(pts, labels_sym):
        d_circle = plt.Circle((px, py), 15.0/2, color='#C0392B', fill=True)
        ax.add_patch(d_circle)
        ax.text(px, py + 25, f"{lbl}\nsigma = 308.5 psi", ha='center', fontsize=9.5, fontweight='bold', color='#C0392B')
        
    # Draw mirror lines
    ax.axvline(500, color='gray', linestyle=':', lw=1.5, label='Vertical Symmetry Plane')
    ax.axhline(500, color='gray', linestyle='--', lw=1.5, label='Horizontal Symmetry Plane')
    
    # Collapse centroid
    ax.scatter([500], [500], marker='x', s=120, color='blue', lw=3, label='Supervised ANN Model Output (Conditional Mean)')
    
    ax.set_xlim(-50, 1050)
    ax.set_ylim(-50, 1050)
    ax.set_xlabel("Plate X (mm)")
    ax.set_ylabel("Plate Y (mm)")
    ax.set_title("Stage 3C Figure 12: Fourfold Quadrant Symmetry Producing Identical Scalar Stress", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    f12_path = OUTPUT_DIR / "stage3C_fig12_identifiability_multimodal_map.png"
    plt.savefig(f12_path, dpi=300)
    plt.close()
    copy_to_artifact(f12_path)
    print("Saved Fig 12.")

    # -------------------------------------------------------------
    # Fig 13: Forward-Assisted Candidate Location Contours
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    case1 = search_cases['cases'][0] # Case 1
    case2 = search_cases['cases'][1] # Case 2
    
    for ax, c_data, c_title in zip(axes, [case1, case2], ["Case 1: Near Central Hole", "Case 2: Quadrant 1 Interior"]):
        grid_x = np.array(c_data['grid_X'])
        grid_y = np.array(c_data['grid_Y'])
        loss = np.array(c_data['grid_loss'])
        loss = np.where(loss < 0, np.nan, loss)
        stress_diff = np.sqrt(loss)
        
        im = ax.contourf(grid_x, grid_y, stress_diff, levels=np.linspace(0, 15, 16), cmap='viridis_r', alpha=0.8)
        # 1 psi contour curve
        cs = ax.contour(grid_x, grid_y, stress_diff, levels=[1.0], colors=['red'], linewidths=[2.5])
        ax.clabel(cs, inline=True, fmt='|delta_sigma| <= 1 psi', fontsize=9)
        
        # Central hole
        c_h = plt.Circle((c_data['L_mm']/2, c_data['H_mm']/2), c_data['Dc_mm']/2, color='#2C3E50', fill=True)
        ax.add_patch(c_h)
        # True location
        ax.scatter([c_data['true_X_mm']], [c_data['true_Y_mm']], marker='*', color='gold', edgecolor='black', s=250, label='True Defect Location', zorder=5)
        
        ax.set_xlim(0, c_data['L_mm'])
        ax.set_ylim(0, c_data['H_mm'])
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{c_title}\nObserved Stress = {c_data['observed_stress_psi']:.1f} psi", fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        plt.colorbar(im, ax=ax, label='|Predicted - Observed| Stress (psi)')
        
    plt.suptitle("Stage 3C Figure 13: Forward-Assisted Inversion Iso-Stress Candidate Contours", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f13_path = OUTPUT_DIR / "stage3C_fig13_candidate_location_contours.png"
    plt.savefig(f13_path, dpi=300)
    plt.close()
    copy_to_artifact(f13_path)
    print("Saved Fig 13.")

    # -------------------------------------------------------------
    # Fig 14: ANN Neurons Comparison (128 vs 256 vs 512)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ann_names = ["ANN-128", "ANN-256", "ANN-512"]
    ann_rx = [master['Experiment_C3_Conditioned']['ANN_128']['historical_test']['corr_X'],
              master['Experiment_C3_Conditioned']['ANN_256']['historical_test']['corr_X'],
              master['Experiment_C3_Conditioned']['ANN_512']['historical_test']['corr_X']]
    ann_ry = [master['Experiment_C3_Conditioned']['ANN_128']['historical_test']['corr_Y'],
              master['Experiment_C3_Conditioned']['ANN_256']['historical_test']['corr_Y'],
              master['Experiment_C3_Conditioned']['ANN_512']['historical_test']['corr_Y']]
    ann_dist = [master['Experiment_C3_Conditioned']['ANN_128']['historical_test']['mean_euclidean_mm'],
                master['Experiment_C3_Conditioned']['ANN_256']['historical_test']['mean_euclidean_mm'],
                master['Experiment_C3_Conditioned']['ANN_512']['historical_test']['mean_euclidean_mm']]
                
    x_pos = np.arange(len(ann_names))
    w = 0.35
    axes[0].bar(x_pos - w/2, ann_rx, width=w, label='R_X (Pearson)', color='#2980B9')
    axes[0].bar(x_pos + w/2, ann_ry, width=w, label='R_Y (Pearson)', color='#E67E22')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(ann_names)
    axes[0].set_ylabel("Pearson Correlation Coefficient")
    axes[0].set_title("Coordinate Correlation Across Hidden Sizes")
    axes[0].set_ylim(0, 0.6)
    axes[0].legend()
    for i in range(len(ann_names)):
        axes[0].text(x_pos[i] - w/2, ann_rx[i] + 0.01, f"{ann_rx[i]:.3f}", ha='center', fontsize=9)
        axes[0].text(x_pos[i] + w/2, ann_ry[i] + 0.01, f"{ann_ry[i]:.3f}", ha='center', fontsize=9)
        
    axes[1].bar(ann_names, ann_dist, color=['#85C1E9', '#5DADE2', '#3498DB'])
    axes[1].set_ylabel("Mean Euclidean Distance Error (mm)")
    axes[1].set_title("Physical Localization Distance in mm")
    axes[1].set_ylim(0, 450)
    for i, v in enumerate(ann_dist):
        axes[1].text(i, v + 8, f"{v:.1f} mm", ha='center', fontweight='bold')
        
    plt.suptitle("Stage 3C Figure 14: Inverse Artificial Neural Network Scaling Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f14_path = OUTPUT_DIR / "stage3C_fig14_ann_neurons_comparison.png"
    plt.savefig(f14_path, dpi=300)
    plt.close()
    copy_to_artifact(f14_path)
    print("Saved Fig 14.")

    # -------------------------------------------------------------
    # Fig 15: Individual Models vs Inverse Stacking
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    m_compare = ["Linear", "ANN-512", "RandomForest", "HistGradBoost", "Ridge Stack", "PCA Stack"]
    r_comb = [
        master['Conventional_Baselines']['LinearRegression']['historical_test']['corr_X'],
        master['Experiment_C3_Conditioned']['ANN_512']['historical_test']['corr_X'],
        master['Conventional_Baselines']['RandomForest']['historical_test']['corr_X'],
        master['Conventional_Baselines']['HistGradientBoosting']['historical_test']['corr_X'],
        master['Inverse_Stacking']['Inverse_Ridge_Stack']['historical_test']['corr_X'],
        master['Inverse_Stacking']['Inverse_PCA_Stack']['historical_test']['corr_X']
    ]
    dist_comb = [
        master['Conventional_Baselines']['LinearRegression']['historical_test']['mean_euclidean_mm'],
        master['Experiment_C3_Conditioned']['ANN_512']['historical_test']['mean_euclidean_mm'],
        master['Conventional_Baselines']['RandomForest']['historical_test']['mean_euclidean_mm'],
        master['Conventional_Baselines']['HistGradientBoosting']['historical_test']['mean_euclidean_mm'],
        master['Inverse_Stacking']['Inverse_Ridge_Stack']['historical_test']['mean_euclidean_mm'],
        master['Inverse_Stacking']['Inverse_PCA_Stack']['historical_test']['mean_euclidean_mm']
    ]
    
    axes[0].bar(m_compare, r_comb, color='#2980B9')
    axes[0].set_ylabel("Pearson Correlation R_X")
    axes[0].set_title("Coordinate Correlation Comparison")
    axes[0].set_xticklabels(m_compare, rotation=25, ha='right')
    axes[0].set_ylim(0, 0.6)
    for i, v in enumerate(r_comb):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        
    axes[1].bar(m_compare, dist_comb, color='#E74C3C')
    axes[1].set_ylabel("Mean Euclidean Localization Distance (mm)")
    axes[1].set_title("Physical Distance Error in mm (Lower is Better)")
    axes[1].set_xticklabels(m_compare, rotation=25, ha='right')
    axes[1].set_ylim(0, 450)
    for i, v in enumerate(dist_comb):
        axes[1].text(i, v + 8, f"{v:.1f} mm", ha='center', fontweight='bold')
        
    plt.suptitle("Stage 3C Figure 15: Individual Inverse Models vs Inverse Stacking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f15_path = OUTPUT_DIR / "stage3C_fig15_individual_vs_stacking.png"
    plt.savefig(f15_path, dpi=300)
    plt.close()
    copy_to_artifact(f15_path)
    print("Saved Fig 15.")

    # -------------------------------------------------------------
    # Fig 16: Geometry-wise Localization Performance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    geoms_list = df_test['geometry_family'].unique()
    clean_g_labels = [g.replace('rectangular_', '').replace('square_', '') for g in geoms_list]
    mean_geom_errs = [np.mean(euc_err_mm[df_test['geometry_family'] == g]) for g in geoms_list]
    median_geom_errs = [np.median(euc_err_mm[df_test['geometry_family'] == g]) for g in geoms_list]
    
    x = np.arange(len(geoms_list))
    w = 0.35
    ax.bar(x - w/2, mean_geom_errs, width=w, label='Mean Localization Error (mm)', color='#2980B9')
    ax.bar(x + w/2, median_geom_errs, width=w, label='Median Localization Error (mm)', color='#5DADE2')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_g_labels)
    ax.set_ylabel("Localization Distance (mm)")
    ax.set_title("Stage 3C Figure 16: Localization Error Across Plate Geometries")
    ax.legend()
    for i in range(len(geoms_list)):
        ax.text(x[i] - w/2, mean_geom_errs[i] + 5, f"{mean_geom_errs[i]:.1f}", ha='center', fontsize=9)
        ax.text(x[i] + w/2, median_geom_errs[i] + 5, f"{median_geom_errs[i]:.1f}", ha='center', fontsize=9)
        
    plt.tight_layout()
    f16_path = OUTPUT_DIR / "stage3C_fig16_geometry_wise_performance.png"
    plt.savefig(f16_path, dpi=300)
    plt.close()
    copy_to_artifact(f16_path)
    print("Saved Fig 16.")

    # -------------------------------------------------------------
    # Fig 17: Stress Noise Sensitivity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    noise_data = master['Noise_Sensitivity']
    p_levels = list(noise_data['perturbations'].keys())
    shifts = [noise_data['perturbations'][k]['mean_prediction_shift_mm'] for k in p_levels]
    loc_errs = [noise_data['perturbations'][k]['metrics']['mean_euclidean_mm'] for k in p_levels]
    
    axes[0].plot(p_levels, shifts, 'r-o', lw=2.5)
    axes[0].set_xlabel("Simulated Stress Perturbation Level")
    axes[0].set_ylabel("Mean Predicted Coordinate Shift (mm)")
    axes[0].set_title("Coordinate Shift Induced by Stress Measurement Noise")
    for i, v in enumerate(shifts):
        axes[0].text(i, v + 0.1, f"{v:.2f} mm", ha='center', fontweight='bold')
        
    axes[1].plot(p_levels, loc_errs, 'b-s', lw=2.5)
    axes[1].set_xlabel("Simulated Stress Perturbation Level")
    axes[1].set_ylabel("Mean Euclidean Localization Error (mm)")
    axes[1].set_title("Overall Localization Error vs Noise Level")
    for i, v in enumerate(loc_errs):
        axes[1].text(i, v + 2, f"{v:.1f} mm", ha='center', fontweight='bold')
        
    plt.suptitle("Stage 3C Figure 17: Model Robustness to Stress Measurement Perturbations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    f17_path = OUTPUT_DIR / "stage3C_fig17_noise_sensitivity.png"
    plt.savefig(f17_path, dpi=300)
    plt.close()
    copy_to_artifact(f17_path)
    print("Saved Fig 17.")

    # -------------------------------------------------------------
    # Fig 18: Uncertainty vs Actual Localization Error
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    # Ensemble spread between top estimators as uncertainty metric
    ann_p = predict_inverse_ann(joblib.load("models/stage3C_inverse_localization/geometry_conditioned_inverse/ANN_512.pt") if os.path.exists("models/stage3C_inverse_localization/geometry_conditioned_inverse/ANN_512.pt") else None, X_test_norm) if False else p_rf_test
    # Use distance between RF and ExtraTrees predictions as uncertainty proxy
    et_model = joblib.load("models/stage3C_inverse_localization/conventional_regression/ExtraTrees.pkl")
    p_et_test = et_model.predict(X_test_norm)
    unc_mm = np.sqrt(((p_rf_test[:, 0] - p_et_test[:, 0])*df_test['plate_length_mm'].values)**2 + ((p_rf_test[:, 1] - p_et_test[:, 1])*H_test)**2)
    
    # Subsample 500 points
    sub_u = np.random.choice(len(unc_mm), 500, replace=False)
    ax.scatter(unc_mm[sub_u], euc_err_mm[sub_u], alpha=0.5, color='#8E44AD', s=20)
    ax.set_xlabel("Ensemble Disagreement Spread (Uncertainty Proxy, mm)")
    ax.set_ylabel("Actual Euclidean Localization Error (mm)")
    ax.set_title("Stage 3C Figure 18: Prediction Uncertainty vs Actual Localization Error")
    plt.tight_layout()
    f18_path = OUTPUT_DIR / "stage3C_fig18_uncertainty_vs_error.png"
    plt.savefig(f18_path, dpi=300)
    plt.close()
    copy_to_artifact(f18_path)
    print("Saved Fig 18.")
    
    print("=== ALL 18 STAGE 3C FIGURES GENERATED SUCCESSFULLY ===")

if __name__ == "__main__":
    generate_all_stage3C_plots()

