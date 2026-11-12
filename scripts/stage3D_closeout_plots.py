"""
Plotting script for Stage 3D Close-Out Audit.
Generates publication-quality audit figures:
1. audit_fig01_logo_attribution_comparison.png
2. audit_fig02_uncertainty_calibration_coverage.png
3. audit_fig03_noise_and_degradation_spectrum.png
4. audit_fig04_anti_leakage_negative_controls.png
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output/stage3D_closeout"
AUDIT_MODEL_DIR = "models/stage3D_closeout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_all_closeout_figures():
    # Load audit files
    logo_file = os.path.join(AUDIT_MODEL_DIR, "d5_logo_audit.json")
    robust_file = os.path.join(AUDIT_MODEL_DIR, "d5_robustness_audit.json")
    leak_file = os.path.join(AUDIT_MODEL_DIR, "anti_leakage_audit.json")
    
    if not (os.path.exists(logo_file) and os.path.exists(robust_file) and os.path.exists(leak_file)):
        print("Audit json files not yet complete. Skipping plot generation until files exist.")
        return
        
    with open(logo_file, 'r') as f:
        logo_data = json.load(f)
    with open(robust_file, 'r') as f:
        robust_data = json.load(f)
    with open(leak_file, 'r') as f:
        leak_data = json.load(f)
        
    geoms = ['square_1000x1000', 'rectangular_1000x900', 'rectangular_1000x800', 'rectangular_1000x700', 'rectangular_1000x600']
    geom_labels = ['1000x1000\n(AR=1.0)', '1000x900\n(AR=1.11)', '1000x800\n(AR=1.25)', '1000x700\n(AR=1.43)', '1000x600\n(AR=1.67)']
    
    # ----------------------------------------------------
    # FIG 1: LOGO Attribution Comparison (D1 vs True D5)
    # ----------------------------------------------------
    # D1 reported errors from Phase J
    d1_logo_errors = [190.96, 128.21, 163.53, 175.82, 179.37]
    d5_logo_errors = [logo_data[g]['Mean_Euclidean_mm'] for g in geoms]
    d5_logo_medians = [logo_data[g]['Median_Euclidean_mm'] for g in geoms]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(geoms))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, d1_logo_errors, width, label='Model D1 (Regional ExtraTrees - Misattributed in Phase J)', color='#d9534f', alpha=0.85, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, d5_logo_errors, width, label='Model D5 (True Spatial U-Net Heatmap - Audited)', color='#2e6f40', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f} mm', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#a94442')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f} mm', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2e6f40')
        
    ax.set_ylabel('Mean Euclidean Localization Error (mm)', fontsize=11, fontweight='bold')
    ax.set_title('Figure Audit-1: Leave-One-Geometry-Out (LOGO) Model Attribution Resolution\nRegional ExtraTrees (Phase J) vs Audited Spatial U-Net Heatmap Model D5', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(geom_labels, fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 220)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audit_fig01_logo_attribution_comparison.png"))
    plt.close()
    print("Saved audit_fig01_logo_attribution_comparison.png")
    
    # ----------------------------------------------------
    # FIG 2: Spatial Uncertainty Calibration
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    nominal_levels = np.array([50.0, 68.27, 90.0, 95.0, 99.0])
    # Average across geometries in LOGO
    emp_cov = np.array([
        np.mean([logo_data[g]['Empirical_Coverage_50pct'] for g in geoms]),
        np.mean([logo_data[g]['Empirical_Coverage_68pct'] for g in geoms]),
        np.mean([logo_data[g]['Empirical_Coverage_90pct'] for g in geoms]),
        np.mean([logo_data[g]['Empirical_Coverage_95pct'] for g in geoms]),
        np.mean([logo_data[g]['Empirical_Coverage_99pct'] for g in geoms])
    ])
    
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='Ideal Calibration (Empirical = Nominal)')
    ax.plot(nominal_levels, emp_cov, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label=f'Model D5 Chi-Square Mahalanobis Coverage (Mean r = {np.mean([logo_data[g]["Correlation_Uncertainty_vs_Error"] for g in geoms]):.2f})')
    
    for nom, emp in zip(nominal_levels, emp_cov):
        ax.annotate(f'{emp:.1f}%', xy=(nom, emp), xytext=(-10, 8), textcoords="offset points", fontsize=9, fontweight='bold', color='#1f77b4')
        
    ax.set_xlabel('Nominal Confidence Level (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Empirical Coverage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Figure Audit-2: Spatial Uncertainty Calibration on Held-Out Geometries\nMahalanobis Ellipse Coverage vs Nominal $\chi_2^2$ Confidence Intervals', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlim(40, 105)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audit_fig02_uncertainty_calibration_coverage.png"))
    plt.close()
    print("Saved audit_fig02_uncertainty_calibration_coverage.png")
    
    # ----------------------------------------------------
    # FIG 3: Model D5 Noise & Missing Field Robustness
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    cond_names = list(robust_data.keys())
    clean_mean = robust_data['Clean_Reference']['Mean_Euclidean_mm']
    cond_labels = [c.replace('_', ' ') for c in cond_names]
    means = [robust_data[c]['Mean_Euclidean_mm'] for c in cond_names]
    medians = [robust_data[c]['Median_Euclidean_mm'] for c in cond_names]
    
    x = np.arange(len(cond_names))
    width = 0.35
    ax.bar(x - width/2, means, width, label='Mean Euclidean Error (mm)', color='#4a7c59', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x + width/2, medians, width, label='Median Euclidean Error (mm)', color='#8fb996', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.axhline(clean_mean, color='red', linestyle='--', linewidth=1.2, label=f'Clean Baseline Mean ({clean_mean:.2f} mm)')
    
    for i, m in enumerate(means):
        ax.annotate(f'{m:.1f}', xy=(x[i] - width/2, m), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.set_ylabel('Localization Error (mm)', fontsize=11, fontweight='bold')
    ax.set_title('Figure Audit-3: Model D5 U-Net Robustness Spectrum Across Signal Degradations', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha='right', fontsize=9)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audit_fig03_noise_and_degradation_spectrum.png"))
    plt.close()
    print("Saved audit_fig03_noise_and_degradation_spectrum.png")
    
    # ----------------------------------------------------
    # FIG 4: Anti-Leakage Negative Controls
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ctrl_names = ['Audited D5 U-Net\n(Full Stress Field)', 'Mask-Only Baseline\n(Zero Stress Input)', 'Shuffled Field Control\n(Permuted Stress)', 'IDW Continuum Inpainting\n(Ablation)']
    ctrl_errors = [
        robust_data['Clean_Reference']['Mean_Euclidean_mm'],
        leak_data['Mask_Only_Control']['Mean_Euclidean_mm'],
        leak_data['Shuffled_Field_Control']['Mean_Euclidean_mm'],
        leak_data['Inpainting_IDW_200']['Mean_Euclidean_mm']
    ]
    colors = ['#2e6f40', '#d9534f', '#e67e22', '#2980b9']
    bars = ax.bar(ctrl_names, ctrl_errors, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8, width=0.5)
    
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.2f} mm', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_ylabel('Mean Euclidean Localization Error (mm)', fontsize=11, fontweight='bold')
    ax.set_title('Figure Audit-4: Anti-Leakage Diagnostic and Negative Control Verification\nProving Information Inversion Comes from Non-Local Stress Fields', fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, max(ctrl_errors) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audit_fig04_anti_leakage_negative_controls.png"))
    plt.close()
    print("Saved audit_fig04_anti_leakage_negative_controls.png")
    print("All closeout figures generated successfully.")

if __name__ == '__main__':
    plot_all_closeout_figures()
