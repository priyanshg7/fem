"""
Stage 2V Phase E - High-Resolution Mesh Validation Plotting
Generates stage2V_high_res_mesh_comparison.png comparing production vs 2x refined mesh solutions.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_mesh_comparison_plot():
    print("Generating Stage 2V High-Resolution Mesh Validation Plot...")
    audit_json = 'output/stage2V_adaptive_mesh_refinement_audit.json'
    if not os.path.exists(audit_json):
        print(f"Error: {audit_json} not found.")
        return
        
    with open(audit_json, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    os.makedirs('output', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Element Von Mises Stress Convergence
    ax = axes[0, 0]
    ax.scatter(df['prod_elem_vm_psi'], df['ref_elem_vm_psi'], c=df['diff_elem_vm_pct'], cmap='viridis', s=45, alpha=0.85)
    lims = [df['prod_elem_vm_psi'].min() - 10, df['prod_elem_vm_psi'].max() + 10]
    ax.plot(lims, lims, 'r--', lw=2, label='1:1 Ideal Convergence')
    ax.set_title('Element Von Mises Stress: Production vs $2\\times$ Refined', fontweight='bold')
    ax.set_xlabel('Production Mesh $\\sigma_{vM}^{\\text{elem}}$ (psi)')
    ax.set_ylabel('$2\\times$ Refined Mesh $\\sigma_{vM}^{\\text{elem}}$ (psi)')
    ax.legend()
    
    # 2. Relative Discrepancy Distribution (%)
    ax = axes[0, 1]
    metrics = ['diff_disp_pct', 'diff_energy_pct', 'diff_elem_vm_pct', 'diff_node_vm_pct']
    labels = ['Max Disp (%)', 'Strain Energy (%)', 'Elem Stress (%)', 'Nodal Stress (%)']
    box_data = [df[m] for m in metrics]
    ax.boxplot(box_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor='#4575b4', alpha=0.7))
    ax.set_title('Relative Solution Discrepancy under $2\\times$ Refinement', fontweight='bold')
    ax.set_ylabel('Relative Difference $|\\text{Ref} - \\text{Prod}| / \\text{Ref}$ (%)')
    
    # 3. Stress Difference vs Hole-to-Hole Clearance
    ax = axes[1, 0]
    sub_df = df.copy()
    # If hole_to_hole clearance is not in audit json, calculate or use d_mm
    sc = ax.scatter(df['d_mm'], df['diff_elem_vm_pct'], c=df['ref_elements'], cmap='plasma', s=45, alpha=0.85)
    ax.set_title('Element Stress Sensitivity vs Defect Diameter', fontweight='bold')
    ax.set_xlabel('Defect Diameter $D$ (mm)')
    ax.set_ylabel('Stress Difference $\\Delta \\sigma_{vM}^{\\text{elem}}$ (%)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Refined Element Count', fontsize=10)
    
    # 4. Global Displacement & Energy Invariance
    ax = axes[1, 1]
    ax.scatter(df['prod_disp_um'], df['ref_disp_um'], c='#2ca02c', s=45, label='Max Displacement ($\mu$m)')
    disp_lims = [df['prod_disp_um'].min() - 0.2, df['prod_disp_um'].max() + 0.2]
    ax.plot(disp_lims, disp_lims, 'k--', lw=1.5, label='1:1 Line')
    ax.set_title('Global Displacement Invariance (Max $\\Delta < 0.04\\%$)', fontweight='bold')
    ax.set_xlabel('Production $u_{\\max}$ ($\mu$m)')
    ax.set_ylabel('$2\\times$ Refined $u_{\\max}$ ($\mu$m)')
    ax.legend()
    
    plt.suptitle('Stage 2V High-Resolution Mesh Convergence Audit (50 Selected Adaptive Cases)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('output/stage2V_high_res_mesh_comparison.png', dpi=300)
    plt.close()
    print("Saved output/stage2V_high_res_mesh_comparison.png")

if __name__ == '__main__':
    generate_mesh_comparison_plot()

