"""
Stage 2V Phase C - Adaptive Candidate Plotting
Generates:
1. stage2V_adaptive_candidate_locations.png (Locations of Strategy A, B, C candidates)
2. stage2V_original_vs_enriched_coverage.png (Coverage expansion)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_candidate_plots():
    print("Generating Stage 2V Adaptive Candidate Plots...")
    cands_path = 'data/stage2V_adaptive_selection/master_adaptive_candidates_1250.csv'
    df_uni_path = 'data/unified_dataset_50375.csv'
    
    if not os.path.exists(cands_path) or not os.path.exists(df_uni_path):
        print("Required CSV files missing.")
        return
        
    df_cands = pd.read_csv(cands_path)
    df_uni = pd.read_csv(df_uni_path)
    
    os.makedirs('output', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # 1. Candidate Locations by Acquisition Strategy across the 5 geometries
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharey=True)
    geom_order = ['rectangular_1000x600', 'rectangular_1000x700', 'rectangular_1000x800', 'rectangular_1000x900', 'square_1000x1000']
    
    colors = {
        'Strategy_A_Model_Disagreement': '#e41a1c',
        'Strategy_B_High_Error_Neighborhood': '#377eb8',
        'Strategy_C_Physical_Risk_Coverage': '#4daf4a'
    }
    strategy_labels = {
        'Strategy_A_Model_Disagreement': 'Strategy A: Disagreement (150)',
        'Strategy_B_High_Error_Neighborhood': 'Strategy B: Error Nbrhd (60)',
        'Strategy_C_Physical_Risk_Coverage': 'Strategy C: Physical Risk (40)'
    }
    
    for ax, g in zip(axes, geom_order):
        sub = df_cands[df_cands['geometry_family'] == g]
        
        # Central hole circle in normalized space
        # center is (0.5, 0.5), radius rc/L = (H/24)/L
        h_val = sub['plate_height_mm'].iloc[0]
        l_val = sub['plate_length_mm'].iloc[0]
        rc = (h_val / 24.0)
        circle_norm = plt.Circle((0.5, 0.5), rc / l_val, color='gray', alpha=0.35, label='Central Hole')
        ax.add_patch(circle_norm)
        
        for strat, color in colors.items():
            s_data = sub[sub['acquisition_strategy'] == strat]
            ax.scatter(
                s_data['defect_x_over_L'], 
                s_data['defect_y_over_H'], 
                c=color, 
                s=35, 
                alpha=0.85, 
                label=strategy_labels[strat] if ax == axes[0] else ""
            )
            
        ax.set_title(f"{g}\n$N = 250$ Adaptive Points", fontsize=11, fontweight='bold')
        ax.set_xlabel('$X / L$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel('$Y / H$')
            ax.legend(loc='lower center', bbox_to_anchor=(2.5, -0.32), ncol=4, fontsize=11)
            
    plt.suptitle('Stage 2V Adaptive Candidate Acquisition Locations by Strategy (1,250 Total)', fontsize=15, fontweight='bold', y=1.03)
    plt.savefig('output/stage2V_adaptive_candidate_locations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved output/stage2V_adaptive_candidate_locations.png")
    
    # 2. Original vs Enriched Dataset Coverage Comparison
    fig, axes = plt.subplots(2, 5, figsize=(24, 9), sharex=True, sharey=True)
    
    for i, g in enumerate(geom_order):
        # Top row: original 10,075
        ax_top = axes[0, i]
        sub_orig = df_uni[df_uni['geometry_family'] == g]
        ax_top.scatter(sub_orig['normalized_defect_x'], sub_orig['normalized_defect_y'], c='#4575b4', s=1.5, alpha=0.25)
        ax_top.set_title(f"Baseline {g}\n(Original $N = 10,075$)", fontsize=11, fontweight='bold')
        if i == 0:
            ax_top.set_ylabel('Baseline $Y / H$')
            
        # Bottom row: enriched (original + adaptive)
        ax_bot = axes[1, i]
        sub_adapt = df_cands[df_cands['geometry_family'] == g]
        ax_bot.scatter(sub_orig['normalized_defect_x'], sub_orig['normalized_defect_y'], c='#4575b4', s=1.5, alpha=0.15, label='Original')
        ax_bot.scatter(sub_adapt['defect_x_over_L'], sub_adapt['defect_y_over_H'], c='#d73027', s=20, alpha=0.85, marker='x', label='Stage 2V Adaptive')
        ax_bot.set_title(f"Enriched {g}\n($N = 10,325$, +250 Adaptive)", fontsize=11, fontweight='bold')
        ax_bot.set_xlabel('$X / L$')
        if i == 0:
            ax_bot.set_ylabel('Enriched $Y / H$')
            ax_bot.legend(loc='lower right', fontsize=9)
            
    plt.suptitle('Parameter Space Density: Original (50,375) vs Adaptively Enriched (51,625)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('output/stage2V_original_vs_enriched_coverage.png', dpi=300)
    plt.close()
    print("Saved output/stage2V_original_vs_enriched_coverage.png")

if __name__ == '__main__':
    generate_candidate_plots()

