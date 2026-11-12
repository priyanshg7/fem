"""
Stage 3C Phase B: Mathematical Formulation and Identifiability Audit
Analyzes the one-to-many non-uniqueness and spatial ambiguity of mapping scalar
maximum Von Mises stress back to 2D defect coordinates (X, Y).
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def run_identifiability_audit():
    print("=== RUNNING STAGE 3C IDENTIFIABILITY AUDIT ===")
    
    df = pd.read_csv("data/enriched_dataset_51625.csv")
    print(f"Loaded enriched dataset: {df.shape}")
    
    # Stress column in psi and Kt
    stress_col = 'max_von_mises_element_psi'
    df['sigma_vM'] = df[stress_col]
    df['Kt'] = df['Kt_element']
    
    # Coordinates normalized
    df['norm_X'] = df['defect_x_over_L']
    df['norm_Y'] = df['defect_y_over_H']
    df['dim_X'] = df['defect_center_x_mm']
    df['dim_Y'] = df['defect_center_y_mm']
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Scalar Stress Nearest-Neighbor Spatial Distance
    print("\n--- 1. Nearest Neighbor Analysis in Scalar Stress Space ---")
    # Sample a representative subset of 5,000 simulations for fast distance matrix
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=5000, replace=False)
    sub_df = df.iloc[sample_indices].copy().reset_index(drop=True)
    
    stresses = sub_df['sigma_vM'].values
    coords_dim = sub_df[['dim_X', 'dim_Y']].values
    coords_norm = sub_df[['norm_X', 'norm_Y']].values
    
    # Sort by stress
    sorted_idx = np.argsort(stresses)
    sorted_stresses = stresses[sorted_idx]
    sorted_coords_dim = coords_dim[sorted_idx]
    sorted_coords_norm = coords_norm[sorted_idx]
    
    # Compare each point with its immediate neighbor in stress space
    d_stress = np.abs(np.diff(sorted_stresses))
    d_spatial_dim = np.sqrt(np.sum(np.diff(sorted_coords_dim, axis=0)**2, axis=1))
    d_spatial_norm = np.sqrt(np.sum(np.diff(sorted_coords_norm, axis=0)**2, axis=1))
    
    # Filter neighbors with delta_stress < 0.05 psi
    tight_stress_mask = d_stress < 0.05
    mean_nn_dist_dim = float(np.mean(d_spatial_dim[tight_stress_mask]))
    median_nn_dist_dim = float(np.median(d_spatial_dim[tight_stress_mask]))
    max_nn_dist_dim = float(np.max(d_spatial_dim[tight_stress_mask]))
    pct_gt_100mm = float(np.mean(d_spatial_dim[tight_stress_mask] > 100.0) * 100.0)
    pct_gt_200mm = float(np.mean(d_spatial_dim[tight_stress_mask] > 200.0) * 100.0)
    
    print(f"When delta_stress < 0.05 psi (virtually identical stress):")
    print(f"  Mean Euclidean spatial distance: {mean_nn_dist_dim:.2f} mm")
    print(f"  Median Euclidean spatial distance: {median_nn_dist_dim:.2f} mm")
    print(f"  Maximum Euclidean spatial distance: {max_nn_dist_dim:.2f} mm")
    print(f"  Percentage of identical-stress pairs separated by > 100 mm: {pct_gt_100mm:.1f}%")
    print(f"  Percentage of identical-stress pairs separated by > 200 mm: {pct_gt_200mm:.1f}%")
    
    # 2. Iso-Stress Interval Spread Analysis across Specific Targets
    print("\n--- 2. Iso-Stress Interval Spread Analysis ---")
    stress_levels = [295.0, 300.0, 310.0, 325.0, 350.0, 400.0]
    tolerances = [0.5, 1.0, 2.0, 5.0]
    
    isostress_results = {}
    for s_target in stress_levels:
        isostress_results[str(s_target)] = {}
        for tol in tolerances:
            mask = np.abs(df['sigma_vM'] - s_target) <= tol
            matching = df[mask]
            count = len(matching)
            
            if count > 1:
                pts_norm = matching[['norm_X', 'norm_Y']].values
                pts_dim = matching[['dim_X', 'dim_Y']].values
                
                # Pairwise distances
                pw_dists_dim = cdist(pts_dim[:1000], pts_dim[:1000]) # cap at 1000 for speed
                triu_idx = np.triu_indices_from(pw_dists_dim, k=1)
                dists_upper = pw_dists_dim[triu_idx]
                
                mean_pw_dist = float(np.mean(dists_upper))
                median_pw_dist = float(np.median(dists_upper))
                max_pw_dist = float(np.max(dists_upper))
                std_X = float(matching['dim_X'].std())
                std_Y = float(matching['dim_Y'].std())
                
                # Symmetry quadrant counts (relative to plate center 500, H/2)
                Xc = matching['plate_length_mm'] / 2.0
                Yc = matching['plate_height_mm'] / 2.0
                q1 = int(np.sum((matching['dim_X'] >= Xc) & (matching['dim_Y'] >= Yc)))
                q2 = int(np.sum((matching['dim_X'] < Xc) & (matching['dim_Y'] >= Yc)))
                q3 = int(np.sum((matching['dim_X'] < Xc) & (matching['dim_Y'] < Yc)))
                q4 = int(np.sum((matching['dim_X'] >= Xc) & (matching['dim_Y'] < Yc)))
                
                isostress_results[str(s_target)][str(tol)] = {
                    'count': count,
                    'std_X_mm': std_X,
                    'std_Y_mm': std_Y,
                    'mean_pairwise_dist_mm': mean_pw_dist,
                    'median_pairwise_dist_mm': median_pw_dist,
                    'max_pairwise_dist_mm': max_pw_dist,
                    'quadrant_counts': {'Q1_TR': q1, 'Q2_TL': q2, 'Q3_BL': q3, 'Q4_BR': q4}
                }
            else:
                isostress_results[str(s_target)][str(tol)] = {'count': count}
                
    # 3. Fixed Defect Diameter vs Variable Defect Diameter Ambiguity
    print("\n--- 3. Fixed vs Variable Diameter Ambiguity ---")
    # Paper fixed ratio: c/a = 0.15, corresponding to defect_d_over_Dc ~ 0.15
    fixed_mask = np.abs(df['defect_d_over_Dc'] - 0.150) <= 0.010
    df_fixed = df[fixed_mask].copy()
    print(f"Fixed diameter subset (d/Dc in [0.14, 0.16]): {len(df_fixed)} simulations")
    
    # Compare spread at 300 psi for fixed vs variable
    mask_fix_300 = np.abs(df_fixed['sigma_vM'] - 300.0) <= 1.0
    pts_fix_dim = df_fixed.loc[mask_fix_300, ['dim_X', 'dim_Y']].values
    pw_fix = cdist(pts_fix_dim[:500], pts_fix_dim[:500])[np.triu_indices(min(len(pts_fix_dim), 500), k=1)]
    
    mask_var_300 = np.abs(df['sigma_vM'] - 300.0) <= 1.0
    pts_var_dim = df.loc[mask_var_300, ['dim_X', 'dim_Y']].values
    pw_var = cdist(pts_var_dim[:500], pts_var_dim[:500])[np.triu_indices(min(len(pts_var_dim), 500), k=1)]
    
    fixed_vs_var = {
        'fixed_diameter_sample_count_at_300psi': int(len(pts_fix_dim)),
        'fixed_diameter_mean_pw_dist_mm': float(np.mean(pw_fix)),
        'fixed_diameter_max_pw_dist_mm': float(np.max(pw_fix)),
        'variable_diameter_sample_count_at_300psi': int(len(pts_var_dim)),
        'variable_diameter_mean_pw_dist_mm': float(np.mean(pw_var)),
        'variable_diameter_max_pw_dist_mm': float(np.max(pw_var))
    }
    print(f"At 300 psi (+- 1 psi):")
    print(f"  Fixed diameter mean spatial separation: {fixed_vs_var['fixed_diameter_mean_pw_dist_mm']:.2f} mm (Max: {fixed_vs_var['fixed_diameter_max_pw_dist_mm']:.2f} mm)")
    print(f"  Variable diameter mean spatial separation: {fixed_vs_var['variable_diameter_mean_pw_dist_mm']:.2f} mm (Max: {fixed_vs_var['variable_diameter_max_pw_dist_mm']:.2f} mm)")

    # 4. Geometry-Specific Ambiguity
    print("\n--- 4. Geometry-Specific Ambiguity Breakdown ---")
    geom_ambiguity = {}
    for g_name in df['geometry_family'].unique():
        sub_g = df[df['geometry_family'] == g_name]
        mask_g = np.abs(sub_g['sigma_vM'] - 300.0) <= 1.0
        pts_g = sub_g.loc[mask_g, ['dim_X', 'dim_Y']].values
        if len(pts_g) > 1:
            pw_g = cdist(pts_g[:400], pts_g[:400])[np.triu_indices(min(len(pts_g), 400), k=1)]
            geom_ambiguity[g_name] = {
                'count_at_300psi': int(len(pts_g)),
                'mean_separation_mm': float(np.mean(pw_g)),
                'max_separation_mm': float(np.max(pw_g))
            }
        print(f"  {g_name}: {geom_ambiguity[g_name]['count_at_300psi']} configs at 300 psi, mean sep = {geom_ambiguity[g_name]['mean_separation_mm']:.2f} mm")

    audit_summary = {
        'nearest_neighbor_analysis': {
            'mean_nn_dist_mm': mean_nn_dist_dim,
            'median_nn_dist_mm': median_nn_dist_dim,
            'max_nn_dist_mm': max_nn_dist_dim,
            'pct_separated_gt_100mm': pct_gt_100mm,
            'pct_separated_gt_200mm': pct_gt_200mm
        },
        'isostress_intervals': isostress_results,
        'fixed_vs_variable_diameter': fixed_vs_var,
        'geometry_specific_ambiguity': geom_ambiguity
    }
    
    with open(output_dir / "stage3C_identifiability_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
    print("\nSaved output/stage3C_identifiability_audit.json successfully.")

if __name__ == "__main__":
    run_identifiability_audit()

