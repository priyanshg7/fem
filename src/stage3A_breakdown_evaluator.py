"""
Stage 3A Sections 9, 10, 11 - Detailed Performance Breakdown and Benchmarking
Evaluates test set predictions by geometry family, defect diameter, clearance,
stress regime, and compares Original vs Enriched ANN models against Stage 2V baselines.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run_performance_breakdowns():
    print("==================================================================")
    print("Starting Stage 3A Performance Breakdown and Benchmarking")
    print("==================================================================")
    
    eval_csv = 'models/stage3A_forward_ann/evaluation/master_evaluation_table.csv'
    if not os.path.exists(eval_csv):
        print(f"Error: {eval_csv} not found.")
        return
        
    df = pd.read_csv(eval_csv)
    print(f"Loaded {len(df)} evaluation records.")
    
    geometries = [
        'rectangular_1000x600',
        'rectangular_1000x700',
        'rectangular_1000x800',
        'rectangular_1000x900',
        'square_1000x1000'
    ]
    architectures = ['ANN_128', 'ANN_256', 'ANN_512']
    experiments = ['original_dataset', 'enriched_dataset']
    
    # 1. Geometry-Specific Performance
    geom_breakdown = {}
    for exp in experiments:
        geom_breakdown[exp] = {}
        for arch in architectures:
            geom_breakdown[exp][arch] = {}
            for g in geometries:
                sub = df[(df['experiment'] == exp) & (df['architecture'] == arch) & (df['geometry_family'] == g)]
                
                # Compute across all seeds
                r2 = r2_score(sub['y_true_kt'], sub['y_pred_kt'])
                mae_psi = mean_absolute_error(sub['y_true_kt'] * 100.0, sub['y_pred_kt'] * 100.0)
                rmse_psi = np.sqrt(mean_squared_error(sub['y_true_kt'] * 100.0, sub['y_pred_kt'] * 100.0))
                
                geom_breakdown[exp][arch][g] = {
                    'r2': float(r2),
                    'mae_psi': float(mae_psi),
                    'rmse_psi': float(rmse_psi),
                    'samples': len(sub)
                }
                
    # 2. Physical Regime Breakdown
    regime_breakdown = {}
    for exp in experiments:
        regime_breakdown[exp] = {}
        for arch in architectures:
            sub_all = df[(df['experiment'] == exp) & (df['architecture'] == arch)]
            
            # Stress regime: Kt <= 3.20 vs Kt > 3.20
            mod_mask = sub_all['y_true_kt'] <= 3.20
            high_mask = sub_all['y_true_kt'] > 3.20
            
            # Clearance: < 25 mm vs >= 25 mm
            tight_mask = sub_all['hole_to_hole_clearance_mm'] < 25.0
            wide_mask = sub_all['hole_to_hole_clearance_mm'] >= 25.0
            
            # Boundary clearance: < 25 mm vs >= 25 mm
            edge_mask = sub_all['boundary_clearance_mm'] < 25.0
            interior_mask = sub_all['boundary_clearance_mm'] >= 25.0
            
            # Defect diameter: small (< 0.12), medium (0.12 - 0.22), large (> 0.22)
            d_small_mask = sub_all['defect_d_over_Dc'] < 0.12
            d_med_mask = (sub_all['defect_d_over_Dc'] >= 0.12) & (sub_all['defect_d_over_Dc'] <= 0.22)
            d_large_mask = sub_all['defect_d_over_Dc'] > 0.22
            
            def calc_sub_metrics(m):
                s = sub_all[m]
                if len(s) == 0:
                    return {'mae_psi': 0.0, 'rmse_psi': 0.0, 'samples': 0}
                return {
                    'mae_psi': float(mean_absolute_error(s['y_true_kt'] * 100.0, s['y_pred_kt'] * 100.0)),
                    'rmse_psi': float(np.sqrt(mean_squared_error(s['y_true_kt'] * 100.0, s['y_pred_kt'] * 100.0))),
                    'samples': len(s)
                }
                
            regime_breakdown[exp][arch] = {
                'stress_moderate_le_3_20': calc_sub_metrics(mod_mask),
                'stress_high_gt_3_20': calc_sub_metrics(high_mask),
                'clearance_tight_lt_25mm': calc_sub_metrics(tight_mask),
                'clearance_wide_ge_25mm': calc_sub_metrics(wide_mask),
                'boundary_edge_lt_25mm': calc_sub_metrics(edge_mask),
                'boundary_interior_ge_25mm': calc_sub_metrics(interior_mask),
                'diameter_small_lt_0_12': calc_sub_metrics(d_small_mask),
                'diameter_medium_0_12_0_22': calc_sub_metrics(d_med_mask),
                'diameter_large_gt_0_22': calc_sub_metrics(d_large_mask)
            }
            
    # 3. Original vs Enriched Relative Improvements
    improvement_summary = {}
    for arch in architectures:
        orig_m = geom_breakdown['original_dataset'][arch]
        enr_m = geom_breakdown['enriched_dataset'][arch]
        
        orig_reg = regime_breakdown['original_dataset'][arch]
        enr_reg = regime_breakdown['enriched_dataset'][arch]
        
        # Overall on test
        sub_orig = df[(df['experiment'] == 'original_dataset') & (df['architecture'] == arch)]
        sub_enr = df[(df['experiment'] == 'enriched_dataset') & (df['architecture'] == arch)]
        
        orig_r2 = r2_score(sub_orig['y_true_kt'], sub_orig['y_pred_kt'])
        enr_r2 = r2_score(sub_enr['y_true_kt'], sub_enr['y_pred_kt'])
        
        orig_mae = mean_absolute_error(sub_orig['y_true_kt'] * 100.0, sub_orig['y_pred_kt'] * 100.0)
        enr_mae = mean_absolute_error(sub_enr['y_true_kt'] * 100.0, sub_enr['y_pred_kt'] * 100.0)
        
        orig_rmse = np.sqrt(mean_squared_error(sub_orig['y_true_kt'] * 100.0, sub_orig['y_pred_kt'] * 100.0))
        enr_rmse = np.sqrt(mean_squared_error(sub_enr['y_true_kt'] * 100.0, sub_enr['y_pred_kt'] * 100.0))
        
        improvement_summary[arch] = {
            'overall_r2_gain': float(enr_r2 - orig_r2),
            'overall_mae_reduction_pct': float((orig_mae - enr_mae) / orig_mae * 100.0),
            'overall_rmse_reduction_pct': float((orig_rmse - enr_rmse) / orig_rmse * 100.0),
            'high_stress_mae_reduction_pct': float(
                (orig_reg['stress_high_gt_3_20']['mae_psi'] - enr_reg['stress_high_gt_3_20']['mae_psi']) /
                orig_reg['stress_high_gt_3_20']['mae_psi'] * 100.0
            ),
            'tight_ligament_mae_reduction_pct': float(
                (orig_reg['clearance_tight_lt_25mm']['mae_psi'] - enr_reg['clearance_tight_lt_25mm']['mae_psi']) /
                orig_reg['clearance_tight_lt_25mm']['mae_psi'] * 100.0
            )
        }
        
    os.makedirs('output', exist_ok=True)
    with open('output/stage3A_geometry_performance.json', 'w') as f:
        json.dump(geom_breakdown, f, indent=2)
    with open('output/stage3A_regime_performance.json', 'w') as f:
        json.dump(regime_breakdown, f, indent=2)
    with open('output/stage3A_improvement_summary.json', 'w') as f:
        json.dump(improvement_summary, f, indent=2)
        
    print("Saved geometry, regime, and improvement summaries to output/")

if __name__ == '__main__':
    run_performance_breakdowns()

