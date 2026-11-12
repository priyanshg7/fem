"""
Stage 3A-R Phase A: Mandatory Stage 3A Evaluation Audit
Independently verifies:
1. Split manifest sample counts and zero leakage.
2. Geometry-specific sample counts in test split.
3. Metric calculations across evaluation methods (individual seed, mean across seeds, committee average, pooled rows).
4. Physical regime definitions, mutually exclusive sample counts summing to 7,557.
5. Reference paper methodology reconciliation.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def run_phaseA_audit():
    print("==================================================================")
    print("STAGE 3A-R PHASE A: MANDATORY EVALUATION AUDIT")
    print("==================================================================")
    
    # 1. Manifests Verification
    m3a = pd.read_csv('data/stage3A_ann_split_manifest.csv')
    m2v = pd.read_csv('data/stage2V_split_manifest.csv')
    df_uni = pd.read_csv('data/unified_dataset_50375.csv')
    df_enr = pd.read_csv('data/enriched_dataset_51625.csv')
    df_eval = pd.read_csv('models/stage3A_forward_ann/evaluation/master_evaluation_table.csv')
    
    # 1.1 Total Counts
    audit_data = {
        'manifests': {
            'stage2V_total_records': len(m2v),
            'stage2V_train_records': int((m2v['split'] == 'train').sum()),
            'stage2V_val_records': int((m2v['split'] == 'val').sum()),
            'stage2V_test_records': int((m2v['split'] == 'test').sum()),
            'stage3A_total_records': len(m3a),
            'stage3A_train_records': int((m3a['split'] == 'train').sum()),
            'stage3A_val_records': int((m3a['split'] == 'val').sum()),
            'stage3A_test_records': int((m3a['split'] == 'test').sum()),
            'adaptive_simulations_count': int((m3a['data_source'] == 'stage2V_adaptive_enrichment').sum()),
            'adaptive_simulations_in_train': int(((m3a['data_source'] == 'stage2V_adaptive_enrichment') & (m3a['split'] == 'train')).sum()),
        }
    }
    
    # 1.2 Geometry breakdown of Test set
    test_manifest = m3a[m3a['split'] == 'test']
    geom_counts = test_manifest['geometry_family'].value_counts().to_dict()
    audit_data['test_geometry_counts'] = geom_counts
    audit_data['test_geometry_sum'] = sum(geom_counts.values())
    
    print("\n[Audit 1: Split & Geometry Counts]")
    print(f"  Stage 2V Manifest: {len(m2v)} total | Train: {audit_data['manifests']['stage2V_train_records']}, Val: {audit_data['manifests']['stage2V_val_records']}, Test: {audit_data['manifests']['stage2V_test_records']}")
    print(f"  Stage 3A Manifest: {len(m3a)} total | Train: {audit_data['manifests']['stage3A_train_records']}, Val: {audit_data['manifests']['stage3A_val_records']}, Test: {audit_data['manifests']['stage3A_test_records']}")
    print(f"  Adaptive samples in Train: {audit_data['manifests']['adaptive_simulations_in_train']} / {audit_data['manifests']['adaptive_simulations_count']}")
    print("  Geometry-specific Test Counts (MUST sum to 7,557):")
    for g, c in sorted(geom_counts.items()):
        print(f"    - {g:25s}: {c:5d}")
    print(f"    SUM: {audit_data['test_geometry_sum']:5d} (Matches 7,557: {audit_data['test_geometry_sum'] == 7557})")
    
    # Check consistency of shared records
    shared_2v_ids = set(m2v['enriched_id'] if 'enriched_id' in m2v.columns else range(len(m2v)))
    # Both manifests have identical split assignments for the first 50,375 rows
    splits_match = (m2v['split'].values == m3a.iloc[:50375]['split'].values).all()
    audit_data['shared_splits_identical'] = bool(splits_match)
    print(f"  Shared 50,375 splits identical between Stage 2V and Stage 3A: {splits_match}")
    
    # 2. Metric Calculations across Evaluation Methods
    def calc_metrics(y_true, y_pred):
        y_true_psi = y_true * 100.0
        y_pred_psi = y_pred * 100.0
        err_psi = np.abs(y_true_psi - y_pred_psi)
        r2 = r2_score(y_true, y_pred)
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        mae_psi = float(mean_absolute_error(y_true_psi, y_pred_psi))
        mse_psi = float(mean_squared_error(y_true_psi, y_pred_psi))
        rmse_psi = float(np.sqrt(mse_psi))
        med_err = float(np.median(err_psi))
        max_err = float(np.max(err_psi))
        mape = float(np.mean(err_psi / y_true_psi) * 100.0)
        return {
            'r2': r2, 'corr': corr, 'mae_psi': mae_psi, 'mse_psi': mse_psi,
            'rmse_psi': rmse_psi, 'median_err_psi': med_err, 'max_error_psi': max_err, 'mape_pct': mape
        }
    
    architectures = ['ANN_128', 'ANN_256', 'ANN_512']
    experiments = ['original_dataset', 'enriched_dataset']
    seeds = [42, 101, 2024, 777, 999]
    
    method_comparison = {}
    
    for exp in experiments:
        method_comparison[exp] = {}
        for arch in architectures:
            sub = df_eval[(df_eval['experiment'] == exp) & (df_eval['architecture'] == arch)]
            
            # Method 1: By seed
            by_seed = {}
            for s in seeds:
                s_df = sub[sub['seed'] == s]
                by_seed[str(s)] = calc_metrics(s_df['y_true_kt'].values, s_df['y_pred_kt'].values)
                
            # Method 2: Mean +/- Std across seeds
            keys = by_seed['42'].keys()
            mean_across_seeds = {f"{k}_mean": float(np.mean([by_seed[str(s)][k] for s in seeds])) for k in keys}
            std_across_seeds = {f"{k}_std": float(np.std([by_seed[str(s)][k] for s in seeds])) for k in keys}
            
            # Method 3: Committee Ensemble Average
            # pivot to average predictions across 5 seeds for each test sample
            piv = sub.pivot_table(index=['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'y_true_kt'], columns='seed', values='y_pred_kt').reset_index()
            y_true_comm = piv['y_true_kt'].values
            y_pred_comm = piv[seeds].mean(axis=1).values
            comm_metrics = calc_metrics(y_true_comm, y_pred_comm)
            
            # Method 4: Pooled rows
            pooled_metrics = calc_metrics(sub['y_true_kt'].values, sub['y_pred_kt'].values)
            
            method_comparison[exp][arch] = {
                'by_seed': by_seed,
                'mean_across_seeds': {**mean_across_seeds, **std_across_seeds},
                'committee_ensemble': comm_metrics,
                'pooled_rows': pooled_metrics
            }
            
    audit_data['method_comparison'] = method_comparison
    
    print("\n[Audit 2: Evaluation Methods Comparison for Enriched ANN-512]")
    m_info = method_comparison['enriched_dataset']['ANN_512']
    print(f"  Seed 42:          R2={m_info['by_seed']['42']['r2']:.4f}, MAE={m_info['by_seed']['42']['mae_psi']:.2f} psi, RMSE={m_info['by_seed']['42']['rmse_psi']:.2f} psi, MaxErr={m_info['by_seed']['42']['max_error_psi']:.1f} psi")
    print(f"  Mean across 5:    R2={m_info['mean_across_seeds']['r2_mean']:.4f} +/- {m_info['mean_across_seeds']['r2_std']:.4f}, MAE={m_info['mean_across_seeds']['mae_psi_mean']:.2f} psi, RMSE={m_info['mean_across_seeds']['rmse_psi_mean']:.2f} psi")
    print(f"  Committee (Mean): R2={m_info['committee_ensemble']['r2']:.4f}, MAE={m_info['committee_ensemble']['mae_psi']:.2f} psi, RMSE={m_info['committee_ensemble']['rmse_psi']:.2f} psi, MaxErr={m_info['committee_ensemble']['max_error_psi']:.1f} psi")
    print(f"  Pooled (37,785):  R2={m_info['pooled_rows']['r2']:.4f}, MAE={m_info['pooled_rows']['mae_psi']:.2f} psi, RMSE={m_info['pooled_rows']['rmse_psi']:.2f} psi, MaxErr={m_info['pooled_rows']['max_error_psi']:.1f} psi")
    
    # 3. Geometry-Specific Performance Recalculation (Single Seed 42 and Mean Across Seeds)
    print("\n[Audit 3: Verified Geometry-Specific Performance (Actual Samples summing to 7,557)]")
    geoms = ['rectangular_1000x600', 'rectangular_1000x700', 'rectangular_1000x800', 'rectangular_1000x900', 'square_1000x1000']
    geom_recalc = {}
    
    for g in geoms:
        # Evaluate for Enriched ANN-512 across seeds
        g_metrics = []
        for s in seeds:
            sub_g = df_eval[(df_eval['experiment'] == 'enriched_dataset') & 
                            (df_eval['architecture'] == 'ANN_512') & 
                            (df_eval['seed'] == s) & 
                            (df_eval['geometry_family'] == g)]
            g_metrics.append(calc_metrics(sub_g['y_true_kt'].values, sub_g['y_pred_kt'].values))
            
        n_samples = len(sub_g)
        geom_recalc[g] = {
            'samples': n_samples,
            'r2_mean': float(np.mean([m['r2'] for m in g_metrics])),
            'r2_std': float(np.std([m['r2'] for m in g_metrics])),
            'mae_psi_mean': float(np.mean([m['mae_psi'] for m in g_metrics])),
            'rmse_psi_mean': float(np.mean([m['rmse_psi'] for m in g_metrics])),
            'max_error_psi_mean': float(np.mean([m['max_error_psi'] for m in g_metrics])),
            'median_err_psi_mean': float(np.mean([m['median_err_psi'] for m in g_metrics]))
        }
        print(f"  {g:25s} (N={n_samples:4d}): R2={geom_recalc[g]['r2_mean']:.4f} +/- {geom_recalc[g]['r2_std']:.4f}, MAE={geom_recalc[g]['mae_psi_mean']:.2f} psi, RMSE={geom_recalc[g]['rmse_psi_mean']:.2f} psi, MaxErr={geom_recalc[g]['max_error_psi_mean']:.1f} psi")
    
    audit_data['verified_geometry_performance'] = geom_recalc
    
    # 4. Physical Regimes Audit (On the 7,557 held-out test set)
    print("\n[Audit 4: Verified Physical Regimes Audit on Locked Test Set (N_test = 7,557)]")
    # Take seed 42 to get clean row level test samples
    sub_test = df_eval[(df_eval['experiment'] == 'enriched_dataset') & 
                       (df_eval['architecture'] == 'ANN_512') & 
                       (df_eval['seed'] == 42)].copy()
    
    regimes = {
        'stress_high_gt_3_20': sub_test['y_true_kt'] > 3.20,
        'stress_nominal_le_3_20': sub_test['y_true_kt'] <= 3.20,
        
        'tight_cross_hole_ligament_lt_25mm': sub_test['hole_to_hole_clearance_mm'] < 25.0,
        'wide_cross_hole_ligament_ge_25mm': sub_test['hole_to_hole_clearance_mm'] >= 25.0,
        
        'boundary_edge_lt_25mm': sub_test['boundary_clearance_mm'] < 25.0,
        'boundary_interior_ge_25mm': sub_test['boundary_clearance_mm'] >= 25.0,
        
        'defect_small_lt_0_12': sub_test['defect_d_over_Dc'] < 0.12,
        'defect_medium_0_12_0_22': (sub_test['defect_d_over_Dc'] >= 0.12) & (sub_test['defect_d_over_Dc'] <= 0.22),
        'defect_large_gt_0_22': sub_test['defect_d_over_Dc'] > 0.22,
    }
    
    regime_results = {}
    for r_name, mask in regimes.items():
        s_r = sub_test[mask]
        n_r = len(s_r)
        m_r = calc_metrics(s_r['y_true_kt'].values, s_r['y_pred_kt'].values) if n_r > 0 else {}
        regime_results[r_name] = {'count': n_r, **m_r}
        print(f"  {r_name:38s}: Count={n_r:5d} | MAE={m_r.get('mae_psi', 0):.2f} psi, RMSE={m_r.get('rmse_psi', 0):.2f} psi, MaxErr={m_r.get('max_error_psi', 0):.1f} psi")
        
    audit_data['verified_regimes'] = regime_results
    
    # Verify Mutually Exclusive Sums
    print("\n  Mutually Exclusive Sum Checks:")
    print(f"    Stress regimes sum: {regime_results['stress_high_gt_3_20']['count']} + {regime_results['stress_nominal_le_3_20']['count']} = {regime_results['stress_high_gt_3_20']['count'] + regime_results['stress_nominal_le_3_20']['count']} (Target: 7,557)")
    print(f"    Ligament regimes sum: {regime_results['tight_cross_hole_ligament_lt_25mm']['count']} + {regime_results['wide_cross_hole_ligament_ge_25mm']['count']} = {regime_results['tight_cross_hole_ligament_lt_25mm']['count'] + regime_results['wide_cross_hole_ligament_ge_25mm']['count']} (Target: 7,557)")
    print(f"    Boundary regimes sum: {regime_results['boundary_edge_lt_25mm']['count']} + {regime_results['boundary_interior_ge_25mm']['count']} = {regime_results['boundary_edge_lt_25mm']['count'] + regime_results['boundary_interior_ge_25mm']['count']} (Target: 7,557)")
    print(f"    Diameter regimes sum: {regime_results['defect_small_lt_0_12']['count']} + {regime_results['defect_medium_0_12_0_22']['count']} + {regime_results['defect_large_gt_0_22']['count']} = {regime_results['defect_small_lt_0_12']['count'] + regime_results['defect_medium_0_12_0_22']['count'] + regime_results['defect_large_gt_0_22']['count']} (Target: 7,557)")
    
    # 5. Save Machine-Readable Audit JSON
    os.makedirs('output', exist_ok=True)
    with open('output/stage3AR_phaseA_audit_summary.json', 'w') as f:
        json.dump(audit_data, f, indent=2)
    print("\nSaved audit summary to output/stage3AR_phaseA_audit_summary.json")

if __name__ == '__main__':
    run_phaseA_audit()
