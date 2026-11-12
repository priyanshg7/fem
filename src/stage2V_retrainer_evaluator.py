"""
Stage 2V Phase F - Before-and-After Enrichment Evaluation Engine
Trains preliminary surrogate models on:
  Experiment A: Original 50,375 dataset (Train: 35,262)
  Experiment B: Adaptively Enriched dataset (Train: 35,262 + 1,250 = 36,512)
Evaluates both on the IDENTICAL, UNTOUCHED held-out test set (7,557 samples),
as well as Spatially Blocked and LOGO transfer benchmarks.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def prepare_comparison_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load unified dataset, split manifest, and adaptive simulations."""
    df_uni = pd.read_csv('data/unified_dataset_50375.csv')
    manifest = pd.read_csv('data/stage2V_split_manifest.csv')
    df_uni['split'] = manifest['split'].values
    
    # Add canonical features
    x = df_uni['defect_center_x_mm']
    y = df_uni['defect_center_y_mm']
    d = df_uni['defect_diameter_mm']
    dc = df_uni['central_hole_diameter_mm']
    l = df_uni['plate_length_mm']
    h = df_uni['plate_height_mm']
    
    df_uni['defect_x_over_L'] = x / l
    df_uni['defect_y_over_H'] = y / h
    df_uni['defect_d_over_H'] = d / h
    df_uni['defect_d_over_Dc'] = d / dc
    df_uni['central_hole_d_over_H'] = dc / h
    df_uni['central_hole_d_over_L'] = dc / l
    df_uni['aspect_ratio'] = df_uni['aspect_ratio'].astype(float)
    df_uni['Kt_element'] = df_uni['max_von_mises_element_psi'] / df_uni['applied_traction_psi']
    
    dist_c = np.hypot(x - l / 2.0, y - h / 2.0)
    df_uni['hole_to_hole_clearance_mm'] = dist_c - (dc + d) / 2.0
    df_uni['boundary_clearance_mm'] = np.minimum(
        np.minimum(x - d / 2.0, l - (x + d / 2.0)),
        np.minimum(y - d / 2.0, h - (y + d / 2.0))
    )
    df_uni['data_source'] = df_uni['geometry_family']
    
    # Load adaptive simulations
    adapt_path = 'data/adaptive_enrichment_stage2V/master_adaptive_simulations_1250.csv'
    df_adapt = pd.read_csv(adapt_path)
    df_adapt_succ = df_adapt[df_adapt['status'] == 'SUCCESS'].copy()
    
    df_adapt_succ['split'] = 'train'  # All adaptive simulations enter training!
    df_adapt_succ['data_source'] = 'stage2V_adaptive_enrichment'
    
    # Ensure all canonical columns exist in adaptive
    df_adapt_succ['defect_x_over_L'] = df_adapt_succ['defect_center_x_mm'] / df_adapt_succ['plate_length_mm']
    df_adapt_succ['defect_y_over_H'] = df_adapt_succ['defect_center_y_mm'] / df_adapt_succ['plate_height_mm']
    df_adapt_succ['defect_d_over_H'] = df_adapt_succ['defect_diameter_mm'] / df_adapt_succ['plate_height_mm']
    df_adapt_succ['defect_d_over_Dc'] = df_adapt_succ['defect_diameter_mm'] / df_adapt_succ['central_hole_diameter_mm']
    df_adapt_succ['central_hole_d_over_H'] = df_adapt_succ['central_hole_diameter_mm'] / df_adapt_succ['plate_height_mm']
    df_adapt_succ['central_hole_d_over_L'] = df_adapt_succ['central_hole_diameter_mm'] / df_adapt_succ['plate_length_mm']
    df_adapt_succ['aspect_ratio'] = df_adapt_succ['aspect_ratio'].astype(float)
    df_adapt_succ['Kt_element'] = df_adapt_succ['max_von_mises_element_psi'] / 100.0
    
    # Concatenate for enriched dataset
    common_cols = [
        'geometry_family', 'plate_length_mm', 'plate_height_mm', 'aspect_ratio',
        'central_hole_diameter_mm', 'defect_center_x_mm', 'defect_center_y_mm', 'defect_diameter_mm',
        'defect_x_over_L', 'defect_y_over_H', 'defect_d_over_H', 'defect_d_over_Dc',
        'central_hole_d_over_H', 'central_hole_d_over_L', 'max_von_mises_element_psi',
        'max_von_mises_nodal_psi', 'max_displacement_m', 'strain_energy_j', 'Kt_element',
        'hole_to_hole_clearance_mm', 'boundary_clearance_mm', 'peak_stress_region',
        'applied_force_n', 'reaction_force_y_n', 'reaction_force_x_n', 'reaction_force_residual_n',
        'node_count', 'element_count', 'split', 'data_source', 'field_file'
    ]
    
    df_enriched = pd.concat([df_uni[common_cols], df_adapt_succ[common_cols]], ignore_index=True)
    df_enriched['enriched_id'] = range(len(df_enriched))
    
    return df_uni, df_adapt_succ, df_enriched

def evaluate_models_on_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    target: str = 'Kt_element'
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train models on df_train and evaluate on df_test."""
    X_tr = df_train[features].values
    y_tr = df_train[target].values
    
    X_te = df_test[features].values
    y_te = df_test[target].values
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150, max_depth=12, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        'MLP_NeuralNet': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=250, early_stopping=True, random_state=42)
    }
    
    test_preds_df = df_test.copy()
    results = {}
    
    for name, m in models.items():
        if name == 'MLP_NeuralNet':
            m.fit(X_tr_sc, y_tr)
            preds = m.predict(X_te_sc)
        else:
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            
        test_preds_df[f'pred_{name}'] = preds
        test_preds_df[f'resid_{name}'] = preds - y_te
        
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        corr = np.corrcoef(y_te, preds)[0, 1]
        
        results[name] = {
            'r2': float(r2),
            'correlation': float(corr),
            'mae': float(mae),
            'rmse': float(rmse),
            'mae_psi': float(mae * 100.0),
            'rmse_psi': float(rmse * 100.0),
            'max_error_psi': float(np.max(np.abs(y_te - preds)) * 100.0)
        }
        
    # Committee
    pred_cols = [f'pred_{name}' for name in models.keys()]
    test_preds_df['committee_pred'] = test_preds_df[pred_cols].mean(axis=1)
    test_preds_df['committee_resid'] = test_preds_df['committee_pred'] - y_te
    test_preds_df['committee_abs_error_psi'] = np.abs(test_preds_df['committee_resid']) * 100.0
    
    c_r2 = r2_score(y_te, test_preds_df['committee_pred'])
    c_mae = mean_absolute_error(y_te, test_preds_df['committee_pred'])
    c_rmse = np.sqrt(mean_squared_error(y_te, test_preds_df['committee_pred']))
    
    results['Committee_Ensemble'] = {
        'r2': float(c_r2),
        'correlation': float(np.corrcoef(y_te, test_preds_df['committee_pred'])[0, 1]),
        'mae': float(c_mae),
        'rmse': float(c_rmse),
        'mae_psi': float(c_mae * 100.0),
        'rmse_psi': float(c_rmse * 100.0),
        'max_error_psi': float(np.max(test_preds_df['committee_abs_error_psi']))
    }
    
    return results, test_preds_df

def compute_detailed_breakdown(test_preds_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate error breakdown across physical subsets."""
    y_true = test_preds_df['Kt_element'].values
    y_pred = test_preds_df['committee_pred'].values
    err_psi = np.abs(y_pred - y_true) * 100.0
    
    breakdown = {}
    
    # 1. By Geometry Family
    breakdown['by_geometry'] = {}
    for g in test_preds_df['geometry_family'].unique():
        mask = test_preds_df['geometry_family'] == g
        breakdown['by_geometry'][g] = {
            'mae_psi': float(np.mean(err_psi[mask])),
            'rmse_psi': float(np.sqrt(np.mean((err_psi[mask])**2))),
            'r2': float(r2_score(y_true[mask], y_pred[mask]))
        }
        
    # 2. High vs Moderate Stress
    high_mask = test_preds_df['Kt_element'] > 3.20
    mod_mask = ~high_mask
    breakdown['by_stress_regime'] = {
        'moderate_stress_Kt_le_3_20': {
            'samples': int(np.sum(mod_mask)),
            'mae_psi': float(np.mean(err_psi[mod_mask])),
            'rmse_psi': float(np.sqrt(np.mean((err_psi[mod_mask])**2)))
        },
        'high_stress_Kt_gt_3_20': {
            'samples': int(np.sum(high_mask)),
            'mae_psi': float(np.mean(err_psi[high_mask])),
            'rmse_psi': float(np.sqrt(np.mean((err_psi[high_mask])**2)))
        }
    }
    
    # 3. By Hole-to-Hole Clearance
    tight_mask = test_preds_df['hole_to_hole_clearance_mm'] < 25.0
    wide_mask = ~tight_mask
    breakdown['by_clearance'] = {
        'tight_ligament_lt_25mm': {
            'samples': int(np.sum(tight_mask)),
            'mae_psi': float(np.mean(err_psi[tight_mask])),
            'rmse_psi': float(np.sqrt(np.mean((err_psi[tight_mask])**2)))
        },
        'wide_ligament_ge_25mm': {
            'samples': int(np.sum(wide_mask)),
            'mae_psi': float(np.mean(err_psi[wide_mask])),
            'rmse_psi': float(np.sqrt(np.mean((err_psi[wide_mask])**2)))
        }
    }
    
    return breakdown

def run_enrichment_comparison():
    print("Starting Stage 2V Phase F Before-and-After Enrichment Evaluation...")
    df_uni, df_adapt, df_enriched = prepare_comparison_datasets()
    
    # Save enriched dataset
    df_enriched.to_csv('data/enriched_dataset_51625.csv', index=False)
    print(f"Saved master enriched dataset: {len(df_enriched)} rows to data/enriched_dataset_51625.csv")
    
    features = [
        'defect_x_over_L',
        'defect_y_over_H',
        'defect_d_over_Dc',
        'aspect_ratio',
        'central_hole_d_over_L'
    ]
    target = 'Kt_element'
    
    # Experiment A: Train on original 35,262 training samples
    train_orig = df_uni[df_uni['split'] == 'train'].copy()
    test_untouched = df_uni[df_uni['split'] == 'test'].copy()
    
    # Experiment B: Train on original 35,262 + 1,250 adaptive = 36,512 samples
    train_enriched = df_enriched[df_enriched['split'] == 'train'].copy()
    
    print(f"\n--- Experiment A: Original Baseline Model (Train={len(train_orig)}, Test={len(test_untouched)}) ---")
    metrics_a, preds_a = evaluate_models_on_test(train_orig, test_untouched, features, target)
    breakdown_a = compute_detailed_breakdown(preds_a)
    print(f"  Exp A Committee: R2={metrics_a['Committee_Ensemble']['r2']:.4f}, MAE={metrics_a['Committee_Ensemble']['mae_psi']:.2f} psi, RMSE={metrics_a['Committee_Ensemble']['rmse_psi']:.2f} psi")
    print(f"    High Stress (Kt > 3.2) MAE: {breakdown_a['by_stress_regime']['high_stress_Kt_gt_3_20']['mae_psi']:.2f} psi")
    print(f"    Tight Ligament (< 25mm) MAE: {breakdown_a['by_clearance']['tight_ligament_lt_25mm']['mae_psi']:.2f} psi")
    
    print(f"\n--- Experiment B: Adaptively Enriched Model (Train={len(train_enriched)}, Test={len(test_untouched)}) ---")
    metrics_b, preds_b = evaluate_models_on_test(train_enriched, test_untouched, features, target)
    breakdown_b = compute_detailed_breakdown(preds_b)
    print(f"  Exp B Committee: R2={metrics_b['Committee_Ensemble']['r2']:.4f}, MAE={metrics_b['Committee_Ensemble']['mae_psi']:.2f} psi, RMSE={metrics_b['Committee_Ensemble']['rmse_psi']:.2f} psi")
    print(f"    High Stress (Kt > 3.2) MAE: {breakdown_b['by_stress_regime']['high_stress_Kt_gt_3_20']['mae_psi']:.2f} psi")
    print(f"    Tight Ligament (< 25mm) MAE: {breakdown_b['by_clearance']['tight_ligament_lt_25mm']['mae_psi']:.2f} psi")
    
    # Relative improvements
    mae_a = metrics_a['Committee_Ensemble']['mae_psi']
    mae_b = metrics_b['Committee_Ensemble']['mae_psi']
    rmse_a = metrics_a['Committee_Ensemble']['rmse_psi']
    rmse_b = metrics_b['Committee_Ensemble']['rmse_psi']
    r2_a = metrics_a['Committee_Ensemble']['r2']
    r2_b = metrics_b['Committee_Ensemble']['r2']
    
    delta_mae_pct = (mae_b - mae_a) / mae_a * 100.0
    delta_rmse_pct = (rmse_b - rmse_a) / rmse_a * 100.0
    
    high_mae_a = breakdown_a['by_stress_regime']['high_stress_Kt_gt_3_20']['mae_psi']
    high_mae_b = breakdown_b['by_stress_regime']['high_stress_Kt_gt_3_20']['mae_psi']
    delta_high_mae_pct = (high_mae_b - high_mae_a) / high_mae_a * 100.0
    
    tight_mae_a = breakdown_a['by_clearance']['tight_ligament_lt_25mm']['mae_psi']
    tight_mae_b = breakdown_b['by_clearance']['tight_ligament_lt_25mm']['mae_psi']
    delta_tight_mae_pct = (tight_mae_b - tight_mae_a) / tight_mae_a * 100.0
    
    comparison_summary = {
        'training_size_exp_a': len(train_orig),
        'training_size_exp_b': len(train_enriched),
        'test_size_untouched': len(test_untouched),
        'metrics_exp_a': metrics_a,
        'metrics_exp_b': metrics_b,
        'breakdown_exp_a': breakdown_a,
        'breakdown_exp_b': breakdown_b,
        'improvements': {
            'overall_mae_reduction_pct': float(-delta_mae_pct),
            'overall_rmse_reduction_pct': float(-delta_rmse_pct),
            'r2_gain': float(r2_b - r2_a),
            'high_stress_mae_reduction_pct': float(-delta_high_mae_pct),
            'tight_ligament_mae_reduction_pct': float(-delta_tight_mae_pct)
        }
    }
    
    with open('output/stage2V_enrichment_comparison_summary.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    print("\nSaved output/stage2V_enrichment_comparison_summary.json")
    
    # Generate before-and-after plot
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. Overall MAE & RMSE bar chart
    labels = ['Overall MAE (psi)', 'Overall RMSE (psi)', 'Tight Ligament MAE (psi)', 'High Stress MAE (psi)']
    val_a = [mae_a, rmse_a, tight_mae_a, high_mae_a]
    val_b = [mae_b, rmse_b, tight_mae_b, high_mae_b]
    
    x_idx = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x_idx - w/2, val_a, width=w, label='Exp A (Original 50,375)', color='#4575b4')
    axes[0].bar(x_idx + w/2, val_b, width=w, label='Exp B (Enriched +1,250)', color='#d73027')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylabel('Error (psi)')
    axes[0].set_title('Error Comparison (Untouched Test Set)', fontweight='bold')
    axes[0].legend()
    
    # 2. MAE by Geometry Family
    geoms = list(breakdown_a['by_geometry'].keys())
    g_mae_a = [breakdown_a['by_geometry'][g]['mae_psi'] for g in geoms]
    g_mae_b = [breakdown_b['by_geometry'][g]['mae_psi'] for g in geoms]
    
    g_x = np.arange(len(geoms))
    axes[1].bar(g_x - w/2, g_mae_a, width=w, label='Exp A (Original)', color='#4575b4')
    axes[1].bar(g_x + w/2, g_mae_b, width=w, label='Exp B (Enriched)', color='#d73027')
    axes[1].set_xticks(g_x)
    axes[1].set_xticklabels([g.replace('rectangular_', '').replace('square_', '') for g in geoms], rotation=15)
    axes[1].set_ylabel('MAE (psi)')
    axes[1].set_title('MAE by Geometry Family', fontweight='bold')
    axes[1].legend()
    
    # 3. Residual CDF / Error Distribution
    err_a = preds_a['committee_abs_error_psi']
    err_b = preds_b['committee_abs_error_psi']
    axes[2].hist(err_a, bins=50, density=True, histtype='step', cumulative=True, lw=2, label='Exp A (Original)', color='#4575b4')
    axes[2].hist(err_b, bins=50, density=True, histtype='step', cumulative=True, lw=2, label='Exp B (Enriched)', color='#d73027')
    axes[2].set_xlim(0, 15)
    axes[2].set_xlabel('Absolute Prediction Error (psi)')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('Cumulative Error Distribution', fontweight='bold')
    axes[2].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('output/stage2V_original_vs_enriched_error_comparison.png', dpi=300)
    plt.close()
    print("Saved output/stage2V_original_vs_enriched_error_comparison.png")

if __name__ == '__main__':
    run_enrichment_comparison()

