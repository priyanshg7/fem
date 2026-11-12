"""
Stage 2V Phase B - Preliminary Surrogate Model Baselines
Trains a 4-model committee on canonical geometry-aware dimensionless features.
Establishes leakage-free Train/Val/Test splits, Spatially Blocked evaluation,
and Leave-One-Geometry-Out (LOGO) generalization experiments.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def create_canonical_dataset(df_uni: pd.DataFrame) -> pd.DataFrame:
    """Add canonical dimensionless features to dataset."""
    df = df_uni.copy()
    
    # Coordinates & dimensions
    x = df['defect_center_x_mm']
    y = df['defect_center_y_mm']
    d = df['defect_diameter_mm']
    dc = df['central_hole_diameter_mm']
    l = df['plate_length_mm']
    h = df['plate_height_mm']
    
    # Canonical dimensionless features
    df['defect_x_over_L'] = x / l
    df['defect_y_over_H'] = y / h
    df['defect_d_over_H'] = d / h
    df['defect_d_over_Dc'] = d / dc
    df['central_hole_d_over_H'] = dc / h
    df['central_hole_d_over_L'] = dc / l
    # aspect_ratio already exists, ensure float
    df['aspect_ratio'] = df['aspect_ratio'].astype(float)
    
    # Primary target: Kt_element
    df['Kt_element'] = df['max_von_mises_element_psi'] / df['applied_traction_psi']
    
    # Clearance metrics for physical error breakdown
    # Distance to central hole: dist((x,y), (L/2, H/2)) - (Dc + d)/2
    xc = l / 2.0
    yc = h / 2.0
    dist_center = np.hypot(x - xc, y - yc)
    df['hole_to_hole_clearance_mm'] = dist_center - (dc + d) / 2.0
    
    # Distance to outer boundary: min(x - d/2, L - (x + d/2), y - d/2, H - (y + d/2))
    df['boundary_clearance_mm'] = np.minimum(
        np.minimum(x - d / 2.0, l - (x + d / 2.0)),
        np.minimum(y - d / 2.0, h - (y + d / 2.0))
    )
    
    return df

def generate_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate leakage-free 70% Train, 15% Val, 15% Test split
    stratified by geometry_family and subset_type.
    """
    df = df.copy()
    df['strat_key'] = df['geometry_family'] + '_' + df['subset_type']
    
    # First split: 70% train vs 30% temp (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, df['strat_key']))
    
    df_temp = df.iloc[temp_idx].copy()
    
    # Second split: 50% of temp is val (15% total), 50% is test (15% total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(df_temp, df_temp['strat_key']))
    
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    
    df['split'] = ''
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'
    
    # Verify exact counts
    print(f"Split counts: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Save manifest
    os.makedirs('data', exist_ok=True)
    manifest = df[['unified_id', 'geometry_family', 'subset_type', 'split']]
    manifest.to_csv('data/stage2V_split_manifest.csv', index=False)
    print("Saved split manifest to data/stage2V_split_manifest.csv")
    
    return df

def train_preliminary_models(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train 4 models and return evaluation results and validation predictions."""
    features = [
        'defect_x_over_L',
        'defect_y_over_H',
        'defect_d_over_Dc',
        'aspect_ratio',
        'central_hole_d_over_L'
    ]
    target = 'Kt_element'
    
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    
    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, target].values
    
    X_val = df.loc[val_mask, features].values
    y_val = df.loc[val_mask, target].values
    
    # Standard scaler for Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    os.makedirs('models/stage2V_preliminary', exist_ok=True)
    joblib.dump(scaler, 'models/stage2V_preliminary/scaler.joblib')
    
    models = {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150, max_depth=12, random_state=42),
        'MLP_NeuralNet': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=250, early_stopping=True, random_state=42)
    }
    
    metrics_summary = {}
    val_preds_df = df.loc[val_mask].copy()
    
    print("\n--- Training Preliminary Model Committee ---")
    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'MLP_NeuralNet':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
        joblib.dump(model, f'models/stage2V_preliminary/{name}.joblib')
        
        # Calculate metrics
        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(y_val, preds)[0, 1]
        mape = np.mean(np.abs((y_val - preds) / y_val)) * 100.0
        max_err = np.max(np.abs(y_val - preds))
        
        # In dimensional terms (psi)
        mae_psi = mae * 100.0
        rmse_psi = rmse * 100.0
        max_err_psi = max_err * 100.0
        
        metrics_summary[name] = {
            'r2': float(r2),
            'correlation': float(corr),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape_pct': float(mape),
            'max_abs_error': float(max_err),
            'mae_psi': float(mae_psi),
            'rmse_psi': float(rmse_psi),
            'max_abs_error_psi': float(max_err_psi)
        }
        
        val_preds_df[f'pred_{name}'] = preds
        val_preds_df[f'resid_{name}'] = preds - y_val
        print(f"  {name}: R2={r2:.4f}, MAE={mae:.4f} ({mae_psi:.2f} psi), RMSE={rmse:.4f} ({rmse_psi:.2f} psi)")
        
    # Model committee consensus and disagreement
    pred_cols = [f'pred_{name}' for name in models.keys()]
    val_preds_df['committee_mean_pred'] = val_preds_df[pred_cols].mean(axis=1)
    val_preds_df['committee_std_disagreement'] = val_preds_df[pred_cols].std(axis=1)
    val_preds_df['committee_abs_error'] = np.abs(val_preds_df['committee_mean_pred'] - val_preds_df[target])
    
    # Committee metrics
    c_r2 = r2_score(y_val, val_preds_df['committee_mean_pred'])
    c_mae = mean_absolute_error(y_val, val_preds_df['committee_mean_pred'])
    c_rmse = np.sqrt(mean_squared_error(y_val, val_preds_df['committee_mean_pred']))
    metrics_summary['Committee_Ensemble'] = {
        'r2': float(c_r2),
        'correlation': float(np.corrcoef(y_val, val_preds_df['committee_mean_pred'])[0, 1]),
        'mae': float(c_mae),
        'rmse': float(c_rmse),
        'mape_pct': float(np.mean(np.abs((y_val - val_preds_df['committee_mean_pred']) / y_val)) * 100.0),
        'max_abs_error': float(np.max(val_preds_df['committee_abs_error'])),
        'mae_psi': float(c_mae * 100.0),
        'rmse_psi': float(c_rmse * 100.0),
        'max_abs_error_psi': float(np.max(val_preds_df['committee_abs_error']) * 100.0)
    }
    print(f"\n  Committee Ensemble: R2={c_r2:.4f}, MAE={c_mae:.4f} ({c_mae*100:.2f} psi), RMSE={c_rmse:.4f}")
    
    # Save validation predictions
    val_preds_df.to_csv('data/stage2V_preliminary_val_predictions.csv', index=False)
    
    return metrics_summary, val_preds_df

def run_generalization_experiments(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Spatially Blocked and Leave-One-Geometry-Out (LOGO) evaluations."""
    features = [
        'defect_x_over_L',
        'defect_y_over_H',
        'defect_d_over_Dc',
        'aspect_ratio',
        'central_hole_d_over_L'
    ]
    target = 'Kt_element'
    gen_results = {'spatial_blocked': {}, 'leave_one_geometry_out': {}}
    
    print("\n--- Running Spatially Blocked Generalization Experiment ---")
    # Hold out defect in quadrant: X/L > 0.65 and Y/H > 0.65 (unseen spatial region)
    spatial_test_mask = (df['defect_x_over_L'] > 0.65) & (df['defect_y_over_H'] > 0.65)
    spatial_train_mask = ~spatial_test_mask
    
    X_sp_train = df.loc[spatial_train_mask, features].values
    y_sp_train = df.loc[spatial_train_mask, target].values
    X_sp_test = df.loc[spatial_test_mask, features].values
    y_sp_test = df.loc[spatial_test_mask, target].values
    
    et_spatial = ExtraTreesRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    et_spatial.fit(X_sp_train, y_sp_train)
    sp_preds = et_spatial.predict(X_sp_test)
    
    gen_results['spatial_blocked'] = {
        'test_condition': 'X/L > 0.65 and Y/H > 0.65 (Top-Right Quadrant Blocked)',
        'train_samples': int(np.sum(spatial_train_mask)),
        'test_samples': int(np.sum(spatial_test_mask)),
        'r2': float(r2_score(y_sp_test, sp_preds)),
        'mae': float(mean_absolute_error(y_sp_test, sp_preds)),
        'rmse': float(np.sqrt(mean_squared_error(y_sp_test, sp_preds))),
        'mae_psi': float(mean_absolute_error(y_sp_test, sp_preds) * 100.0)
    }
    print(f"  Spatially Blocked (Held-out Quad): R2={gen_results['spatial_blocked']['r2']:.4f}, MAE={gen_results['spatial_blocked']['mae_psi']:.2f} psi")
    
    print("\n--- Running Leave-One-Geometry-Out (LOGO) Generalization Experiment ---")
    logo_results = {}
    geometries = df['geometry_family'].unique()
    
    for held_out_geom in geometries:
        test_mask = df['geometry_family'] == held_out_geom
        train_mask = ~test_mask
        
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, target].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, target].values
        
        rf_logo = HistGradientBoostingRegressor(max_iter=150, max_depth=12, random_state=42)
        rf_logo.fit(X_tr, y_tr)
        logo_preds = rf_logo.predict(X_te)
        
        logo_results[held_out_geom] = {
            'train_samples': int(np.sum(train_mask)),
            'test_samples': int(np.sum(test_mask)),
            'r2': float(r2_score(y_te, logo_preds)),
            'mae': float(mean_absolute_error(y_te, logo_preds)),
            'rmse': float(np.sqrt(mean_squared_error(y_te, logo_preds))),
            'mae_psi': float(mean_absolute_error(y_te, logo_preds) * 100.0)
        }
        print(f"  Held-out {held_out_geom}: R2={logo_results[held_out_geom]['r2']:.4f}, MAE={logo_results[held_out_geom]['mae_psi']:.2f} psi")
        
    gen_results['leave_one_geometry_out'] = logo_results
    return gen_results

if __name__ == '__main__':
    df_uni = pd.read_csv('data/unified_dataset_50375.csv')
    df = create_canonical_dataset(df_uni)
    df = generate_splits(df)
    
    metrics, val_preds = train_preliminary_models(df)
    gen_experiments = run_generalization_experiments(df)
    
    # Save full metrics summary
    full_output = {
        'model_metrics': metrics,
        'generalization_experiments': gen_experiments
    }
    with open('output/stage2V_preliminary_model_metrics.json', 'w') as f:
        json.dump(full_output, f, indent=2)
    print("\nStage 2V Phase B Baseline Modeling Completed Successfully.")

