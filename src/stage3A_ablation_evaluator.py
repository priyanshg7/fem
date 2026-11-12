"""
Stage 3A Section 8 - Controlled Feature Ablation Experiment
Compares three dimensionless feature representations on Enriched data using ANN-512 across 5 seeds:
  1. Primary 4-Input: [X/L, Y/H, D/Dc, L/H]
  2. Extended 5-Input: [X/L, Y/H, D/Dc, L/H, Dc/L]
  3. Alternative 4-Input with D/H: [X/L, Y/H, D/H, L/H]
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.stage3A_ann_trainer import ForwardANN, train_single_ann, compute_metrics, set_seed

def run_ablation_experiment():
    print("\n==================================================================")
    print("Starting Stage 3A Feature Ablation Experiment")
    print("==================================================================")
    
    df_enrich = pd.read_csv('data/enriched_dataset_51625.csv')
    
    # Feature configurations
    feature_configs = {
        'primary_4_inputs': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio'],
        'extended_5_inputs_with_Dc_L': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'central_hole_d_over_L'],
        'alternative_4_inputs_with_D_H': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_H', 'aspect_ratio']
    }
    target = 'Kt_element'
    seeds = [42, 101, 2024, 777, 999]
    batch_size = 512
    hidden_neurons = 512
    
    train_df = df_enrich[df_enrich['split'] == 'train']
    val_df = df_enrich[df_enrich['split'] == 'val']
    test_df = df_enrich[df_enrich['split'] == 'test']
    
    y_train = train_df[[target]].values
    y_val = val_df[[target]].values
    y_test = test_df[[target]].values
    
    y_test_phys = test_df[target].values
    
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train)
    y_val_sc = scaler_y.transform(y_val)
    y_test_sc = scaler_y.transform(y_test)
    
    ablation_summary = {}
    
    for cfg_name, feat_cols in feature_configs.items():
        print(f"\n--- Testing Feature Configuration: {cfg_name} ({len(feat_cols)} inputs) ---")
        
        scaler_X = StandardScaler()
        X_train_sc = scaler_X.fit_transform(train_df[feat_cols].values)
        X_val_sc = scaler_X.transform(val_df[feat_cols].values)
        X_test_sc = scaler_X.transform(test_df[feat_cols].values)
        
        t_X_train = torch.tensor(X_train_sc, dtype=torch.float32)
        t_y_train = torch.tensor(y_train_sc, dtype=torch.float32)
        t_X_val = torch.tensor(X_val_sc, dtype=torch.float32)
        t_y_val = torch.tensor(y_val_sc, dtype=torch.float32)
        t_X_test = torch.tensor(X_test_sc, dtype=torch.float32)
        t_y_test = torch.tensor(y_test_sc, dtype=torch.float32)
        
        seed_results = []
        for seed in seeds:
            set_seed(seed)
            train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)
            
            model = ForwardANN(in_features=len(feat_cols), hidden_neurons=hidden_neurons)
            model, info = train_single_ann(
                model=model,
                train_loader=train_loader,
                t_X_val=t_X_val,
                t_y_val=t_y_val,
                target_scaler=scaler_y,
                epochs=150,
                lr=2e-3,
                patience=20
            )
            
            model.eval()
            with torch.no_grad():
                pred_test_sc = model(t_X_test).cpu().numpy()
            pred_test_kt = scaler_y.inverse_transform(pred_test_sc).flatten()
            
            m_test = compute_metrics(y_test_phys, pred_test_kt)
            m_test['best_epoch'] = info['best_epoch']
            seed_results.append(m_test)
            
        r2_vals = [m['r2'] for m in seed_results]
        mae_vals = [m['mae_psi'] for m in seed_results]
        rmse_vals = [m['rmse_psi'] for m in seed_results]
        
        ablation_summary[cfg_name] = {
            'features': feat_cols,
            'input_dim': len(feat_cols),
            'test_r2_mean': float(np.mean(r2_vals)),
            'test_r2_std': float(np.std(r2_vals)),
            'test_mae_psi_mean': float(np.mean(mae_vals)),
            'test_mae_psi_std': float(np.std(mae_vals)),
            'test_rmse_psi_mean': float(np.mean(rmse_vals)),
            'test_rmse_psi_std': float(np.std(rmse_vals)),
            'individual_seeds': seed_results
        }
        
        print(f"  Result: Test R2 = {np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f} | "
              f"MAE = {np.mean(mae_vals):.2f} +/- {np.std(mae_vals):.2f} psi | "
              f"RMSE = {np.mean(rmse_vals):.2f} +/- {np.std(rmse_vals):.2f} psi")
              
    os.makedirs('output', exist_ok=True)
    with open('output/stage3A_ablation_metrics.json', 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    print("Saved output/stage3A_ablation_metrics.json")

if __name__ == '__main__':
    run_ablation_experiment()
