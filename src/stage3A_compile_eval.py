"""
Compiles evaluation records and summary metrics from saved checkpoints.
Runs test set inference across all 30 trained models and outputs master_evaluation_table.csv
and seed_summary_metrics.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from src.stage3A_ann_trainer import ForwardANN, compute_metrics

def compile_evaluation_tables():
    print("Compiling Master Evaluation Table and Summary Metrics from Checkpoints...")
    base_dir = Path('models/stage3A_forward_ann')
    prep_dir = base_dir / 'preprocessing'
    eval_dir = base_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    df_uni = pd.read_csv('data/unified_dataset_50375.csv')
    df_enrich = pd.read_csv('data/enriched_dataset_51625.csv')
    manifest = pd.read_csv('data/stage2V_split_manifest.csv')
    
    # Add canonical features to df_uni
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
    df_uni['split'] = manifest['split'].values
    
    features = ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio']
    target = 'Kt_element'
    
    architectures = {'ANN_128': 128, 'ANN_256': 256, 'ANN_512': 512}
    seeds = [42, 101, 2024, 777, 999]
    experiments = {
        'original_dataset': df_uni,
        'enriched_dataset': df_enrich
    }
    
    master_records = []
    summary_metrics = {}
    
    for exp_name, df_exp in experiments.items():
        summary_metrics[exp_name] = {}
        scaler_X = joblib.load(prep_dir / f"scaler_inputs_{exp_name}.joblib")
        scaler_y = joblib.load(prep_dir / f"scaler_target_{exp_name}.joblib")
        
        val_df = df_exp[df_exp['split'] == 'val']
        test_df = df_exp[df_exp['split'] == 'test']
        
        X_val_sc = scaler_X.transform(val_df[features].values)
        X_test_sc = scaler_X.transform(test_df[features].values)
        
        t_X_val = torch.tensor(X_val_sc, dtype=torch.float32)
        t_X_test = torch.tensor(X_test_sc, dtype=torch.float32)
        
        y_val_phys = val_df[target].values
        y_test_phys = test_df[target].values
        
        for arch_name, hidden_neurons in architectures.items():
            arch_dir = base_dir / exp_name / arch_name
            seed_metrics_val = []
            seed_metrics_test = []
            
            for seed in seeds:
                ckpt_path = arch_dir / f"model_{arch_name}_seed_{seed}.pt"
                model = ForwardANN(in_features=4, hidden_neurons=hidden_neurons)
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                model.eval()
                
                with torch.no_grad():
                    pred_val_sc = model(t_X_val).numpy()
                    pred_test_sc = model(t_X_test).numpy()
                    
                pred_val_kt = scaler_y.inverse_transform(pred_val_sc).flatten()
                pred_test_kt = scaler_y.inverse_transform(pred_test_sc).flatten()
                
                m_val = compute_metrics(y_val_phys, pred_val_kt)
                m_test = compute_metrics(y_test_phys, pred_test_kt)
                
                with open(arch_dir / f"history_seed_{seed}.json") as f:
                    h_info = json.load(f)
                m_val['best_epoch'] = h_info['best_epoch']
                m_val['train_time_s'] = h_info['train_time_s']
                m_test['best_epoch'] = h_info['best_epoch']
                m_test['train_time_s'] = h_info['train_time_s']
                
                seed_metrics_val.append(m_val)
                seed_metrics_test.append(m_test)
                
                df_rec = pd.DataFrame({
                    'experiment': exp_name,
                    'architecture': arch_name,
                    'seed': seed,
                    'split': 'test',
                    'geometry_family': test_df['geometry_family'].values,
                    'aspect_ratio': test_df['aspect_ratio'].values,
                    'defect_x_over_L': test_df['defect_x_over_L'].values,
                    'defect_y_over_H': test_df['defect_y_over_H'].values,
                    'defect_d_over_Dc': test_df['defect_d_over_Dc'].values,
                    'hole_to_hole_clearance_mm': test_df['hole_to_hole_clearance_mm'].values,
                    'boundary_clearance_mm': test_df['boundary_clearance_mm'].values,
                    'y_true_kt': y_test_phys,
                    'y_pred_kt': pred_test_kt,
                    'abs_error_psi': np.abs(y_test_phys - pred_test_kt) * 100.0,
                    'signed_error_psi': (pred_test_kt - y_test_phys) * 100.0
                })
                master_records.append(df_rec)
                
            def aggregate_metrics(metrics_list: list) -> dict:
                res = {}
                keys = metrics_list[0].keys()
                for k in keys:
                    vals = [m[k] for m in metrics_list]
                    res[f'{k}_mean'] = float(np.mean(vals))
                    res[f'{k}_std'] = float(np.std(vals))
                return res
                
            ckpt_sample = arch_dir / f"model_{arch_name}_seed_42.pt"
            summary_metrics[exp_name][arch_name] = {
                'validation': aggregate_metrics(seed_metrics_val),
                'test': aggregate_metrics(seed_metrics_test),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'model_size_kb': round(os.path.getsize(ckpt_sample) / 1024, 2)
            }
            
    df_master = pd.concat(master_records, ignore_index=True)
    df_master.to_csv(eval_dir / 'master_evaluation_table.csv', index=False)
    print(f"Saved master evaluation table: {len(df_master)} rows to {eval_dir / 'master_evaluation_table.csv'}")
    
    with open(eval_dir / 'seed_summary_metrics.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Saved summary metrics to {eval_dir / 'seed_summary_metrics.json'}")
    
    # Save Preprocessing Metadata
    prep_meta = {
        'input_features': features,
        'target': target,
        'scaler_type': 'StandardScaler',
        'fit_scope': 'Strictly on training split',
        'architectures': architectures,
        'training_policy': {
            'optimizer': 'Adam',
            'learning_rate': 2e-3,
            'batch_size': 512,
            'weight_decay': 1e-5,
            'scheduler': 'ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)',
            'early_stopping_patience': 20,
            'max_epochs': 150,
            'loss_function': 'MSELoss'
        },
        'dependencies': {
            'torch': torch.__version__,
            'scikit-learn': '1.8.0',
            'numpy': np.__version__,
            'pandas': pd.__version__
        }
    }
    with open(prep_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(prep_meta, f, indent=2)
    print("Saved preprocessing metadata.")

if __name__ == '__main__':
    compile_evaluation_tables()
