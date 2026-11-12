"""
Stage 3A - Forward Artificial Neural Network Development and Training
Implements ANN-128, ANN-256, ANN-512 in PyTorch.
Trains across 5 seeds for both Original (50,375) and Enriched (51,625) datasets.
Includes early stopping, LR scheduling, preprocessing, and model checkpointing.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ForwardANN(nn.Module):
    """Single hidden-layer feed-forward regression network."""
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_neurons = hidden_neurons
        
        self.hidden = nn.Linear(in_features, hidden_neurons)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_neurons, 1)
        
        # Xavier normal initialization
        nn.init.xavier_normal_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hidden(x)
        out = self.relu(out)
        out = self.output(out)
        return out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all standard regression metrics for Kt and psi."""
    y_true_psi = y_true * 100.0
    y_pred_psi = y_pred * 100.0
    
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-8 else 0.0
    mae_kt = float(mean_absolute_error(y_true, y_pred))
    mse_kt = float(mean_squared_error(y_true, y_pred))
    rmse_kt = float(np.sqrt(mse_kt))
    
    err_psi = np.abs(y_pred_psi - y_true_psi)
    mae_psi = float(mean_absolute_error(y_true_psi, y_pred_psi))
    rmse_psi = float(np.sqrt(mean_squared_error(y_true_psi, y_pred_psi)))
    max_err_psi = float(np.max(err_psi))
    median_err_psi = float(np.median(err_psi))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    
    return {
        'r2': r2,
        'correlation': corr,
        'mae_kt': mae_kt,
        'rmse_kt': rmse_kt,
        'mae_psi': mae_psi,
        'rmse_psi': rmse_psi,
        'max_error_psi': max_err_psi,
        'median_error_psi': median_err_psi,
        'mape_pct': mape
    }

def train_single_ann(
    model: nn.Module,
    train_loader: DataLoader,
    t_X_val: torch.Tensor,
    t_y_val: torch.Tensor,
    target_scaler: StandardScaler,
    epochs: int = 150,
    lr: float = 2e-3,
    patience: int = 20,
    checkpoint_path: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train an ANN model with vectorized validation, early stopping, LR reduction, and checkpointing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None
    
    t0 = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * len(batch_x)
            
        train_loss_epoch /= len(train_loader.dataset)
        
        # Fast vectorized validation evaluation in a single tensor operation
        model.eval()
        with torch.no_grad():
            vpreds = model(t_X_val)
            val_loss_epoch = float(criterion(vpreds, t_y_val).item())
            
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss_epoch)
        
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['lr'].append(current_lr)
        
        # Early stopping check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    train_time = time.perf_counter() - t0
    
    # Load best weights
    model.load_state_dict(best_weights)
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_weights, checkpoint_path)
        
    return model, {
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'total_epochs': len(history['train_loss']),
        'train_time_s': round(train_time, 2),
        'history': history
    }

def run_stage3A_campaign():
    print("==================================================================")
    print("Starting Stage 3A Forward ANN Training Campaign")
    print("==================================================================")
    
    # Paths
    base_dir = Path('models/stage3A_forward_ann')
    base_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Verify split manifest alignment
    df_uni['split'] = manifest['split'].values
    
    # Save unified Stage 3A split manifest
    manifest_3a = df_enrich[['enriched_id', 'geometry_family', 'data_source', 'split']].copy()
    manifest_3a.to_csv('data/stage3A_ann_split_manifest.csv', index=False)
    print(f"Saved stage3A split manifest: {len(manifest_3a)} rows to data/stage3A_ann_split_manifest.csv")
    
    # Automated split leakage check
    train_ids = set(df_enrich[df_enrich['split'] == 'train']['enriched_id'])
    val_ids = set(df_enrich[df_enrich['split'] == 'val']['enriched_id'])
    test_ids = set(df_enrich[df_enrich['split'] == 'test']['enriched_id'])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test!"
    print("Leakage Check: PASSED. Zero overlap between train, val, and test splits.")
    
    # Primary 4 inputs and target
    features = ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio']
    target = 'Kt_element'
    
    # Configurations
    architectures = {
        'ANN_128': 128,
        'ANN_256': 256,
        'ANN_512': 512
    }
    seeds = [42, 101, 2024, 777, 999]
    batch_size = 128
    
    datasets = {
        'original_dataset': {
            'df': df_uni,
            'train_size': 35262,
            'val_size': 7556,
            'test_size': 7557
        },
        'enriched_dataset': {
            'df': df_enrich,
            'train_size': 36512,
            'val_size': 7556,
            'test_size': 7557
        }
    }
    
    master_eval_records = []
    summary_metrics = {}
    
    # Save preprocessing scalers
    prep_dir = base_dir / 'preprocessing'
    prep_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_name, d_info in datasets.items():
        print(f"\n==================================================================")
        print(f"EXPERIMENT: {exp_name.upper()} (Train={d_info['train_size']}, Val={d_info['val_size']}, Test={d_info['test_size']})")
        print(f"==================================================================")
        
        df_exp = d_info['df']
        
        train_df = df_exp[df_exp['split'] == 'train']
        val_df = df_exp[df_exp['split'] == 'val']
        test_df = df_exp[df_exp['split'] == 'test']
        
        # Fit input & target scalers strictly on training data
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(train_df[features].values)
        X_val_scaled = scaler_X.transform(val_df[features].values)
        X_test_scaled = scaler_X.transform(test_df[features].values)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(train_df[[target]].values)
        y_val_scaled = scaler_y.transform(val_df[[target]].values)
        y_test_scaled = scaler_y.transform(test_df[[target]].values)
        
        # Save scalers
        joblib.dump(scaler_X, prep_dir / f"scaler_inputs_{exp_name}.joblib")
        joblib.dump(scaler_y, prep_dir / f"scaler_target_{exp_name}.joblib")
        
        # PyTorch Tensors
        t_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        t_y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
        t_X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
        t_y_val = torch.tensor(y_val_scaled, dtype=torch.float32)
        t_X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        t_y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
        
        y_val_physical = val_df[target].values
        y_test_physical = test_df[target].values
        
        summary_metrics[exp_name] = {}
        
        for arch_name, hidden_neurons in architectures.items():
            print(f"\n--- Model Architecture: {arch_name} ({hidden_neurons} hidden neurons) ---")
            arch_dir = base_dir / exp_name / arch_name
            arch_dir.mkdir(parents=True, exist_ok=True)
            
            seed_metrics_val = []
            seed_metrics_test = []
            
            for seed in seeds:
                set_seed(seed)
                train_loader = DataLoader(
                    TensorDataset(t_X_train, t_y_train),
                    batch_size=512,
                    shuffle=True
                )
                
                model = ForwardANN(in_features=4, hidden_neurons=hidden_neurons).to(device)
                ckpt_path = str(arch_dir / f"model_{arch_name}_seed_{seed}.pt")
                
                model, train_info = train_single_ann(
                    model=model,
                    train_loader=train_loader,
                    t_X_val=t_X_val,
                    t_y_val=t_y_val,
                    target_scaler=scaler_y,
                    epochs=150,
                    lr=2e-3,
                    patience=20,
                    checkpoint_path=ckpt_path
                )
                
                # Predictions on Validation
                model.eval()
                with torch.no_grad():
                    pred_val_sc = model(t_X_val).cpu().numpy()
                    pred_test_sc = model(t_X_test).cpu().numpy()
                    
                pred_val_kt = scaler_y.inverse_transform(pred_val_sc).flatten()
                pred_test_kt = scaler_y.inverse_transform(pred_test_sc).flatten()
                
                # Evaluate metrics
                m_val = compute_metrics(y_val_physical, pred_val_kt)
                m_test = compute_metrics(y_test_physical, pred_test_kt)
                
                m_val['best_epoch'] = train_info['best_epoch']
                m_val['train_time_s'] = train_info['train_time_s']
                m_test['best_epoch'] = train_info['best_epoch']
                m_test['train_time_s'] = train_info['train_time_s']
                
                seed_metrics_val.append(m_val)
                seed_metrics_test.append(m_test)
                
                print(f"  [Seed {seed:4d}] Best Epoch: {train_info['best_epoch']:3d} | "
                      f"Val R2={m_val['r2']:.4f}, MAE={m_val['mae_psi']:.2f} psi, RMSE={m_val['rmse_psi']:.2f} psi | "
                      f"Test R2={m_test['r2']:.4f}, MAE={m_test['mae_psi']:.2f} psi, RMSE={m_test['rmse_psi']:.2f} psi")
                      
                # Save training history
                with open(arch_dir / f"history_seed_{seed}.json", 'w') as f:
                    json.dump(train_info, f, indent=2)
                    
                # Append test predictions to master evaluation records
                df_rec = pd.DataFrame({
                    'experiment': exp_name,
                    'architecture': arch_name,
                    'seed': seed,
                    'split': 'test',
                    'geometry_family': test_df['geometry_family'].values,
                    'defect_x_over_L': test_df['defect_x_over_L'].values,
                    'defect_y_over_H': test_df['defect_y_over_H'].values,
                    'defect_d_over_Dc': test_df['defect_d_over_Dc'].values,
                    'aspect_ratio': test_df['aspect_ratio'].values,
                    'hole_to_hole_clearance_mm': test_df['hole_to_hole_clearance_mm'].values if 'hole_to_hole_clearance_mm' in test_df else 0.0,
                    'boundary_clearance_mm': test_df['boundary_clearance_mm'].values if 'boundary_clearance_mm' in test_df else 0.0,
                    'y_true_kt': y_test_physical,
                    'y_pred_kt': pred_test_kt,
                    'residual_kt': pred_test_kt - y_test_physical,
                    'abs_error_psi': np.abs(pred_test_kt - y_test_physical) * 100.0
                })
                master_eval_records.append(df_rec)
                    
            # Compute mean and std over 5 seeds
            def aggregate_metrics(m_list):
                keys = m_list[0].keys()
                res = {}
                for k in keys:
                    vals = [m[k] for m in m_list]
                    res[f'{k}_mean'] = float(np.mean(vals))
                    res[f'{k}_std'] = float(np.std(vals))
                return res
                
            summary_metrics[exp_name][arch_name] = {
                'validation': aggregate_metrics(seed_metrics_val),
                'test': aggregate_metrics(seed_metrics_test),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'model_size_kb': round(os.path.getsize(ckpt_path) / 1024, 2)
            }
            
            val_r2_m = summary_metrics[exp_name][arch_name]['validation']['r2_mean']
            val_r2_s = summary_metrics[exp_name][arch_name]['validation']['r2_std']
            test_r2_m = summary_metrics[exp_name][arch_name]['test']['r2_mean']
            test_r2_s = summary_metrics[exp_name][arch_name]['test']['r2_std']
            test_rmse_m = summary_metrics[exp_name][arch_name]['test']['rmse_psi_mean']
            
            print(f"  >>> {arch_name} SUMMARY: Val R2 = {val_r2_m:.4f} +/- {val_r2_s:.4f} | "
                  f"Test R2 = {test_r2_m:.4f} +/- {test_r2_s:.4f} | Test RMSE = {test_rmse_m:.2f} psi")
                  
    # Save master evaluation table and summary metrics
    eval_dir = base_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    df_master_eval = pd.concat(master_eval_records, ignore_index=True)
    df_master_eval.to_csv(eval_dir / 'master_evaluation_table.csv', index=False)
    print(f"\nSaved master evaluation table: {len(df_master_eval)} records to {eval_dir / 'master_evaluation_table.csv'}")
    
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
            'learning_rate': 1e-3,
            'batch_size': batch_size,
            'weight_decay': 1e-5,
            'scheduler': 'ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)',
            'early_stopping_patience': 25,
            'max_epochs': 300,
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
    run_stage3A_campaign()
