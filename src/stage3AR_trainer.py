"""
Stage 3A-R Comprehensive Model Training and Experiment Orchestrator
Executes:
- Phase B: Baseline reproduction
- Phase D: Physical feature engineering ablation (Sets A, B, C, D, E)
- Phase E: Deeper ANN architectures (D1, D2, D3, ResMLP)
- Phase F: Improved loss functions (MSE, Huber, WeightedMSE)
- Phase G: Multi-output regional stress prediction
- Phase H: Conventional ML benchmarks (HistGradientBoosting, ExtraTrees, RandomForest)
- Phase I: Evaluation on both Historical (7,557) and Confirmatory (500) test sets
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor

from src.stage3AR_models import (
    ShallowANN, DeepMLP_D1, DeepMLP_D2, DeepMLP_D3, ResMLP,
    MultiOutputANN, PhysicallyWeightedMSELoss
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true_kt: np.ndarray, y_pred_kt: np.ndarray) -> Dict[str, float]:
    y_true_psi = y_true_kt * 100.0
    y_pred_psi = y_pred_kt * 100.0
    err_psi = np.abs(y_true_psi - y_pred_psi)
    
    r2 = float(r2_score(y_true_kt, y_pred_kt))
    corr = float(np.corrcoef(y_true_kt, y_pred_kt)[0, 1]) if len(y_true_kt) > 1 else 1.0
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all physics-informed geometric interaction features."""
    df = df.copy()
    L = df['plate_length_mm']
    H = df['plate_height_mm']
    Dc = df['central_hole_diameter_mm']
    D = df['defect_diameter_mm']
    X = df['defect_center_x_mm']
    Y = df['defect_center_y_mm']
    Xc = L / 2.0
    Yc = H / 2.0
    
    # Base dimensionless
    df['defect_x_over_L'] = X / L
    df['defect_y_over_H'] = Y / H
    df['defect_d_over_Dc'] = D / Dc
    df['defect_d_over_H'] = D / H
    df['aspect_ratio'] = L / H
    
    # Hole interaction features
    d_centres = np.hypot(X - Xc, Y - Yc)
    df['center_separation_mm'] = d_centres
    df['center_separation_over_Dc'] = d_centres / Dc
    df['center_separation_over_H'] = d_centres / H
    
    ligament_clearance = d_centres - (Dc + D) / 2.0
    df['ligament_clearance_mm'] = ligament_clearance
    df['ligament_clearance_over_Dc'] = ligament_clearance / Dc
    df['ligament_clearance_over_H'] = ligament_clearance / H
    
    # Boundary clearance features
    c_left = X - D / 2.0
    c_right = L - X - D / 2.0
    c_bot = Y - D / 2.0
    c_top = H - Y - D / 2.0
    min_bnd = np.minimum(np.minimum(c_left, c_right), np.minimum(c_bot, c_top))
    df['min_boundary_clearance_mm'] = min_bnd
    df['min_boundary_clearance_over_H'] = min_bnd / H
    df['min_boundary_clearance_over_D'] = min_bnd / D
    
    # Relative directional features
    df['rel_dx_over_Dc'] = (X - Xc) / Dc
    df['rel_dy_over_Dc'] = (Y - Yc) / Dc
    df['rel_dx_over_L'] = (X - Xc) / L
    df['rel_dy_over_H'] = (Y - Yc) / H
    
    return df

FEATURE_SETS = {
    'Set_A_Original': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio'],
    'Set_B_Hole_Interaction': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'ligament_clearance_over_Dc'],
    'Set_C_Boundary_Clearance': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'min_boundary_clearance_over_H'],
    'Set_D_Interaction_Boundary': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'ligament_clearance_over_Dc', 'min_boundary_clearance_over_H'],
    'Set_E_Compact_Engineered': ['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc', 'aspect_ratio', 'ligament_clearance_over_Dc', 'min_boundary_clearance_over_H', 'rel_dx_over_Dc', 'rel_dy_over_Dc', 'center_separation_over_Dc']
}

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    t_X_val: torch.Tensor,
    t_y_val: torch.Tensor,
    loss_fn_name: str = 'mse',
    t_weights_train: torch.Tensor = None,
    epochs: int = 150,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    scheduler_type: str = 'plateau'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train single neural network with early stopping and learning rate scheduling."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
    if loss_fn_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn_name == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    elif loss_fn_name == 'weighted_mse':
        criterion = PhysicallyWeightedMSELoss()
    else:
        criterion = nn.MSELoss()
        
    val_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_weights = None
    
    t0 = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            if loss_fn_name == 'weighted_mse':
                bx, by, bw = batch
                preds = model(bx)
                loss = criterion(preds, by, bw)
            else:
                bx, by = batch[:2]
                preds = model(bx)
                loss = criterion(preds, by)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
            
        train_loss /= len(train_loader.dataset)
        
        # Vectorized validation loss
        model.eval()
        with torch.no_grad():
            vpreds = model(t_X_val)
            val_loss = float(val_criterion(vpreds, t_y_val).item())
            
        if scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    duration = time.perf_counter() - t0
    return model, {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_time_s': duration}

def run_stage3AR_pipeline():
    print("==================================================================")
    print("STARTING STAGE 3A-R COMPREHENSIVE SURROGATE PIPELINE")
    print("==================================================================")
    
    base_dir = Path('models/stage3AR_forward_improved')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading Datasets and Manifests...")
    df_enr = pd.read_csv('data/enriched_dataset_51625.csv')
    df_50 = pd.read_csv('data/unified_dataset_50375.csv')
    df_adapt = pd.read_csv('data/adaptive_enrichment_stage2V/master_adaptive_simulations_1250.csv')
    df_conf = pd.read_csv('data/stage3AR_confirmatory_test/confirmatory_dataset_500.csv')
    
    # Merge regional stresses into df_enr
    df_enr['max_vm_center_psi'] = list(df_50['max_vm_center_psi'].values) + list(df_adapt['max_vm_center_psi'].values)
    df_enr['max_vm_defect_psi'] = list(df_50['max_vm_defect_psi'].values) + list(df_adapt['max_vm_defect_psi'].values)
    df_enr['Kt_center'] = df_enr['max_vm_center_psi'] / 100.0
    df_enr['Kt_defect'] = df_enr['max_vm_defect_psi'] / 100.0
    
    df_conf['Kt_center'] = df_conf['max_vm_center_psi'] / 100.0
    df_conf['Kt_defect'] = df_conf['max_vm_defect_psi'] / 100.0
    
    # Feature Engineering
    print("Engineering Physics-Informed Geometric Features...")
    df_enr = engineer_features(df_enr)
    df_conf = engineer_features(df_conf)
    
    # Physical Weights calculation for Weighted Loss (Experiment L3)
    # w = 1.0 + 2.0*exp(-c_hole/15) + 1.0*exp(-c_bnd/15)
    w_unnorm = 1.0 + 2.0 * np.exp(-df_enr['ligament_clearance_mm'] / 15.0) + 1.0 * np.exp(-df_enr['min_boundary_clearance_mm'] / 15.0)
    w_train_mean = np.mean(w_unnorm[df_enr['split'] == 'train'])
    df_enr['sample_weight'] = w_unnorm / w_train_mean
    
    train_df = df_enr[df_enr['split'] == 'train'].copy()
    val_df = df_enr[df_enr['split'] == 'val'].copy()
    hist_test_df = df_enr[df_enr['split'] == 'test'].copy()
    
    print(f"Data Partition: Train={len(train_df)}, Val={len(val_df)}, Historical Test={len(hist_test_df)}, Confirmatory Test={len(df_conf)}")
    
    seeds = [42, 101, 2024, 777, 999]
    batch_size = 512
    target = 'Kt_element'
    
    # Storage for all evaluation results
    experiment_results = {}
    
    # -------------------------------------------------------------
    # EXPERIMENT 1: Phase D - Feature Engineering Screening on Validation
    # (Testing Sets A, B, C, D, E with ANN-512)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("EXPERIMENT 1: Phase D - Physical Feature Engineering Screening (ANN-512)")
    print("-------------------------------------------------------------")
    
    feat_screening = {}
    
    for f_name, feat_cols in FEATURE_SETS.items():
        print(f"\nEvaluating Feature Set: {f_name} ({len(feat_cols)} features)")
        scaler_X = StandardScaler()
        X_train_sc = scaler_X.fit_transform(train_df[feat_cols].values)
        X_val_sc = scaler_X.transform(val_df[feat_cols].values)
        
        scaler_y = StandardScaler()
        y_train_sc = scaler_y.fit_transform(train_df[[target]].values)
        y_val_sc = scaler_y.transform(val_df[[target]].values)
        
        t_X_train = torch.tensor(X_train_sc, dtype=torch.float32)
        t_y_train = torch.tensor(y_train_sc, dtype=torch.float32)
        t_X_val = torch.tensor(X_val_sc, dtype=torch.float32)
        t_y_val = torch.tensor(y_val_sc, dtype=torch.float32)
        
        val_seed_metrics = []
        for s in seeds:
            set_seed(s)
            train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)
            model = ShallowANN(in_features=len(feat_cols), hidden_neurons=512)
            model, _ = train_single_model(model, train_loader, t_X_val, t_y_val, epochs=150, lr=2e-3, patience=20)
            
            model.eval()
            with torch.no_grad():
                preds_val_sc = model(t_X_val).numpy()
            preds_val_kt = scaler_y.inverse_transform(preds_val_sc).flatten()
            val_seed_metrics.append(compute_metrics(val_df[target].values, preds_val_kt))
            
        r2_m = float(np.mean([m['r2'] for m in val_seed_metrics]))
        mae_m = float(np.mean([m['mae_psi'] for m in val_seed_metrics]))
        rmse_m = float(np.mean([m['rmse_psi'] for m in val_seed_metrics]))
        
        feat_screening[f_name] = {
            'features': feat_cols,
            'val_r2_mean': r2_m,
            'val_mae_psi_mean': mae_m,
            'val_rmse_psi_mean': rmse_m,
            'seeds': val_seed_metrics
        }
        print(f"  >>> {f_name:25s}: Val R2 = {r2_m:.4f} | Val MAE = {mae_m:.2f} psi | Val RMSE = {rmse_m:.2f} psi")
        
    experiment_results['phaseD_feature_screening'] = feat_screening
    
    # Select best feature set based purely on validation performance
    best_feat_name = max(feat_screening.keys(), key=lambda k: feat_screening[k]['val_r2_mean'])
    best_features = FEATURE_SETS[best_feat_name]
    print(f"\nSelected Best Feature Set (based strictly on Validation): {best_feat_name}")
    
    # -------------------------------------------------------------
    # EXPERIMENT 2: Phase E - Deeper ANN Architectures
    # (Testing ANN-128, 256, 512, D1, D2, D3, ResMLP on best features)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print(f"EXPERIMENT 2: Phase E - Deep Network Exploration (using {best_feat_name})")
    print("-------------------------------------------------------------")
    
    # Fit scalers on selected best features
    scaler_X_best = StandardScaler()
    X_train_best_sc = scaler_X_best.fit_transform(train_df[best_features].values)
    X_val_best_sc = scaler_X_best.transform(val_df[best_features].values)
    X_hist_test_sc = scaler_X_best.transform(hist_test_df[best_features].values)
    X_conf_sc = scaler_X_best.transform(df_conf[best_features].values)
    
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(train_df[[target]].values)
    y_val_sc = scaler_y.transform(val_df[[target]].values)
    
    t_X_train_b = torch.tensor(X_train_best_sc, dtype=torch.float32)
    t_y_train_b = torch.tensor(y_train_sc, dtype=torch.float32)
    t_X_val_b = torch.tensor(X_val_best_sc, dtype=torch.float32)
    t_y_val_b = torch.tensor(y_val_sc, dtype=torch.float32)
    t_X_hist_b = torch.tensor(X_hist_test_sc, dtype=torch.float32)
    t_X_conf_b = torch.tensor(X_conf_sc, dtype=torch.float32)
    
    y_val_phys = val_df[target].values
    y_hist_phys = hist_test_df[target].values
    y_conf_phys = df_conf[target].values
    
    in_dim = len(best_features)
    
    arch_constructors = {
        'ANN_128': lambda: ShallowANN(in_dim, 128),
        'ANN_256': lambda: ShallowANN(in_dim, 256),
        'ANN_512': lambda: ShallowANN(in_dim, 512),
        'DeepMLP_D1': lambda: DeepMLP_D1(in_dim),
        'DeepMLP_D2': lambda: DeepMLP_D2(in_dim),
        'DeepMLP_D3': lambda: DeepMLP_D3(in_dim),
        'ResMLP': lambda: ResMLP(in_dim)
    }
    
    arch_results = {}
    saved_models = {}
    
    for a_name, ctor in arch_constructors.items():
        print(f"\nTraining Architecture: {a_name} ...")
        a_dir = base_dir / 'architectures' / a_name
        a_dir.mkdir(parents=True, exist_ok=True)
        
        val_seeds = []
        hist_seeds = []
        conf_seeds = []
        
        for s in seeds:
            set_seed(s)
            train_loader = DataLoader(TensorDataset(t_X_train_b, t_y_train_b), batch_size=batch_size, shuffle=True)
            model = ctor()
            model, t_info = train_single_model(model, train_loader, t_X_val_b, t_y_val_b, epochs=150, lr=2e-3, patience=20)
            
            # Save checkpoint
            ckpt_p = a_dir / f"model_{a_name}_seed_{s}.pt"
            torch.save(model.state_dict(), ckpt_p)
            
            model.eval()
            with torch.no_grad():
                pred_v = scaler_y.inverse_transform(model(t_X_val_b).numpy()).flatten()
                pred_h = scaler_y.inverse_transform(model(t_X_hist_b).numpy()).flatten()
                pred_c = scaler_y.inverse_transform(model(t_X_conf_b).numpy()).flatten()
                
            val_seeds.append(compute_metrics(y_val_phys, pred_v))
            hist_seeds.append(compute_metrics(y_hist_phys, pred_h))
            conf_seeds.append(compute_metrics(y_conf_phys, pred_c))
            
        def agg(m_list):
            return {
                'r2_mean': float(np.mean([m['r2'] for m in m_list])),
                'r2_std': float(np.std([m['r2'] for m in m_list])),
                'mae_psi_mean': float(np.mean([m['mae_psi'] for m in m_list])),
                'mae_psi_std': float(np.std([m['mae_psi'] for m in m_list])),
                'rmse_psi_mean': float(np.mean([m['rmse_psi'] for m in m_list])),
                'rmse_psi_std': float(np.std([m['rmse_psi'] for m in m_list])),
                'max_error_psi_mean': float(np.mean([m['max_error_psi'] for m in m_list]))
            }
            
        arch_results[a_name] = {
            'param_count': sum(p.numel() for p in model.parameters()),
            'val': agg(val_seeds),
            'historical_test': agg(hist_seeds),
            'confirmatory_test': agg(conf_seeds)
        }
        
        print(f"  >>> {a_name:12s} ({arch_results[a_name]['param_count']:6d} params): "
              f"Val R2={arch_results[a_name]['val']['r2_mean']:.4f} | "
              f"HistTest R2={arch_results[a_name]['historical_test']['r2_mean']:.4f}, RMSE={arch_results[a_name]['historical_test']['rmse_psi_mean']:.2f} psi | "
              f"ConfTest R2={arch_results[a_name]['confirmatory_test']['r2_mean']:.4f}, RMSE={arch_results[a_name]['confirmatory_test']['rmse_psi_mean']:.2f} psi")
              
    experiment_results['phaseE_architecture_results'] = arch_results
    
    # -------------------------------------------------------------
    # EXPERIMENT 3: Phase F - Loss Functions & Optimization (on best deep model)
    # (Testing MSE vs Huber vs WeightedMSE on DeepMLP_D2)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("EXPERIMENT 3: Phase F - Loss Function Comparisons (on DeepMLP_D2)")
    print("-------------------------------------------------------------")
    
    t_weights_train = torch.tensor(train_df['sample_weight'].values, dtype=torch.float32).unsqueeze(1)
    losses_to_test = ['mse', 'huber', 'weighted_mse']
    loss_results = {}
    
    for l_name in losses_to_test:
        print(f"\nTraining with Loss Function: {l_name} ...")
        l_dir = base_dir / 'loss_experiments' / l_name
        l_dir.mkdir(parents=True, exist_ok=True)
        
        val_seeds = []
        hist_seeds = []
        conf_seeds = []
        
        for s in seeds:
            set_seed(s)
            if l_name == 'weighted_mse':
                train_loader = DataLoader(TensorDataset(t_X_train_b, t_y_train_b, t_weights_train), batch_size=batch_size, shuffle=True)
            else:
                train_loader = DataLoader(TensorDataset(t_X_train_b, t_y_train_b), batch_size=batch_size, shuffle=True)
                
            model = DeepMLP_D2(in_dim)
            model, _ = train_single_model(model, train_loader, t_X_val_b, t_y_val_b, loss_fn_name=l_name, epochs=150, lr=2e-3, patience=20)
            
            torch.save(model.state_dict(), l_dir / f"model_d2_{l_name}_seed_{s}.pt")
            
            model.eval()
            with torch.no_grad():
                pred_v = scaler_y.inverse_transform(model(t_X_val_b).numpy()).flatten()
                pred_h = scaler_y.inverse_transform(model(t_X_hist_b).numpy()).flatten()
                pred_c = scaler_y.inverse_transform(model(t_X_conf_b).numpy()).flatten()
                
            val_seeds.append(compute_metrics(y_val_phys, pred_v))
            hist_seeds.append(compute_metrics(y_hist_phys, pred_h))
            conf_seeds.append(compute_metrics(y_conf_phys, pred_c))
            
        loss_results[l_name] = {
            'val': agg(val_seeds),
            'historical_test': agg(hist_seeds),
            'confirmatory_test': agg(conf_seeds)
        }
        print(f"  >>> Loss {l_name:12s}: Val R2={loss_results[l_name]['val']['r2_mean']:.4f} | "
              f"HistTest R2={loss_results[l_name]['historical_test']['r2_mean']:.4f}, RMSE={loss_results[l_name]['historical_test']['rmse_psi_mean']:.2f} psi | "
              f"ConfTest R2={loss_results[l_name]['confirmatory_test']['r2_mean']:.4f}, RMSE={loss_results[l_name]['confirmatory_test']['rmse_psi_mean']:.2f} psi")
              
    experiment_results['phaseF_loss_results'] = loss_results
    
    # -------------------------------------------------------------
    # EXPERIMENT 4: Phase G - Multi-Output Regional Stress Prediction
    # (Predicting Central-hole Kt and Defect-hole Kt, then taking max)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("EXPERIMENT 4: Phase G - Multi-Output Regional Stress Model")
    print("-------------------------------------------------------------")
    
    # Scale regional targets
    scaler_yc = StandardScaler()
    scaler_yd = StandardScaler()
    
    yc_train_sc = scaler_yc.fit_transform(train_df[['Kt_center']].values)
    yd_train_sc = scaler_yd.fit_transform(train_df[['Kt_defect']].values)
    y_reg_train_sc = np.hstack([yc_train_sc, yd_train_sc])
    
    yc_val_sc = scaler_yc.transform(val_df[['Kt_center']].values)
    yd_val_sc = scaler_yd.transform(val_df[['Kt_defect']].values)
    y_reg_val_sc = np.hstack([yc_val_sc, yd_val_sc])
    
    t_y_reg_train = torch.tensor(y_reg_train_sc, dtype=torch.float32)
    t_y_reg_val = torch.tensor(y_reg_val_sc, dtype=torch.float32)
    
    mo_dir = base_dir / 'multi_output'
    mo_dir.mkdir(parents=True, exist_ok=True)
    
    mo_val_seeds = []
    mo_hist_seeds = []
    mo_conf_seeds = []
    mo_reg_breakdown = []
    
    for s in seeds:
        set_seed(s)
        train_loader = DataLoader(TensorDataset(t_X_train_b, t_y_reg_train), batch_size=batch_size, shuffle=True)
        mo_model = MultiOutputANN(in_dim)
        
        optimizer = torch.optim.AdamW(mo_model.parameters(), lr=2e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.MSELoss()
        
        best_v_loss = float('inf')
        best_mo_weights = None
        
        for ep in range(1, 151):
            mo_model.train()
            t_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred_c, pred_d = mo_model(bx)
                loss_c = criterion(pred_c, by[:, 0:1])
                loss_d = criterion(pred_d, by[:, 1:2])
                loss = loss_c + loss_d
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * len(bx)
                
            mo_model.eval()
            with torch.no_grad():
                vc, vd = mo_model(t_X_val_b)
                v_loss = float((criterion(vc, t_y_reg_val[:, 0:1]) + criterion(vd, t_y_reg_val[:, 1:2])).item())
                
            scheduler.step(v_loss)
            if v_loss < best_v_loss - 1e-5:
                best_v_loss = v_loss
                best_mo_weights = {k: v.clone() for k, v in mo_model.state_dict().items()}
                
        if best_mo_weights is not None:
            mo_model.load_state_dict(best_mo_weights)
            
        torch.save(mo_model.state_dict(), mo_dir / f"multi_output_seed_{s}.pt")
        
        # Predictions & Derived Global Maximum
        mo_model.eval()
        with torch.no_grad():
            vc, vd = mo_model(t_X_val_b)
            hc, hd = mo_model(t_X_hist_b)
            cc, cd = mo_model(t_X_conf_b)
            
        pred_vc_kt = scaler_yc.inverse_transform(vc.numpy()).flatten()
        pred_vd_kt = scaler_yd.inverse_transform(vd.numpy()).flatten()
        pred_v_global = np.maximum(pred_vc_kt, pred_vd_kt)
        
        pred_hc_kt = scaler_yc.inverse_transform(hc.numpy()).flatten()
        pred_hd_kt = scaler_yd.inverse_transform(hd.numpy()).flatten()
        pred_h_global = np.maximum(pred_hc_kt, pred_hd_kt)
        
        pred_cc_kt = scaler_yc.inverse_transform(cc.numpy()).flatten()
        pred_cd_kt = scaler_yd.inverse_transform(cd.numpy()).flatten()
        pred_c_global = np.maximum(pred_cc_kt, pred_cd_kt)
        
        mo_val_seeds.append(compute_metrics(y_val_phys, pred_v_global))
        mo_hist_seeds.append(compute_metrics(y_hist_phys, pred_h_global))
        mo_conf_seeds.append(compute_metrics(y_conf_phys, pred_c_global))
        
        # Regional metrics on historical test
        mc_c = compute_metrics(hist_test_df['Kt_center'].values, pred_hc_kt)
        mc_d = compute_metrics(hist_test_df['Kt_defect'].values, pred_hd_kt)
        
        # Region classification accuracy: which hole controls?
        true_center_dominant = (hist_test_df['Kt_center'] >= hist_test_df['Kt_defect']).values
        pred_center_dominant = (pred_hc_kt >= pred_hd_kt)
        reg_acc = float(np.mean(true_center_dominant == pred_center_dominant) * 100.0)
        
        mo_reg_breakdown.append({
            'center_hole_mae_psi': mc_c['mae_psi'],
            'center_hole_rmse_psi': mc_c['rmse_psi'],
            'defect_hole_mae_psi': mc_d['mae_psi'],
            'defect_hole_rmse_psi': mc_d['rmse_psi'],
            'region_classification_accuracy_pct': reg_acc
        })
        
    mo_summary = {
        'val': agg(mo_val_seeds),
        'historical_test': agg(mo_hist_seeds),
        'confirmatory_test': agg(mo_conf_seeds),
        'regional_breakdown': {
            'center_hole_mae_psi': float(np.mean([m['center_hole_mae_psi'] for m in mo_reg_breakdown])),
            'center_hole_rmse_psi': float(np.mean([m['center_hole_rmse_psi'] for m in mo_reg_breakdown])),
            'defect_hole_mae_psi': float(np.mean([m['defect_hole_mae_psi'] for m in mo_reg_breakdown])),
            'defect_hole_rmse_psi': float(np.mean([m['defect_hole_rmse_psi'] for m in mo_reg_breakdown])),
            'region_accuracy_pct': float(np.mean([m['region_classification_accuracy_pct'] for m in mo_reg_breakdown]))
        }
    }
    experiment_results['phaseG_multi_output'] = mo_summary
    print(f"  >>> Multi-Output Derived Global Stress: Val R2={mo_summary['val']['r2_mean']:.4f} | "
          f"HistTest R2={mo_summary['historical_test']['r2_mean']:.4f}, RMSE={mo_summary['historical_test']['rmse_psi_mean']:.2f} psi | "
          f"ConfTest R2={mo_summary['confirmatory_test']['r2_mean']:.4f}, RMSE={mo_summary['confirmatory_test']['rmse_psi_mean']:.2f} psi")
    print(f"      Regional Accuracy: {mo_summary['regional_breakdown']['region_accuracy_pct']:.2f}% | "
          f"Center Hole MAE: {mo_summary['regional_breakdown']['center_hole_mae_psi']:.2f} psi | "
          f"Defect Hole MAE: {mo_summary['regional_breakdown']['defect_hole_mae_psi']:.2f} psi")
          
    # -------------------------------------------------------------
    # EXPERIMENT 5: Phase H - Conventional ML Benchmarks
    # (HistGradientBoosting, ExtraTrees, RandomForest on identical data)
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("EXPERIMENT 5: Phase H - Conventional Machine Learning Benchmarks")
    print("-------------------------------------------------------------")
    
    ml_models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=20, learning_rate=0.08, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, min_samples_split=5, n_jobs=-1, random_state=42)
    }
    
    ml_results = {}
    ml_dir = base_dir / 'conventional_ml'
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    for ml_name, clf in ml_models.items():
        print(f"\nTraining Benchmark: {ml_name} ...")
        t0 = time.perf_counter()
        clf.fit(train_df[best_features].values, train_df[target].values)
        dur = time.perf_counter() - t0
        
        joblib.dump(clf, ml_dir / f"model_{ml_name}.joblib")
        
        pred_v_ml = clf.predict(val_df[best_features].values)
        pred_h_ml = clf.predict(hist_test_df[best_features].values)
        pred_c_ml = clf.predict(df_conf[best_features].values)
        
        mv = compute_metrics(y_val_phys, pred_v_ml)
        mh = compute_metrics(y_hist_phys, pred_h_ml)
        mc = compute_metrics(y_conf_phys, pred_c_ml)
        
        ml_results[ml_name] = {
            'train_time_s': dur,
            'val': mv,
            'historical_test': mh,
            'confirmatory_test': mc
        }
        print(f"  >>> {ml_name:22s} ({dur:.1f}s): Val R2={mv['r2']:.4f} | "
              f"HistTest R2={mh['r2']:.4f}, RMSE={mh['rmse_psi']:.2f} psi, MAE={mh['mae_psi']:.2f} psi | "
              f"ConfTest R2={mc['r2']:.4f}, RMSE={mc['rmse_psi']:.2f} psi, MAE={mc['mae_psi']:.2f} psi")
              
    experiment_results['phaseH_conventional_ml'] = ml_results
    
    # Save master experiment results
    with open(base_dir / 'stage3AR_master_experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nSaved master experiment results to {base_dir / 'stage3AR_master_experiment_results.json'}")
    
    # Save Preprocessing Metadata and Scalers
    prep_dir = base_dir / 'preprocessing'
    prep_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X_best, prep_dir / 'scaler_inputs_best_engineered.joblib')
    joblib.dump(scaler_y, prep_dir / 'scaler_target_global.joblib')
    joblib.dump(scaler_yc, prep_dir / 'scaler_target_center.joblib')
    joblib.dump(scaler_yd, prep_dir / 'scaler_target_defect.joblib')
    
    with open(prep_dir / 'engineered_features_metadata.json', 'w') as f:
        json.dump({
            'selected_feature_set': best_feat_name,
            'features': best_features,
            'feature_definitions': {
                'defect_x_over_L': 'X / L',
                'defect_y_over_H': 'Y / H',
                'defect_d_over_Dc': 'D / Dc',
                'aspect_ratio': 'L / H',
                'ligament_clearance_over_Dc': '(d_centres - (Dc+D)/2) / Dc',
                'min_boundary_clearance_over_H': 'min(left, right, bot, top) / H',
                'rel_dx_over_Dc': '(X - Xc) / Dc',
                'rel_dy_over_Dc': '(Y - Yc) / Dc',
                'center_separation_over_Dc': 'sqrt((X-Xc)^2 + (Y-Yc)^2) / Dc'
            }
        }, f, indent=2)
    print("Saved preprocessing artifacts.")

if __name__ == '__main__':
    run_stage3AR_pipeline()

