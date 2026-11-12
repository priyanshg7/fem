"""
Stage 3C: Comprehensive Inverse Machine Learning Engine
Implements:
1. Experiment C1: Reference-Faithful Scalar Inverse Model (Square Plate, Fixed Diameter d/Dc = 0.15)
2. Experiment C2: Extended Scalar-Only Inverse Model (5 Geometries, Variable Diameter)
3. Experiment C3: Geometry- and Diameter-Conditioned Inverse Model (Inputs: Kt_max, L/H, d/Dc)
4. Inverse ANNs: ANN-128, ANN-256, ANN-512 (5 training seeds each)
5. Conventional Multi-Output Regressors: Linear, Ridge, SVR, RandomForest, ExtraTrees, HistGradBoost, KNN
6. 5-Fold Cross-Fitting for Leakage-Free Inverse Stacking
7. Forward-Model-Assisted Inverse Candidate Search (Derivative-free grid search with Stage 3B ensemble)
8. Stress Noise Perturbation Analysis (+-0.5%, +-1%, +-2%, +-5%)
9. Strict Metrics: Normalized errors, Dimensional errors (mm), Euclidean distance, Tolerance percentages (5, 10, 25, 50, 100 mm)
"""

import os
import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Forward predictor from Stage 3B for Forward-Assisted Search
from src.stage3B_inference import ForwardStressEnsemblePredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_inverse_metrics(
    y_true_norm: np.ndarray, # [N, 2] -> [X/L, Y/H]
    y_pred_norm: np.ndarray, # [N, 2] -> [X_hat/L, Y_hat/H]
    L_mm: np.ndarray,        # [N]
    H_mm: np.ndarray         # [N]
) -> Dict[str, Any]:
    # Clip predictions to plate domain [0, 1]
    y_pred_clipped = np.clip(y_pred_norm, 0.0, 1.0)
    
    # Normalized errors
    err_X_norm = np.abs(y_pred_clipped[:, 0] - y_true_norm[:, 0])
    err_Y_norm = np.abs(y_pred_clipped[:, 1] - y_true_norm[:, 1])
    
    r2_X = float(r2_score(y_true_norm[:, 0], y_pred_clipped[:, 0]))
    r2_Y = float(r2_score(y_true_norm[:, 1], y_pred_clipped[:, 1]))
    
    corr_X = float(np.corrcoef(y_true_norm[:, 0], y_pred_clipped[:, 0])[0, 1]) if len(y_true_norm) > 1 else 0.0
    corr_Y = float(np.corrcoef(y_true_norm[:, 1], y_pred_clipped[:, 1])[0, 1]) if len(y_true_norm) > 1 else 0.0
    
    mse_X_norm = float(mean_squared_error(y_true_norm[:, 0], y_pred_clipped[:, 0]))
    mse_Y_norm = float(mean_squared_error(y_true_norm[:, 1], y_pred_clipped[:, 1]))
    
    mae_X_norm = float(mean_absolute_error(y_true_norm[:, 0], y_pred_clipped[:, 0]))
    mae_Y_norm = float(mean_absolute_error(y_true_norm[:, 1], y_pred_clipped[:, 1]))
    
    # Dimensional errors in mm
    pred_X_mm = y_pred_clipped[:, 0] * L_mm
    pred_Y_mm = y_pred_clipped[:, 1] * H_mm
    true_X_mm = y_true_norm[:, 0] * L_mm
    true_Y_mm = y_true_norm[:, 1] * H_mm
    
    err_X_mm = np.abs(pred_X_mm - true_X_mm)
    err_Y_mm = np.abs(pred_Y_mm - true_Y_mm)
    
    # Euclidean localization error in mm
    euc_dist_mm = np.sqrt((pred_X_mm - true_X_mm)**2 + (pred_Y_mm - true_Y_mm)**2)
    norm_dist = np.sqrt(err_X_norm**2 + err_Y_norm**2)
    
    # Tolerances
    pct_5mm = float(np.mean(euc_dist_mm <= 5.0) * 100.0)
    pct_10mm = float(np.mean(euc_dist_mm <= 10.0) * 100.0)
    pct_25mm = float(np.mean(euc_dist_mm <= 25.0) * 100.0)
    pct_50mm = float(np.mean(euc_dist_mm <= 50.0) * 100.0)
    pct_100mm = float(np.mean(euc_dist_mm <= 100.0) * 100.0)
    
    return {
        'r2_X': r2_X, 'r2_Y': r2_Y,
        'corr_X': corr_X, 'corr_Y': corr_Y,
        'mse_X_norm': mse_X_norm, 'mse_Y_norm': mse_Y_norm,
        'mae_X_norm': mae_X_norm, 'mae_Y_norm': mae_Y_norm,
        'mae_X_mm': float(np.mean(err_X_mm)),
        'mae_Y_mm': float(np.mean(err_Y_mm)),
        'median_X_mm': float(np.median(err_X_mm)),
        'median_Y_mm': float(np.median(err_Y_mm)),
        'max_X_mm': float(np.max(err_X_mm)),
        'max_Y_mm': float(np.max(err_Y_mm)),
        'mean_euclidean_mm': float(np.mean(euc_dist_mm)),
        'median_euclidean_mm': float(np.median(euc_dist_mm)),
        'p95_euclidean_mm': float(np.percentile(euc_dist_mm, 95)),
        'max_euclidean_mm': float(np.max(euc_dist_mm)),
        'mean_norm_dist': float(np.mean(norm_dist)),
        'pct_within_5mm': pct_5mm,
        'pct_within_10mm': pct_10mm,
        'pct_within_25mm': pct_25mm,
        'pct_within_50mm': pct_50mm,
        'pct_within_100mm': pct_100mm
    }

# Multi-output Inverse PyTorch Architecture
class InverseANN(nn.Module):
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 2), # Outputs: [X/L, Y/H]
            nn.Sigmoid() # Bounds output directly into normalized [0, 1] range!
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_inverse_ann(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 2e-3
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    t_X = torch.tensor(X_train, dtype=torch.float32, device=device)
    t_y = torch.tensor(y_train, dtype=torch.float32, device=device)
    dataset = TensorDataset(t_X, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    t_X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    best_loss = float('inf')
    best_weights = None
    
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_loss = float(loss_fn(model(t_X_val), t_y_val).item())
            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model

def predict_inverse_ann(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_X = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(t_X).cpu().numpy()
    return preds

class Stage3CInverseEngine:
    def __init__(self):
        self.base_dir = Path("models/stage3C_inverse_localization")
        subdirs = [
            "reference_inverse_ann", "scalar_only_inverse", "geometry_conditioned_inverse",
            "conventional_regression", "inverse_stacking", "forward_assisted_search",
            "preprocessing", "uncertainty", "evaluation", "figures", "reports"
        ]
        for s in subdirs:
            (self.base_dir / s).mkdir(parents=True, exist_ok=True)
            
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self):
        print("Loading datasets for Stage 3C...")
        df_enriched = pd.read_csv("data/enriched_dataset_51625.csv")
        df_manifest = pd.read_csv("data/stage3A_ann_split_manifest.csv")
        df_enriched['split'] = df_manifest['split'].values
        
        # Test sets
        df_3AR = pd.read_csv("data/stage3AR_confirmatory_test/confirmatory_dataset_500.csv")
        df_3B = pd.read_csv("data/stage3B_blind_confirmatory/confirmatory_dataset_500.csv")
        df_3C = pd.read_csv("data/stage3C_inverse_confirmatory/confirmatory_dataset_500.csv")
        
        self.df_train = df_enriched[df_enriched['split'] == 'train'].reset_index(drop=True)
        self.df_val = df_enriched[df_enriched['split'] == 'val'].reset_index(drop=True)
        self.df_test = df_enriched[df_enriched['split'] == 'test'].reset_index(drop=True)
        self.df_3AR = df_3AR.reset_index(drop=True)
        self.df_3B = df_3B.reset_index(drop=True)
        self.df_3C = df_3C.reset_index(drop=True)
        
        print(f"  Train: {len(self.df_train)}, Val: {len(self.df_val)}, Hist Test: {len(self.df_test)}")
        print(f"  Stage 3A-R Conf: {len(self.df_3AR)}, Stage 3B Blind Conf: {len(self.df_3B)}, Stage 3C Dedicated Conf: {len(self.df_3C)}")
        
        # Reference-Faithful subset (C1): Single square plate, fixed diameter d/Dc ~ 0.150
        mask_sq_c1_train = (self.df_train['geometry_family'] == 'square_1000x1000') & (np.abs(self.df_train['defect_d_over_Dc'] - 0.150) <= 0.015)
        mask_sq_c1_val = (self.df_val['geometry_family'] == 'square_1000x1000') & (np.abs(self.df_val['defect_d_over_Dc'] - 0.150) <= 0.015)
        mask_sq_c1_test = (self.df_test['geometry_family'] == 'square_1000x1000') & (np.abs(self.df_test['defect_d_over_Dc'] - 0.150) <= 0.015)
        
        self.df_c1_train = self.df_train[mask_sq_c1_train].reset_index(drop=True)
        self.df_c1_val = self.df_val[mask_sq_c1_val].reset_index(drop=True)
        self.df_c1_test = self.df_test[mask_sq_c1_test].reset_index(drop=True)
        
        mask_c1_conf = (self.df_3C['geometry_family'] == 'square_1000x1000') & (self.df_3C['sample_type'] == 'fixed_diameter_reference')
        self.df_c1_conf = self.df_3C[mask_c1_conf].reset_index(drop=True)
        print(f"  Experiment C1 (Square Plate, Fixed Diameter): Train={len(self.df_c1_train)}, Val={len(self.df_c1_val)}, Test={len(self.df_c1_test)}, Conf={len(self.df_c1_conf)}")

    def run_experiment_C1(self) -> Dict[str, Any]:
        """Experiment C1: Reference-Faithful Scalar Inversion (Square Plate, Fixed Defect Diameter)."""
        print("\n=== RUNNING EXPERIMENT C1: REFERENCE-FAITHFUL INVERSE MODEL ===")
        # Input: [Kt_element], Target: [defect_x_over_L, defect_y_over_H]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.df_c1_train[['Kt_element']].values)
        y_train = self.df_c1_train[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_val = scaler.transform(self.df_c1_val[['Kt_element']].values)
        y_val = self.df_c1_val[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_test = scaler.transform(self.df_c1_test[['Kt_element']].values)
        y_test = self.df_c1_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_c1_test['plate_length_mm'].values
        H_test = self.df_c1_test['plate_height_mm'].values
        
        X_conf = scaler.transform(self.df_c1_conf[['Kt_element']].values)
        y_conf = self.df_c1_conf[['defect_x_over_L', 'defect_y_over_H']].values
        L_conf = self.df_c1_conf['plate_length_mm'].values
        H_conf = self.df_c1_conf['plate_height_mm'].values
        
        joblib.dump(scaler, self.base_dir / "preprocessing/c1_scaler.pkl")
        
        c1_results = {}
        for neurons in [128, 256, 512]:
            seed_runs = []
            for seed in [42, 101, 202, 303, 404]:
                set_seed(seed)
                model = InverseANN(1, neurons)
                fitted = train_inverse_ann(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=256)
                p_test = predict_inverse_ann(fitted, X_test)
                p_conf = predict_inverse_ann(fitted, X_conf)
                
                m_test = compute_inverse_metrics(y_test, p_test, L_test, H_test)
                m_conf = compute_inverse_metrics(y_conf, p_conf, L_conf, H_conf)
                seed_runs.append({'seed': seed, 'test': m_test, 'conf': m_conf, 'pred_test': p_test})
                
            # Average metrics across 5 seeds
            avg_m_test = {k: float(np.mean([r['test'][k] for r in seed_runs])) for k in seed_runs[0]['test']}
            avg_m_conf = {k: float(np.mean([r['conf'][k] for r in seed_runs])) for k in seed_runs[0]['conf']}
            
            c1_results[f"ANN_{neurons}"] = {
                'historical_test': avg_m_test,
                'confirmatory': avg_m_conf,
                'seed_std_euc_mm': float(np.mean([r['test']['mean_euclidean_mm'] for r in seed_runs]))
            }
            print(f"  C1 ANN-{neurons} -> Test Rx: {avg_m_test['corr_X']:.4f}, Ry: {avg_m_test['corr_Y']:.4f}, MSEx: {avg_m_test['mse_X_norm']:.4f}, MSEy: {avg_m_test['mse_Y_norm']:.4f}, Mean Euc Dist: {avg_m_test['mean_euclidean_mm']:.1f} mm")
            
        return c1_results

    def run_experiment_C2(self) -> Dict[str, Any]:
        """Experiment C2: Extended Scalar-Only Inversion (Multi-Geometry, Variable Defect Diameter)."""
        print("\n=== RUNNING EXPERIMENT C2: EXTENDED SCALAR-ONLY INVERSE MODEL ===")
        # Input: [Kt_element], Target: [defect_x_over_L, defect_y_over_H]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.df_train[['Kt_element']].values)
        y_train = self.df_train[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_val = scaler.transform(self.df_val[['Kt_element']].values)
        y_val = self.df_val[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_test = scaler.transform(self.df_test[['Kt_element']].values)
        y_test = self.df_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_test['plate_length_mm'].values
        H_test = self.df_test['plate_height_mm'].values
        
        X_conf = scaler.transform(self.df_3C[['Kt_element']].values)
        y_conf = self.df_3C[['defect_x_over_L', 'defect_y_over_H']].values
        L_conf = self.df_3C['plate_length_mm'].values
        H_conf = self.df_3C['plate_height_mm'].values
        
        joblib.dump(scaler, self.base_dir / "preprocessing/c2_scaler.pkl")
        
        c2_results = {}
        for neurons in [128, 256, 512]:
            set_seed(42)
            model = InverseANN(1, neurons)
            fitted = train_inverse_ann(model, X_train, y_train, X_val, y_val, epochs=35, batch_size=512)
            p_test = predict_inverse_ann(fitted, X_test)
            p_conf = predict_inverse_ann(fitted, X_conf)
            
            m_test = compute_inverse_metrics(y_test, p_test, L_test, H_test)
            m_conf = compute_inverse_metrics(y_conf, p_conf, L_conf, H_conf)
            c2_results[f"ANN_{neurons}"] = {'historical_test': m_test, 'confirmatory': m_conf}
            print(f"  C2 ANN-{neurons} -> Test Rx: {m_test['corr_X']:.4f}, Ry: {m_test['corr_Y']:.4f}, Mean Euc Dist: {m_test['mean_euclidean_mm']:.1f} mm, <=25mm: {m_test['pct_within_25mm']:.1f}%")
            
        return c2_results

    def run_experiment_C3(self) -> Dict[str, Any]:
        """Experiment C3: Geometry- and Diameter-Conditioned Inversion (Inputs: Kt_max, L/H, d/Dc)."""
        print("\n=== RUNNING EXPERIMENT C3: GEOMETRY- AND DIAMETER-CONDITIONED INVERSE MODEL ===")
        # Inputs: [Kt_element, aspect_ratio, defect_d_over_Dc]
        in_cols = ['Kt_element', 'aspect_ratio', 'defect_d_over_Dc']
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.df_train[in_cols].values)
        y_train = self.df_train[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_val = scaler.transform(self.df_val[in_cols].values)
        y_val = self.df_val[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_test = scaler.transform(self.df_test[in_cols].values)
        y_test = self.df_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_test['plate_length_mm'].values
        H_test = self.df_test['plate_height_mm'].values
        
        X_conf = scaler.transform(self.df_3C[in_cols].values)
        y_conf = self.df_3C[['defect_x_over_L', 'defect_y_over_H']].values
        L_conf = self.df_3C['plate_length_mm'].values
        H_conf = self.df_3C['plate_height_mm'].values
        
        joblib.dump(scaler, self.base_dir / "preprocessing/c3_scaler.pkl")
        
        c3_results = {}
        for neurons in [128, 256, 512]:
            set_seed(42)
            model = InverseANN(3, neurons)
            fitted = train_inverse_ann(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=512)
            p_test = predict_inverse_ann(fitted, X_test)
            p_conf = predict_inverse_ann(fitted, X_conf)
            
            m_test = compute_inverse_metrics(y_test, p_test, L_test, H_test)
            m_conf = compute_inverse_metrics(y_conf, p_conf, L_conf, H_conf)
            c3_results[f"ANN_{neurons}"] = {'historical_test': m_test, 'confirmatory': m_conf}
            print(f"  C3 ANN-{neurons} -> Test Rx: {m_test['corr_X']:.4f}, Ry: {m_test['corr_Y']:.4f}, Mean Euc Dist: {m_test['mean_euclidean_mm']:.1f} mm, <=25mm: {m_test['pct_within_25mm']:.1f}%")
            
        return c3_results

    def run_conventional_baselines(self) -> Dict[str, Any]:
        """Phase E: Benchmark Conventional Multi-Output Regressors on Formulation C3."""
        print("\n=== RUNNING PHASE E: CONVENTIONAL MULTI-OUTPUT REGRESSORS ===")
        in_cols = ['Kt_element', 'aspect_ratio', 'defect_d_over_Dc']
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.df_train[in_cols].values)
        y_train = self.df_train[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_test = scaler.transform(self.df_test[in_cols].values)
        y_test = self.df_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_test['plate_length_mm'].values
        H_test = self.df_test['plate_height_mm'].values
        
        X_conf = scaler.transform(self.df_3C[in_cols].values)
        y_conf = self.df_3C[['defect_x_over_L', 'defect_y_over_H']].values
        L_conf = self.df_3C['plate_length_mm'].values
        H_conf = self.df_3C['plate_height_mm'].values
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=10.0),
            'KNN': KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=-1),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_split=4, n_jobs=-1, random_state=42),
            'HistGradientBoosting': MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=42))
        }
        
        conv_results = {}
        for name, m in models.items():
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            p_test = m.predict(X_test)
            p_conf = m.predict(X_conf)
            elapsed = time.perf_counter() - t0
            
            m_test = compute_inverse_metrics(y_test, p_test, L_test, H_test)
            m_conf = compute_inverse_metrics(y_conf, p_conf, L_conf, H_conf)
            conv_results[name] = {'historical_test': m_test, 'confirmatory': m_conf, 'fit_time_s': elapsed}
            print(f"  {name:<22} -> Test Rx: {m_test['corr_X']:.4f}, Ry: {m_test['corr_Y']:.4f}, Mean Euc: {m_test['mean_euclidean_mm']:.1f} mm, <=25mm: {m_test['pct_within_25mm']:.1f}%")
            joblib.dump(m, self.base_dir / f"conventional_regression/{name}.pkl")
            
        return conv_results

    def run_inverse_stacking(self) -> Dict[str, Any]:
        """Phase F: Inverse Stacked Ensemble Models via 5-Fold OOF Predictions."""
        print("\n=== RUNNING PHASE F: INVERSE STACKED ENSEMBLES ===")
        in_cols = ['Kt_element', 'aspect_ratio', 'defect_d_over_Dc']
        raw_train_X = self.df_train[in_cols].values
        raw_train_y = self.df_train[['defect_x_over_L', 'defect_y_over_H']].values
        
        raw_test_X = self.df_test[in_cols].values
        y_test = self.df_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_test['plate_length_mm'].values
        H_test = self.df_test['plate_height_mm'].values
        
        raw_conf_X = self.df_3C[in_cols].values
        y_conf = self.df_3C[['defect_x_over_L', 'defect_y_over_H']].values
        L_conf = self.df_3C['plate_length_mm'].values
        H_conf = self.df_3C['plate_height_mm'].values
        
        # 4 Diverse Base Learners: ANN-512, RandomForest, ExtraTrees, HistGradientBoosting
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        N_trn = len(raw_train_X)
        N_tst = len(raw_test_X)
        N_cnf = len(raw_conf_X)
        
        # 4 models * 2 outputs = 8 meta-features
        oof_matrix = np.zeros((N_trn, 8), dtype=np.float32)
        test_folds = np.zeros((5, N_tst, 8), dtype=np.float32)
        conf_folds = np.zeros((5, N_cnf, 8), dtype=np.float32)
        
        for fold, (trn_idx, val_idx) in enumerate(kf.split(raw_train_X)):
            fold_scaler = StandardScaler()
            X_trn = fold_scaler.fit_transform(raw_train_X[trn_idx])
            y_trn = raw_train_y[trn_idx]
            
            X_val = fold_scaler.transform(raw_train_X[val_idx])
            X_tst = fold_scaler.transform(raw_test_X)
            X_cnf = fold_scaler.transform(raw_conf_X)
            
            # Base 1: ANN-512
            ann = InverseANN(3, 512)
            ann = train_inverse_ann(ann, X_trn, y_trn, X_val, raw_train_y[val_idx], epochs=30, batch_size=512)
            oof_matrix[val_idx, 0:2] = predict_inverse_ann(ann, X_val)
            test_folds[fold, :, 0:2] = predict_inverse_ann(ann, X_tst)
            conf_folds[fold, :, 0:2] = predict_inverse_ann(ann, X_cnf)
            
            # Base 2: RandomForest
            rf = RandomForestRegressor(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42+fold)
            rf.fit(X_trn, y_trn)
            oof_matrix[val_idx, 2:4] = rf.predict(X_val)
            test_folds[fold, :, 2:4] = rf.predict(X_tst)
            conf_folds[fold, :, 2:4] = rf.predict(X_cnf)
            
            # Base 3: ExtraTrees
            et = ExtraTreesRegressor(n_estimators=50, max_depth=16, n_jobs=-1, random_state=42+fold)
            et.fit(X_trn, y_trn)
            oof_matrix[val_idx, 4:6] = et.predict(X_val)
            test_folds[fold, :, 4:6] = et.predict(X_tst)
            conf_folds[fold, :, 4:6] = et.predict(X_cnf)
            
            # Base 4: HistGradientBoosting
            hgb = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42+fold))
            hgb.fit(X_trn, y_trn)
            oof_matrix[val_idx, 6:8] = hgb.predict(X_val)
            test_folds[fold, :, 6:8] = hgb.predict(X_tst)
            conf_folds[fold, :, 6:8] = hgb.predict(X_cnf)
            
        test_pred_matrix = np.mean(test_folds, axis=0)
        conf_pred_matrix = np.mean(conf_folds, axis=0)
        
        # Meta 1: MultiOutput Ridge
        meta_ridge = MultiOutputRegressor(Ridge(alpha=1.0))
        meta_ridge.fit(oof_matrix, raw_train_y)
        p_ridge_test = meta_ridge.predict(test_pred_matrix)
        p_ridge_conf = meta_ridge.predict(conf_pred_matrix)
        m_ridge_test = compute_inverse_metrics(y_test, p_ridge_test, L_test, H_test)
        m_ridge_conf = compute_inverse_metrics(y_conf, p_ridge_conf, L_conf, H_conf)
        
        # Meta 2: PCA-Stacked Meta
        pca = PCA(n_components=4)
        oof_pca = pca.fit_transform(oof_matrix)
        test_pca = pca.transform(test_pred_matrix)
        conf_pca = pca.transform(conf_pred_matrix)
        meta_pca = MultiOutputRegressor(Ridge(alpha=1.0))
        meta_pca.fit(oof_pca, raw_train_y)
        p_pca_test = meta_pca.predict(test_pca)
        p_pca_conf = meta_pca.predict(conf_pca)
        m_pca_test = compute_inverse_metrics(y_test, p_pca_test, L_test, H_test)
        m_pca_conf = compute_inverse_metrics(y_conf, p_pca_conf, L_conf, H_conf)
        
        stacking_res = {
            'Inverse_Ridge_Stack': {'historical_test': m_ridge_test, 'confirmatory': m_ridge_conf},
            'Inverse_PCA_Stack': {'historical_test': m_pca_test, 'confirmatory': m_pca_conf}
        }
        print(f"  Inverse Ridge Stack -> Test Rx: {m_ridge_test['corr_X']:.4f}, Ry: {m_ridge_test['corr_Y']:.4f}, Mean Euc: {m_ridge_test['mean_euclidean_mm']:.1f} mm, <=25mm: {m_ridge_test['pct_within_25mm']:.1f}%")
        print(f"  Inverse PCA Stack   -> Test Rx: {m_pca_test['corr_X']:.4f}, Ry: {m_pca_test['corr_Y']:.4f}, Mean Euc: {m_pca_test['mean_euclidean_mm']:.1f} mm, <=25mm: {m_pca_test['pct_within_25mm']:.1f}%")
        
        joblib.dump(meta_ridge, self.base_dir / "inverse_stacking/meta_ridge.pkl")
        joblib.dump(meta_pca, self.base_dir / "inverse_stacking/meta_pca.pkl")
        joblib.dump(pca, self.base_dir / "inverse_stacking/pca_transform.pkl")
        
        return stacking_res

    def run_forward_assisted_candidate_search(self) -> Dict[str, Any]:
        """Phase G: Forward-Model-Assisted Inversion via Stage 3B Forward Predictor."""
        print("\n=== RUNNING PHASE G: FORWARD-MODEL-ASSISTED INVERSE SEARCH ===")
        # Load Stage 3B forward predictor
        forward_predictor = ForwardStressEnsemblePredictor()
        
        # Select 5 representative test cases with varying defect locations
        test_cases = [
            # 1. Close to central hole
            self.df_test[(self.df_test['geometry_family'] == 'square_1000x1000') & (self.df_test['hole_to_hole_clearance_mm'] < 15.0)].iloc[0],
            # 2. Quadrant 1 interior
            self.df_test[(self.df_test['defect_x_over_L'] > 0.65) & (self.df_test['defect_y_over_H'] > 0.65)].iloc[0],
            # 3. Near left boundary
            self.df_test[(self.df_test['defect_x_over_L'] < 0.15)].iloc[0],
            # 4. Rectangular 1000x600 case
            self.df_test[(self.df_test['geometry_family'] == 'rectangular_1000x600')].iloc[0],
            # 5. Near top boundary
            self.df_test[(self.df_test['defect_y_over_H'] > 0.85)].iloc[0]
        ]
        
        search_results = []
        for idx, row in enumerate(test_cases):
            L = float(row['plate_length_mm'])
            H = float(row['plate_height_mm'])
            Dc = float(row['central_hole_diameter_mm'])
            D = float(row['defect_diameter_mm'])
            true_X = float(row['defect_center_x_mm'])
            true_Y = float(row['defect_center_y_mm'])
            obs_stress = float(row['max_von_mises_element_psi'])
            obs_kt = float(row['Kt_element'])
            
            # 2D Grid Search: evaluate forward model on a 40x40 spatial grid across plate
            xs = np.linspace(D/2 + 0.01*H, L - D/2 - 0.01*H, 40)
            ys = np.linspace(D/2 + 0.01*H, H - D/2 - 0.01*H, 40)
            XX, YY = np.meshgrid(xs, ys)
            
            grid_loss = np.full(XX.shape, np.nan)
            Xc = L / 2.0
            Yc = H / 2.0
            
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    x_cand = XX[i, j]
                    y_cand = YY[i, j]
                    dist_c = np.hypot(x_cand - Xc, y_cand - Yc)
                    if dist_c > (Dc + D)/2 + 0.01*H:
                        try:
                            res = forward_predictor.predict(
                                plate_length_mm=L,
                                plate_height_mm=H,
                                central_hole_diameter_mm=Dc,
                                defect_diameter_mm=D,
                                defect_center_x_mm=x_cand,
                                defect_center_y_mm=y_cand,
                                applied_traction_psi=100.0,
                                framework="heterogeneous"
                            )
                            pred_stress = res['predicted_max_stress_psi']
                            grid_loss[i, j] = (pred_stress - obs_stress)**2
                        except Exception:
                            pass
                            
            # Identify candidate regions: points where stress error <= 1.0 psi
            valid_mask = ~np.isnan(grid_loss)
            candidate_mask = valid_mask & (np.sqrt(grid_loss) <= 1.0)
            cand_count = int(np.sum(candidate_mask))
            
            cand_xs = XX[candidate_mask]
            cand_ys = YY[candidate_mask]
            
            if cand_count > 0:
                dists_to_true = np.sqrt((cand_xs - true_X)**2 + (cand_ys - true_Y)**2)
                min_dist_to_true = float(np.min(dists_to_true))
                mean_dist_between_cands = float(np.mean(cdist(np.column_stack([cand_xs, cand_ys]), np.column_stack([cand_xs, cand_ys]))))
            else:
                min_dist_to_true = np.nan
                mean_dist_between_cands = np.nan
                
            res_dict = {
                'case_id': int(idx + 1),
                'geometry': str(row['geometry_family']),
                'L_mm': L, 'H_mm': H, 'Dc_mm': Dc, 'D_mm': D,
                'true_X_mm': true_X, 'true_Y_mm': true_Y,
                'observed_stress_psi': obs_stress,
                'candidate_points_within_1psi': cand_count,
                'min_dist_to_true_mm': min_dist_to_true,
                'candidate_cloud_dispersion_mm': mean_dist_between_cands,
                'grid_X': xs.tolist(),
                'grid_Y': ys.tolist(),
                'grid_loss': np.where(np.isnan(grid_loss), -1.0, grid_loss).tolist()
            }
            search_results.append(res_dict)
            print(f"  Case {idx+1} ({row['geometry_family']}) -> Obs Stress: {obs_stress:.1f} psi. Found {cand_count} candidate locations within 1 psi contour! (Min dist to true: {min_dist_to_true:.1f} mm, Cloud dispersion: {mean_dist_between_cands:.1f} mm)")
            
        with open(self.base_dir / "forward_assisted_search/search_results.json", "w") as f:
            json.dump(search_results, f, indent=2)
            
        return {'cases': search_results}

    def run_noise_sensitivity_analysis(self, best_model, scaler, in_cols) -> Dict[str, Any]:
        """Phase L: Stress Measurement Perturbation Analysis (+-0.5%, +-1%, +-2%, +-5%)."""
        print("\n=== RUNNING PHASE L: STRESS NOISE SENSITIVITY ANALYSIS ===")
        noise_levels = [0.005, 0.010, 0.020, 0.050]
        
        X_test_raw = self.df_test[in_cols].values.copy()
        y_test = self.df_test[['defect_x_over_L', 'defect_y_over_H']].values
        L_test = self.df_test['plate_length_mm'].values
        H_test = self.df_test['plate_height_mm'].values
        
        # Baseline unperturbed
        X_test_norm = scaler.transform(X_test_raw)
        p_base = best_model.predict(X_test_norm)
        m_base = compute_inverse_metrics(y_test, p_base, L_test, H_test)
        
        noise_results = {'baseline': m_base, 'perturbations': {}}
        
        for nl in noise_levels:
            np.random.seed(42)
            # Perturb only the stress column (col 0: Kt_element)
            noise_factor = 1.0 + np.random.normal(0.0, nl, size=len(X_test_raw))
            X_pert = X_test_raw.copy()
            X_pert[:, 0] = X_pert[:, 0] * noise_factor
            
            X_pert_norm = scaler.transform(X_pert)
            p_pert = best_model.predict(X_pert_norm)
            m_pert = compute_inverse_metrics(y_test, p_pert, L_test, H_test)
            
            # Coordinate shift due to stress perturbation
            shift_mm = np.sqrt(((p_pert[:, 0] - p_base[:, 0])*L_test)**2 + ((p_pert[:, 1] - p_base[:, 1])*H_test)**2)
            mean_shift = float(np.mean(shift_mm))
            max_shift = float(np.max(shift_mm))
            
            noise_results['perturbations'][f"+-{nl*100:.1f}%"] = {
                'metrics': m_pert,
                'mean_prediction_shift_mm': mean_shift,
                'max_prediction_shift_mm': max_shift
            }
            print(f"  Stress Noise +- {nl*100:3.1f}% -> Mean Loc Error: {m_pert['mean_euclidean_mm']:.1f} mm, Mean Pred Shift: {mean_shift:.2f} mm (Max: {max_shift:.1f} mm)")
            
        with open(self.base_dir / "uncertainty/noise_sensitivity.json", "w") as f:
            json.dump(noise_results, f, indent=2)
            
        return noise_results

def run_stage3C_full_pipeline():
    engine = Stage3CInverseEngine()
    engine.load_datasets()
    
    # 1. Experiments C1, C2, C3
    c1_res = engine.run_experiment_C1()
    c2_res = engine.run_experiment_C2()
    c3_res = engine.run_experiment_C3()
    
    # 2. Conventional Baselines (Phase E)
    conv_res = engine.run_conventional_baselines()
    
    # 3. Inverse Stacking (Phase F)
    stack_res = engine.run_inverse_stacking()
    
    # 4. Forward-Assisted Inverse Search (Phase G)
    search_res = engine.run_forward_assisted_candidate_search()
    
    # 5. Noise Sensitivity (Phase L)
    in_cols = ['Kt_element', 'aspect_ratio', 'defect_d_over_Dc']
    scaler = joblib.load(engine.base_dir / "preprocessing/c3_scaler.pkl")
    best_model = joblib.load(engine.base_dir / "conventional_regression/RandomForest.pkl")
    noise_res = engine.run_noise_sensitivity_analysis(best_model, scaler, in_cols)
    
    # Master summary
    master_summary = {
        'Experiment_C1_Reference': c1_res,
        'Experiment_C2_Scalar_MultiGeom': c2_res,
        'Experiment_C3_Conditioned': c3_res,
        'Conventional_Baselines': conv_res,
        'Inverse_Stacking': stack_res,
        'Noise_Sensitivity': noise_res
    }
    with open(engine.output_dir / "stage3C_master_results.json", "w") as f:
        json.dump(master_summary, f, indent=2)
        
    print("\nStage 3C Comprehensive Inverse Engine Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_stage3C_full_pipeline()

