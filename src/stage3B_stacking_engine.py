"""
Stage 3B: Comprehensive Forward Stacking Engine
Implements:
1. Feature extraction with Feature Set D (physical clearances & aspect ratio).
2. Phase D: Single-Model Baselines (ANNs, DeepMLP, ResMLP, Random Forest, Extra Trees, HistGradientBoosting).
3. 5-Fold Cross-Fitting for Out-Of-Fold (OOF) Prediction Matrix (Zero Leakage).
4. Phase E: Ensemble 1 — DNN-Only Stacking.
5. Phase F: Ensemble 2 — PCA-Based Stacking.
6. Phase G: Ensemble 3 — Multi-Level Meta-Stacking.
7. Phase H: Ensemble 4 — Modern Heterogeneous Stacking.
8. Phase I & Section 24: Model Diversity Analysis & Leave-One-Geometry-Out (LOGO) Cross-Validation.
9. Comprehensive Evaluation across:
   - Historical Test Set (N = 7,557)
   - Retrospective Confirmatory Set (N = 500)
   - New Stage 3B Blind Confirmatory Set (N = 500: in-domain + out-of-domain d/Dc in [0.32, 0.45])
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
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Metric calculation
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

# Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    L = df['plate_length_mm']
    H = df['plate_height_mm']
    Dc = df['central_hole_diameter_mm']
    D = df['defect_diameter_mm']
    X = df['defect_center_x_mm']
    Y = df['defect_center_y_mm']
    Xc = L / 2.0
    Yc = H / 2.0
    
    df['defect_x_over_L'] = X / L
    df['defect_y_over_H'] = Y / H
    df['defect_d_over_Dc'] = D / Dc
    df['defect_d_over_H'] = D / H
    df['aspect_ratio'] = L / H
    
    d_centres = np.hypot(X - Xc, Y - Yc)
    ligament_clearance = d_centres - (Dc + D) / 2.0
    df['ligament_clearance_over_Dc'] = ligament_clearance / Dc
    
    c_left = X - D / 2.0
    c_right = L - X - D / 2.0
    c_bot = Y - D / 2.0
    c_top = H - Y - D / 2.0
    min_bnd = np.minimum(np.minimum(c_left, c_right), np.minimum(c_bot, c_top))
    df['min_boundary_clearance_over_H'] = min_bnd / H
    
    return df

FEATURE_COLS = [
    'defect_x_over_L',
    'defect_y_over_H',
    'defect_d_over_Dc',
    'aspect_ratio',
    'ligament_clearance_over_Dc',
    'min_boundary_clearance_over_H'
]

# PyTorch Architectures
class ShallowANN(nn.Module):
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D1(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D2(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class ResMLP(nn.Module):
    def __init__(self, in_features: int, width: int = 256, num_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, width),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(width, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_proj(h)

def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 2e-3
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    t_X = torch.tensor(X_train, dtype=torch.float32, device=device)
    t_y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    dataset = TensorDataset(t_X, t_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def predict_torch_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_X = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(t_X).cpu().numpy().flatten()
    return preds

# Main Runner Class
class Stage3BStackingOrchestrator:
    def __init__(self):
        self.output_dir = Path("output")
        self.models_dir = Path("models/stage3B_forward_stacking")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
    def load_data(self):
        print("Loading datasets...")
        # Main enriched dataset
        df_enriched = pd.read_csv("data/enriched_dataset_51625.csv")
        df_manifest = pd.read_csv("data/stage3A_ann_split_manifest.csv")
        df_enriched['split'] = df_manifest['split'].values
        df_enriched = engineer_features(df_enriched)
        
        # Test sets
        df_conf_3AR = pd.read_csv("data/stage3AR_confirmatory_test/confirmatory_dataset_500.csv")
        df_conf_3AR = engineer_features(df_conf_3AR)
        
        df_blind_3B = pd.read_csv("data/stage3B_blind_confirmatory/confirmatory_dataset_500.csv")
        df_blind_3B = engineer_features(df_blind_3B)
        
        self.df_train = df_enriched[df_enriched['split'] == 'train'].copy().reset_index(drop=True)
        self.df_val = df_enriched[df_enriched['split'] == 'val'].copy().reset_index(drop=True)
        self.df_test = df_enriched[df_enriched['split'] == 'test'].copy().reset_index(drop=True)
        self.df_conf_3AR = df_conf_3AR.copy().reset_index(drop=True)
        self.df_blind_3B = df_blind_3B.copy().reset_index(drop=True)
        
        print(f"  Train: {len(self.df_train)}, Val: {len(self.df_val)}, Historical Test: {len(self.df_test)}")
        print(f"  Stage 3A-R Confirmatory: {len(self.df_conf_3AR)}, Stage 3B Blind Confirmatory: {len(self.df_blind_3B)}")
        
        # Standardize features using training set ONLY
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.df_train[FEATURE_COLS].values)
        self.y_train = self.df_train['Kt_element'].values
        
        self.X_val = self.scaler.transform(self.df_val[FEATURE_COLS].values)
        self.y_val = self.df_val['Kt_element'].values
        
        self.X_test = self.scaler.transform(self.df_test[FEATURE_COLS].values)
        self.y_test = self.df_test['Kt_element'].values
        
        self.X_conf_3AR = self.scaler.transform(self.df_conf_3AR[FEATURE_COLS].values)
        self.y_conf_3AR = self.df_conf_3AR['Kt_element'].values
        
        self.X_blind_3B = self.scaler.transform(self.df_blind_3B[FEATURE_COLS].values)
        self.y_blind_3B = self.df_blind_3B['Kt_element'].values
        
        # Save scaler
        joblib.dump(self.scaler, self.models_dir / "stage3B_feature_scaler.pkl")

    def run_single_model_baselines(self) -> Dict[str, Any]:
        print("\n--- PHASE D: SINGLE-MODEL BASELINES ---")
        set_seed(42)
        
        models_dict = {
            'ANN_128': ShallowANN(len(FEATURE_COLS), 128),
            'ANN_256': ShallowANN(len(FEATURE_COLS), 256),
            'ANN_512': ShallowANN(len(FEATURE_COLS), 512),
            'DeepMLP_D1': DeepMLP_D1(len(FEATURE_COLS)),
            'DeepMLP_D2': DeepMLP_D2(len(FEATURE_COLS)),
            'ResMLP': ResMLP(len(FEATURE_COLS), width=256, num_blocks=3),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_split=4, n_jobs=-1, random_state=42),
            'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, random_state=42)
        }
        
        baseline_results = {}
        for name, m in models_dict.items():
            t0 = time.perf_counter()
            print(f"Training baseline: {name}...")
            if 'ANN' in name or 'MLP' in name:
                fitted = train_torch_model(m, self.X_train, self.y_train, epochs=50, batch_size=512)
                p_val = predict_torch_model(fitted, self.X_val)
                p_test = predict_torch_model(fitted, self.X_test)
                p_3AR = predict_torch_model(fitted, self.X_conf_3AR)
                p_3B = predict_torch_model(fitted, self.X_blind_3B)
                torch.save(fitted.state_dict(), self.models_dir / f"baseline_{name}.pt")
            else:
                m.fit(self.X_train, self.y_train)
                p_val = m.predict(self.X_val)
                p_test = m.predict(self.X_test)
                p_3AR = m.predict(self.X_conf_3AR)
                p_3B = m.predict(self.X_blind_3B)
                joblib.dump(m, self.models_dir / f"baseline_{name}.pkl")
                
            elapsed = time.perf_counter() - t0
            m_val = compute_metrics(self.y_val, p_val)
            m_test = compute_metrics(self.y_test, p_test)
            m_3AR = compute_metrics(self.y_conf_3AR, p_3AR)
            m_3B = compute_metrics(self.y_blind_3B, p_3B)
            
            baseline_results[name] = {
                'fit_time_s': elapsed,
                'val': m_val,
                'historical_test': m_test,
                'confirmatory_3AR': m_3AR,
                'blind_confirmatory_3B': m_3B
            }
            print(f"  {name} -> Hist Test R2: {m_test['r2']:.4f}, RMSE: {m_test['rmse_psi']:.2f} psi, MaxErr: {m_test['max_error_psi']:.2f} psi (Fit: {elapsed:.1f}s)")
            
        with open(self.output_dir / "stage3B_single_model_baselines.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
            
        return baseline_results

    def generate_5fold_oof_predictions(self):
        print("\n--- 5-FOLD CROSS-FITTING FOR ZERO-LEAKAGE OOF PREDICTIONS ---")
        set_seed(42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        base_learner_names = [
            'ANN_512', 'DeepMLP_D1', 'DeepMLP_D2', 'ResMLP',
            'RandomForest', 'ExtraTrees', 'HistGradientBoosting'
        ]
        
        N_train = len(self.df_train)
        N_test = len(self.df_test)
        N_3AR = len(self.df_conf_3AR)
        N_3B = len(self.df_blind_3B)
        M = len(base_learner_names)
        
        oof_matrix = np.zeros((N_train, M), dtype=np.float32)
        test_folds = np.zeros((5, N_test, M), dtype=np.float32)
        conf_3AR_folds = np.zeros((5, N_3AR, M), dtype=np.float32)
        blind_3B_folds = np.zeros((5, N_3B, M), dtype=np.float32)
        
        raw_train_feats = self.df_train[FEATURE_COLS].values
        raw_test_feats = self.df_test[FEATURE_COLS].values
        raw_3AR_feats = self.df_conf_3AR[FEATURE_COLS].values
        raw_3B_feats = self.df_blind_3B[FEATURE_COLS].values
        y_raw_train = self.y_train
        
        for fold, (trn_idx, val_idx) in enumerate(kf.split(raw_train_feats)):
            t_fold = time.perf_counter()
            print(f"\n--- Cross-fitting Fold {fold+1}/5 ---")
            
            # Fit scaler strictly on trn_idx
            fold_scaler = StandardScaler()
            X_trn = fold_scaler.fit_transform(raw_train_feats[trn_idx])
            y_trn = y_raw_train[trn_idx]
            
            X_val_fold = fold_scaler.transform(raw_train_feats[val_idx])
            X_test_fold = fold_scaler.transform(raw_test_feats)
            X_3AR_fold = fold_scaler.transform(raw_3AR_feats)
            X_3B_fold = fold_scaler.transform(raw_3B_feats)
            
            fold_models = {
                'ANN_512': ShallowANN(len(FEATURE_COLS), 512),
                'DeepMLP_D1': DeepMLP_D1(len(FEATURE_COLS)),
                'DeepMLP_D2': DeepMLP_D2(len(FEATURE_COLS)),
                'ResMLP': ResMLP(len(FEATURE_COLS), width=256, num_blocks=3),
                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42 + fold),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_split=4, n_jobs=-1, random_state=42 + fold),
                'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, random_state=42 + fold)
            }
            
            for m_idx, m_name in enumerate(base_learner_names):
                m = fold_models[m_name]
                if 'ANN' in m_name or 'MLP' in m_name:
                    fitted = train_torch_model(m, X_trn, y_trn, epochs=45, batch_size=512)
                    p_oof = predict_torch_model(fitted, X_val_fold)
                    p_test = predict_torch_model(fitted, X_test_fold)
                    p_3AR = predict_torch_model(fitted, X_3AR_fold)
                    p_3B = predict_torch_model(fitted, X_3B_fold)
                else:
                    m.fit(X_trn, y_trn)
                    p_oof = m.predict(X_val_fold)
                    p_test = m.predict(X_test_fold)
                    p_3AR = m.predict(X_3AR_fold)
                    p_3B = m.predict(X_3B_fold)
                    
                oof_matrix[val_idx, m_idx] = p_oof
                test_folds[fold, :, m_idx] = p_test
                conf_3AR_folds[fold, :, m_idx] = p_3AR
                blind_3B_folds[fold, :, m_idx] = p_3B
                
            print(f"  Fold {fold+1} completed in {time.perf_counter() - t_fold:.1f}s")
            
        test_pred_matrix = np.mean(test_folds, axis=0)
        conf_3AR_pred_matrix = np.mean(conf_3AR_folds, axis=0)
        blind_3B_pred_matrix = np.mean(blind_3B_folds, axis=0)
        
        np.savez_compressed(
            self.output_dir / "stage3B_oof_predictions.npz",
            base_learner_names=base_learner_names,
            oof_matrix=oof_matrix,
            y_train=self.y_train,
            test_matrix=test_pred_matrix,
            y_test=self.y_test,
            conf_3AR_matrix=conf_3AR_pred_matrix,
            y_conf_3AR=self.y_conf_3AR,
            blind_3B_matrix=blind_3B_pred_matrix,
            y_blind_3B=self.y_blind_3B
        )
        print("Successfully generated and saved OOF prediction matrices.")
        
        self.base_learner_names = base_learner_names
        self.oof_matrix = oof_matrix
        self.test_matrix = test_pred_matrix
        self.conf_3AR_matrix = conf_3AR_pred_matrix
        self.blind_3B_matrix = blind_3B_pred_matrix

    def run_stacking_frameworks(self) -> Dict[str, Any]:
        print("\n--- PHASES E, F, G, H: EVALUATING 4 STACKING FRAMEWORKS ---")
        
        stacking_results = {}
        
        # 1. Framework 1: DNN-Only Stacking (Base: ANN-512, DeepMLP-D1, DeepMLP-D2, ResMLP)
        print("\n=== Framework 1: DNN-Only Stacked Ensemble ===")
        dnn_indices = [0, 1, 2, 3] # Indices for ANN_512, D1, D2, ResMLP
        Z_dnn_oof = self.oof_matrix[:, dnn_indices]
        Z_dnn_test = self.test_matrix[:, dnn_indices]
        Z_dnn_3AR = self.conf_3AR_matrix[:, dnn_indices]
        Z_dnn_3B = self.blind_3B_matrix[:, dnn_indices]
        
        # Meta 1A: Ridge
        meta_dnn_ridge = Ridge(alpha=1.0)
        meta_dnn_ridge.fit(Z_dnn_oof, self.y_train)
        pred_dnn_test = meta_dnn_ridge.predict(Z_dnn_test)
        pred_dnn_3AR = meta_dnn_ridge.predict(Z_dnn_3AR)
        pred_dnn_3B = meta_dnn_ridge.predict(Z_dnn_3B)
        
        stacking_results['Framework_1_DNN_Ridge'] = {
            'meta_weights': meta_dnn_ridge.coef_.tolist(),
            'intercept': float(meta_dnn_ridge.intercept_),
            'historical_test': compute_metrics(self.y_test, pred_dnn_test),
            'confirmatory_3AR': compute_metrics(self.y_conf_3AR, pred_dnn_3AR),
            'blind_confirmatory_3B': compute_metrics(self.y_blind_3B, pred_dnn_3B)
        }
        print(f"DNN-Ridge -> Hist Test R2: {stacking_results['Framework_1_DNN_Ridge']['historical_test']['r2']:.4f}, Corr R: {stacking_results['Framework_1_DNN_Ridge']['historical_test']['corr']:.4f}, RMSE: {stacking_results['Framework_1_DNN_Ridge']['historical_test']['rmse_psi']:.2f} psi")
        joblib.dump(meta_dnn_ridge, self.models_dir / "meta_framework1_dnn_ridge.pkl")
        
        # Meta 1B: Non-negative Linear
        meta_dnn_nn = LinearRegression(positive=True)
        meta_dnn_nn.fit(Z_dnn_oof, self.y_train)
        pred_dnn_nn_test = meta_dnn_nn.predict(Z_dnn_test)
        pred_dnn_nn_3AR = meta_dnn_nn.predict(Z_dnn_3AR)
        pred_dnn_nn_3B = meta_dnn_nn.predict(Z_dnn_3B)
        stacking_results['Framework_1_DNN_NonNegative'] = {
            'meta_weights': meta_dnn_nn.coef_.tolist(),
            'intercept': float(meta_dnn_nn.intercept_),
            'historical_test': compute_metrics(self.y_test, pred_dnn_nn_test),
            'confirmatory_3AR': compute_metrics(self.y_conf_3AR, pred_dnn_nn_3AR),
            'blind_confirmatory_3B': compute_metrics(self.y_blind_3B, pred_dnn_nn_3B)
        }
        
        # 2. Framework 2: PCA-Based Stacking
        print("\n=== Framework 2: PCA-Based Stacked Ensemble ===")
        # Fit scaler on OOF matrix
        pca_scaler = StandardScaler()
        Z_all_oof_norm = pca_scaler.fit_transform(self.oof_matrix)
        Z_all_test_norm = pca_scaler.transform(self.test_matrix)
        Z_all_3AR_norm = pca_scaler.transform(self.conf_3AR_matrix)
        Z_all_3B_norm = pca_scaler.transform(self.blind_3B_matrix)
        
        pca = PCA()
        Z_pca_oof = pca.fit_transform(Z_all_oof_norm)
        evr = pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        # Select k components explaining > 99% variance
        k_components = int(np.argmax(cum_evr >= 0.99) + 1)
        print(f"PCA explained variance ratios: {evr[:4].round(4)}. Selected k={k_components} components (cumulative: {cum_evr[k_components-1]*100:.2f}%)")
        
        Z_pca_oof_k = Z_pca_oof[:, :k_components]
        Z_pca_test_k = pca.transform(Z_all_test_norm)[:, :k_components]
        Z_pca_3AR_k = pca.transform(Z_all_3AR_norm)[:, :k_components]
        Z_pca_3B_k = pca.transform(Z_all_3B_norm)[:, :k_components]
        
        meta_pca = Ridge(alpha=1.0)
        meta_pca.fit(Z_pca_oof_k, self.y_train)
        pred_pca_test = meta_pca.predict(Z_pca_test_k)
        pred_pca_3AR = meta_pca.predict(Z_pca_3AR_k)
        pred_pca_3B = meta_pca.predict(Z_pca_3B_k)
        
        stacking_results['Framework_2_PCA_Ridge'] = {
            'k_components': k_components,
            'explained_variance_ratio': evr.tolist(),
            'cumulative_variance': cum_evr.tolist(),
            'historical_test': compute_metrics(self.y_test, pred_pca_test),
            'confirmatory_3AR': compute_metrics(self.y_conf_3AR, pred_pca_3AR),
            'blind_confirmatory_3B': compute_metrics(self.y_blind_3B, pred_pca_3B)
        }
        print(f"PCA-Ridge -> Hist Test R2: {stacking_results['Framework_2_PCA_Ridge']['historical_test']['r2']:.4f}, Corr R: {stacking_results['Framework_2_PCA_Ridge']['historical_test']['corr']:.4f}, RMSE: {stacking_results['Framework_2_PCA_Ridge']['historical_test']['rmse_psi']:.2f} psi")
        joblib.dump(meta_pca, self.models_dir / "meta_framework2_pca_ridge.pkl")
        joblib.dump(pca, self.models_dir / "meta_framework2_pca.pkl")
        joblib.dump(pca_scaler, self.models_dir / "meta_framework2_pca_scaler.pkl")
        
        # 3. Framework 3: Multi-Level Meta-Stacking
        print("\n=== Framework 3: Multi-Level Meta-Stacked Ensemble ===")
        # Level 0 is self.oof_matrix
        # Level 1: Cross-fitted Ridge and ExtraTrees
        kf_l1 = KFold(n_splits=5, shuffle=True, random_state=123)
        l1_oof = np.zeros((len(self.y_train), 2), dtype=np.float32)
        l1_test_folds = np.zeros((5, len(self.y_test), 2), dtype=np.float32)
        l1_3AR_folds = np.zeros((5, len(self.y_conf_3AR), 2), dtype=np.float32)
        l1_3B_folds = np.zeros((5, len(self.y_blind_3B), 2), dtype=np.float32)
        
        for fold, (trn_idx, val_idx) in enumerate(kf_l1.split(self.oof_matrix)):
            m_ridge = Ridge(alpha=1.0)
            m_ridge.fit(self.oof_matrix[trn_idx], self.y_train[trn_idx])
            
            m_et = ExtraTreesRegressor(n_estimators=50, max_depth=12, random_state=42+fold, n_jobs=-1)
            m_et.fit(self.oof_matrix[trn_idx], self.y_train[trn_idx])
            
            l1_oof[val_idx, 0] = m_ridge.predict(self.oof_matrix[val_idx])
            l1_oof[val_idx, 1] = m_et.predict(self.oof_matrix[val_idx])
            
            l1_test_folds[fold, :, 0] = m_ridge.predict(self.test_matrix)
            l1_test_folds[fold, :, 1] = m_et.predict(self.test_matrix)
            
            l1_3AR_folds[fold, :, 0] = m_ridge.predict(self.conf_3AR_matrix)
            l1_3AR_folds[fold, :, 1] = m_et.predict(self.conf_3AR_matrix)
            
            l1_3B_folds[fold, :, 0] = m_ridge.predict(self.blind_3B_matrix)
            l1_3B_folds[fold, :, 1] = m_et.predict(self.blind_3B_matrix)
            
        l1_test = np.mean(l1_test_folds, axis=0)
        l1_3AR = np.mean(l1_3AR_folds, axis=0)
        l1_3B = np.mean(l1_3B_folds, axis=0)
        
        # Level 2 Final Meta-Learner: Non-negative Linear
        l2_meta = LinearRegression(positive=True)
        l2_meta.fit(l1_oof, self.y_train)
        pred_l2_test = l2_meta.predict(l1_test)
        pred_l2_3AR = l2_meta.predict(l1_3AR)
        pred_l2_3B = l2_meta.predict(l1_3B)
        
        stacking_results['Framework_3_MultiLevel_Meta'] = {
            'l2_weights': l2_meta.coef_.tolist(),
            'l2_intercept': float(l2_meta.intercept_),
            'historical_test': compute_metrics(self.y_test, pred_l2_test),
            'confirmatory_3AR': compute_metrics(self.y_conf_3AR, pred_l2_3AR),
            'blind_confirmatory_3B': compute_metrics(self.y_blind_3B, pred_l2_3B)
        }
        print(f"Multi-Level Meta -> Hist Test R2: {stacking_results['Framework_3_MultiLevel_Meta']['historical_test']['r2']:.4f}, Corr R: {stacking_results['Framework_3_MultiLevel_Meta']['historical_test']['corr']:.4f}, RMSE: {stacking_results['Framework_3_MultiLevel_Meta']['historical_test']['rmse_psi']:.2f} psi")
        joblib.dump(l2_meta, self.models_dir / "meta_framework3_l2_meta.pkl")
        
        # 4. Framework 4: Modern Heterogeneous Stacked Regression
        print("\n=== Framework 4: Modern Heterogeneous Stacked Regression ===")
        # Base: ResMLP (index 3), DeepMLP-D2 (index 2), RandomForest (index 4), ExtraTrees (index 5), HistGradientBoosting (index 6)
        het_indices = [2, 3, 4, 5, 6]
        het_names = [self.base_learner_names[i] for i in het_indices]
        Z_het_oof = self.oof_matrix[:, het_indices]
        Z_het_test = self.test_matrix[:, het_indices]
        Z_het_3AR = self.conf_3AR_matrix[:, het_indices]
        Z_het_3B = self.blind_3B_matrix[:, het_indices]
        
        # Fit non-negative constrained Ridge / ElasticNet
        meta_het = LinearRegression(positive=True)
        meta_het.fit(Z_het_oof, self.y_train)
        pred_het_test = meta_het.predict(Z_het_test)
        pred_het_3AR = meta_het.predict(Z_het_3AR)
        pred_het_3B = meta_het.predict(Z_het_3B)
        
        # Normalized weights
        w_raw = meta_het.coef_
        w_norm = (w_raw / np.sum(w_raw)).tolist()
        
        stacking_results['Framework_4_Modern_Heterogeneous'] = {
            'base_models': het_names,
            'raw_weights': w_raw.tolist(),
            'normalized_weights': dict(zip(het_names, w_norm)),
            'intercept': float(meta_het.intercept_),
            'historical_test': compute_metrics(self.y_test, pred_het_test),
            'confirmatory_3AR': compute_metrics(self.y_conf_3AR, pred_het_3AR),
            'blind_confirmatory_3B': compute_metrics(self.y_blind_3B, pred_het_3B)
        }
        print(f"Modern Heterogeneous -> Hist Test R2: {stacking_results['Framework_4_Modern_Heterogeneous']['historical_test']['r2']:.4f}, Corr R: {stacking_results['Framework_4_Modern_Heterogeneous']['historical_test']['corr']:.4f}, RMSE: {stacking_results['Framework_4_Modern_Heterogeneous']['historical_test']['rmse_psi']:.2f} psi")
        print(f"  Weights: {dict(zip(het_names, np.round(w_norm, 4)))}")
        joblib.dump(meta_het, self.models_dir / "meta_framework4_heterogeneous.pkl")
        
        with open(self.output_dir / "stage3B_stacking_results.json", "w") as f:
            json.dump(stacking_results, f, indent=2)
            
        return stacking_results

    def run_logo_experiments(self) -> Dict[str, Any]:
        print("\n--- PHASE I & SECTION 24: LEAVE-ONE-GEOMETRY-OUT (LOGO) EXPERIMENTS ---")
        set_seed(42)
        
        geometries = [
            'rectangular_1000x600',
            'rectangular_1000x700',
            'rectangular_1000x800',
            'rectangular_1000x900',
            'square_1000x1000'
        ]
        
        # Combine train, val, test from enriched dataset for LOGO
        df_full = pd.concat([self.df_train, self.df_val, self.df_test], ignore_index=True)
        
        logo_results = {}
        for holdout_geom in geometries:
            t0 = time.perf_counter()
            print(f"\nRunning LOGO holdout: {holdout_geom}...")
            
            df_in = df_full[df_full['geometry_family'] != holdout_geom]
            df_out = df_full[df_full['geometry_family'] == holdout_geom]
            
            # Scaler fitted strictly on remaining 4 geometries
            logo_scaler = StandardScaler()
            X_in = logo_scaler.fit_transform(df_in[FEATURE_COLS].values)
            y_in = df_in['Kt_element'].values
            
            X_out = logo_scaler.transform(df_out[FEATURE_COLS].values)
            y_out = df_out['Kt_element'].values
            
            # Fast, high-capacity heterogeneous pair: ResMLP + HistGradientBoosting
            resmlp = ResMLP(len(FEATURE_COLS), width=256, num_blocks=3)
            resmlp = train_torch_model(resmlp, X_in, y_in, epochs=40, batch_size=512)
            p_resmlp = predict_torch_model(resmlp, X_out)
            
            hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=63, random_state=42)
            hgb.fit(X_in, y_in)
            p_hgb = hgb.predict(X_out)
            
            # Blend
            p_blend = 0.5 * p_resmlp + 0.5 * p_hgb
            
            m_resmlp = compute_metrics(y_out, p_resmlp)
            m_hgb = compute_metrics(y_out, p_hgb)
            m_blend = compute_metrics(y_out, p_blend)
            
            elapsed = time.perf_counter() - t0
            logo_results[holdout_geom] = {
                'holdout_count': len(df_out),
                'train_count': len(df_in),
                'time_s': elapsed,
                'resmlp': m_resmlp,
                'hgb': m_hgb,
                'ensemble_blend': m_blend
            }
            print(f"  Holdout {holdout_geom} -> Blend R2: {m_blend['r2']:.4f}, Corr R: {m_blend['corr']:.4f}, RMSE: {m_blend['rmse_psi']:.2f} psi (in {elapsed:.1f}s)")
            
        with open(self.output_dir / "stage3B_logo_results.json", "w") as f:
            json.dump(logo_results, f, indent=2)
            
        return logo_results

    def compute_diversity_analysis(self) -> Dict[str, Any]:
        print("\n--- PHASE I: MODEL DIVERSITY & RESIDUAL CORRELATION ANALYSIS ---")
        # Pairwise correlations on OOF predictions
        pred_corr = pd.DataFrame(self.oof_matrix, columns=self.base_learner_names).corr().values.tolist()
        
        # Residuals
        residuals = self.oof_matrix - self.y_train[:, np.newaxis]
        resid_corr = pd.DataFrame(residuals, columns=self.base_learner_names).corr().values.tolist()
        
        diversity_data = {
            'base_models': self.base_learner_names,
            'prediction_correlation_matrix': pred_corr,
            'residual_correlation_matrix': resid_corr
        }
        with open(self.output_dir / "stage3B_diversity_analysis.json", "w") as f:
            json.dump(diversity_data, f, indent=2)
            
        print("Model diversity matrices saved successfully.")
        return diversity_data

def run_stage3B_full_pipeline():
    orch = Stage3BStackingOrchestrator()
    orch.load_data()
    orch.run_single_model_baselines()
    orch.generate_5fold_oof_predictions()
    orch.run_stacking_frameworks()
    orch.compute_diversity_analysis()
    orch.run_logo_experiments()
    print("\nStage 3B Full Orchestration Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_stage3B_full_pipeline()
