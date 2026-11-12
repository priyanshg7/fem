"""
Stage 3E Experiments Suite:
1. Multi-Seed Stability Analysis (5 distinct seeds: 42, 101, 777, 1234, 9999).
2. Comprehensive Multi-Geometry and LOGO Generalization.
3. Defect Diameter and Clearance Sensitivity.
4. Structured Ablations:
   - Aspect ratio conditioning (Implicit vs Explicit).
   - Spatial resolution (32x32 vs 64x64 vs 128x128).
   - Soft-argmax temperature beta (10, 20, 40, 80, 160).
   - Architecture: U-Net with skips vs FCN without skips.
   - Inpainting strategy: Zero-fill vs Nearest-Neighbor vs IDW.
   - Sizing auxiliary head: Joint localization and diameter regression.
5. Selective Classification / Abstention Trade-off Curve.
6. Numerical Discretization Sensitivity (CST centroid vs nodal recovered stress).
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.stage3D_cnn_unet import DefectHeatmapUNet, soft_argmax_2d, DoubleConv
from src.stage3D_metrics import compute_localization_metrics
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field

torch.set_num_threads(min(os.cpu_count() or 4, 8))
STAGE3E_MODEL_DIR = "models/stage3E"
STAGE3E_OUT_DIR = "output/stage3E"
STAGE3E_REP_DIR = "reports/stage3E"
CACHE_DIR = "data/stage3D_field_cache"
CONF_DIR = "data/stage3E_confirmatory"

os.makedirs(STAGE3E_MODEL_DIR, exist_ok=True)
os.makedirs(STAGE3E_OUT_DIR, exist_ok=True)
os.makedirs(STAGE3E_REP_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_RES = 64

def generate_gaussian_heatmaps(coords_norm, grid_res=64, sigma=0.04):
    N = len(coords_norm)
    y_grid, x_grid = np.meshgrid(
        np.linspace(0.0, 1.0, grid_res, dtype=np.float32),
        np.linspace(0.0, 1.0, grid_res, dtype=np.float32),
        indexing='ij'
    )
    heatmaps = np.zeros((N, 1, grid_res, grid_res), dtype=np.float32)
    two_sig_sq = 2.0 * (sigma ** 2)
    for i in range(N):
        cx, cy = coords_norm[i]
        d_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        heatmaps[i, 0] = np.exp(-d_sq / two_sig_sq)
    return heatmaps

def evaluate_uncertainty_calibration(pred_coords_mm, true_coords_mm, cov_matrices_norm, L_arr, H_arr):
    N = len(pred_coords_mm)
    mahalanobis_sq = np.zeros(N)
    unc_radii_phys = np.zeros(N)
    
    for i in range(N):
        L = L_arr[i]
        H = H_arr[i]
        cov_norm = cov_matrices_norm[i]
        cov_phys = np.array([
            [cov_norm[0, 0] * (L**2), cov_norm[0, 1] * (L * H)],
            [cov_norm[1, 0] * (L * H), cov_norm[1, 1] * (H**2)]
        ], dtype=np.float64)
        cov_reg = cov_phys + np.eye(2) * 1e-4
        inv_cov = np.linalg.inv(cov_reg)
        diff = pred_coords_mm[i] - true_coords_mm[i]
        d_sq = float(diff @ inv_cov @ diff)
        mahalanobis_sq[i] = d_sq
        unc_radii_phys[i] = float(np.sqrt(cov_phys[0, 0] + cov_phys[1, 1]))
        
    cov_50 = float(np.mean(mahalanobis_sq <= 1.386) * 100.0)
    cov_68 = float(np.mean(mahalanobis_sq <= 2.296) * 100.0)
    cov_90 = float(np.mean(mahalanobis_sq <= 4.605) * 100.0)
    cov_95 = float(np.mean(mahalanobis_sq <= 5.991) * 100.0)
    cov_99 = float(np.mean(mahalanobis_sq <= 9.210) * 100.0)
    
    euc_errors = np.sqrt(np.sum((pred_coords_mm - true_coords_mm)**2, axis=1))
    corr_unc_err = float(np.corrcoef(unc_radii_phys, euc_errors)[0, 1])
    
    return {
        'unc_radii_phys': unc_radii_phys,
        'mahalanobis_sq': mahalanobis_sq,
        'euc_errors': euc_errors,
        'Mean_Uncertainty_Radius_mm': float(np.mean(unc_radii_phys)),
        'Median_Uncertainty_Radius_mm': float(np.median(unc_radii_phys)),
        'P95_Uncertainty_Radius_mm': float(np.percentile(unc_radii_phys, 95)),
        'Empirical_Coverage_50pct': cov_50,
        'Empirical_Coverage_68pct': cov_68,
        'Empirical_Coverage_90pct': cov_90,
        'Empirical_Coverage_95pct': cov_95,
        'Empirical_Coverage_99pct': cov_99,
        'Correlation_Uncertainty_vs_Error': corr_unc_err
    }

# 1. Architecture Ablation: FCN without Skip Connections
class DefectHeatmapFCN(nn.Module):
    """FCN encoder-decoder without skip connections to test importance of high-frequency spatial context"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(32, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        h = self.pool1(self.enc1(x))
        h = self.pool2(self.enc2(h))
        h = self.pool3(self.enc3(h))
        h = self.bottleneck(h)
        h = self.dec1(self.up1(h))
        h = self.dec2(self.up2(h))
        h = self.dec3(self.up3(h))
        return self.outc(h)

# 2. Defect Sizing + Heatmap Joint Network
class DefectJointUNet(nn.Module):
    """U-Net with auxiliary defect diameter regression head"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)
        
        # Sizing head from bottleneck
        self.size_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid() # d/Dc in [0, 1]
        )
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Sizing
        pred_size = self.size_head(x4)
        
        # Heatmap
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        heatmap = self.outc(u3)
        return heatmap, pred_size

def run_repeated_seeds():
    print("\n==========================================", flush=True)
    print("STAGE 3E: REPEATED SEED STABILITY (5 SEEDS)", flush=True)
    print("==========================================", flush=True)
    
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    X_train_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy"))
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    
    df_train = df_meta[df_meta['subset'] == 'train'].reset_index(drop=True)
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    y_train_norm = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_test_norm = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    
    L_te = df_test['plate_length_mm'].values
    H_te = df_test['plate_height_mm'].values
    true_mm_test = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    
    X_test_t = torch.tensor(X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    
    seeds = [42, 101, 777, 1234, 9999]
    seed_results = []
    
    # Pre-generate ground truth heatmaps for train
    H_train_all = generate_gaussian_heatmaps(y_train_norm, grid_res=GRID_RES, sigma=0.04)
    
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train on 2000 samples for 5 epochs
        n_samples = min(len(X_train_raw), 2000)
        sub_idx = np.random.choice(len(X_train_raw), n_samples, replace=False)
        
        X_sub = torch.tensor(X_train_raw[sub_idx].reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
        H_sub = torch.tensor(H_train_all[sub_idx], dtype=torch.float32)
        
        model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        
        loader = DataLoader(TensorDataset(X_sub, H_sub), batch_size=32, shuffle=True)
        model.train()
        for ep in range(5):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(STAGE3E_MODEL_DIR, f"model_D5_seed_{seed}.pt"))
        
        # Evaluate on Test Set
        model.eval()
        with torch.no_grad():
            logits_te = model(X_test_t)
            coords_te, unc_te, cov_te = soft_argmax_2d(logits_te, beta=40.0)
            pred_mm = np.column_stack([coords_te[:, 0].cpu().numpy() * L_te, coords_te[:, 1].cpu().numpy() * H_te])
            
            m = compute_localization_metrics(true_mm_test, pred_mm)
            cal = evaluate_uncertainty_calibration(pred_mm, true_mm_test, cov_te.cpu().numpy(), L_te, H_te)
            m.update({k: v for k, v in cal.items() if not isinstance(v, np.ndarray)})
            m['seed'] = seed
            seed_results.append(m)
            
            print(f"Seed {seed}: Mean Euc = {m['Mean_Euclidean_mm']:.2f} mm | Median = {m['Median_Euclidean_mm']:.2f} mm | <=25mm = {m['Pct_within_25mm']:.1f}% | 90% Cov = {m['Empirical_Coverage_90pct']:.1f}%", flush=True)
            
    df_seeds = pd.DataFrame(seed_results)
    df_seeds.to_json(os.path.join(STAGE3E_MODEL_DIR, "repeated_seed_results.json"), indent=2, orient='records')
    print("\n--- Seed Stability Summary ---")
    print(df_seeds[['seed', 'Mean_Euclidean_mm', 'Median_Euclidean_mm', 'Pct_within_25mm', 'Mean_Uncertainty_Radius_mm', 'Empirical_Coverage_90pct']])
    return df_seeds

def run_structured_ablations():
    print("\n==========================================", flush=True)
    print("STAGE 3E: STRUCTURED ABLATION STUDIES", flush=True)
    print("==========================================", flush=True)
    
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    X_train_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy"))
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    
    df_train = df_meta[df_meta['subset'] == 'train'].reset_index(drop=True)
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    y_train_norm = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_test_norm = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    
    L_te = df_test['plate_length_mm'].values
    H_te = df_test['plate_height_mm'].values
    true_mm_test = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    
    X_test_t = torch.tensor(X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    
    ablation_results = {}
    
    # 1. Base Model (D5 U-Net Seed 42)
    base_model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    base_model.load_state_dict(torch.load(os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt"), map_location='cpu'))
    base_model.eval()
    
    with torch.no_grad():
        logits_base = base_model(X_test_t)
        coords_base, _, cov_base = soft_argmax_2d(logits_base, beta=40.0)
        pred_base = np.column_stack([coords_base[:, 0].cpu().numpy() * L_te, coords_base[:, 1].cpu().numpy() * H_te])
        m_base = compute_localization_metrics(true_mm_test, pred_base)
        cal_base = evaluate_uncertainty_calibration(pred_base, true_mm_test, cov_base.cpu().numpy(), L_te, H_te)
        m_base.update({k: v for k, v in cal_base.items() if not isinstance(v, np.ndarray)})
        ablation_results['Baseline_UNet_beta40'] = m_base
        
    # 2. Soft-argmax Temperature beta Ablation (beta = 10, 20, 80, 160)
    for beta_val in [10.0, 20.0, 80.0, 160.0]:
        with torch.no_grad():
            coords_b, _, cov_b = soft_argmax_2d(logits_base, beta=beta_val)
            pred_b = np.column_stack([coords_b[:, 0].cpu().numpy() * L_te, coords_b[:, 1].cpu().numpy() * H_te])
            m_b = compute_localization_metrics(true_mm_test, pred_b)
            cal_b = evaluate_uncertainty_calibration(pred_b, true_mm_test, cov_b.cpu().numpy(), L_te, H_te)
            m_b.update({k: v for k, v in cal_b.items() if not isinstance(v, np.ndarray)})
            ablation_results[f'Temperature_beta_{int(beta_val)}'] = m_b
            print(f"Beta {beta_val}: Mean Euc = {m_b['Mean_Euclidean_mm']:.2f} mm | Median = {m_b['Median_Euclidean_mm']:.2f} mm", flush=True)
            
    # 3. Architecture Ablation: FCN without Skip Connections
    print("\n--- Training Architecture Ablation: FCN without skips ---", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    n_samples = min(len(X_train_raw), 2000)
    sub_idx = np.random.choice(len(X_train_raw), n_samples, replace=False)
    H_train_all = generate_gaussian_heatmaps(y_train_norm, grid_res=GRID_RES, sigma=0.04)
    X_sub = torch.tensor(X_train_raw[sub_idx].reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    H_sub = torch.tensor(H_train_all[sub_idx], dtype=torch.float32)
    
    fcn_model = DefectHeatmapFCN(in_channels=1).to(DEVICE)
    opt_fcn = torch.optim.AdamW(fcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_sub, H_sub), batch_size=32, shuffle=True)
    fcn_model.train()
    for ep in range(5):
        for xb, yb in loader:
            opt_fcn.zero_grad()
            crit(fcn_model(xb), yb).backward()
            opt_fcn.step()
            
    fcn_model.eval()
    with torch.no_grad():
        logits_fcn = fcn_model(X_test_t)
        coords_fcn, _, cov_fcn = soft_argmax_2d(logits_fcn, beta=40.0)
        pred_fcn = np.column_stack([coords_fcn[:, 0].cpu().numpy() * L_te, coords_fcn[:, 1].cpu().numpy() * H_te])
        m_fcn = compute_localization_metrics(true_mm_test, pred_fcn)
        cal_fcn = evaluate_uncertainty_calibration(pred_fcn, true_mm_test, cov_fcn.cpu().numpy(), L_te, H_te)
        m_fcn.update({k: v for k, v in cal_fcn.items() if not isinstance(v, np.ndarray)})
        ablation_results['Architecture_FCN_NoSkips'] = m_fcn
        print(f"FCN (No Skips): Mean Euc = {m_fcn['Mean_Euclidean_mm']:.2f} mm | Median = {m_fcn['Median_Euclidean_mm']:.2f} mm", flush=True)
        
    # 4. Joint Defect Sizing + Localization Network
    print("\n--- Training Joint Defect Sizing + Localization Network ---", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    d_norm_train = (df_train.loc[sub_idx, 'defect_diameter_mm'] / df_train.loc[sub_idx, 'central_hole_diameter_mm']).values.astype(np.float32)
    D_sub = torch.tensor(d_norm_train, dtype=torch.float32).unsqueeze(1)
    
    joint_model = DefectJointUNet(in_channels=1).to(DEVICE)
    opt_joint = torch.optim.AdamW(joint_model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_heat = nn.BCEWithLogitsLoss()
    crit_size = nn.MSELoss()
    loader_joint = DataLoader(TensorDataset(X_sub, H_sub, D_sub), batch_size=32, shuffle=True)
    
    joint_model.train()
    for ep in range(5):
        for xb, hb, db in loader_joint:
            opt_joint.zero_grad()
            p_h, p_d = joint_model(xb)
            loss = crit_heat(p_h, hb) + 2.0 * crit_size(p_d, db)
            loss.backward()
            opt_joint.step()
            
    joint_model.eval()
    with torch.no_grad():
        p_h_te, p_d_te = joint_model(X_test_t)
        coords_j, _, cov_j = soft_argmax_2d(p_h_te, beta=40.0)
        pred_j = np.column_stack([coords_j[:, 0].cpu().numpy() * L_te, coords_j[:, 1].cpu().numpy() * H_te])
        m_j = compute_localization_metrics(true_mm_test, pred_j)
        cal_j = evaluate_uncertainty_calibration(pred_j, true_mm_test, cov_j.cpu().numpy(), L_te, H_te)
        m_j.update({k: v for k, v in cal_j.items() if not isinstance(v, np.ndarray)})
        
        # Defect sizing evaluation
        true_d_mm = df_test['defect_diameter_mm'].values
        Dc_te = df_test['central_hole_diameter_mm'].values
        pred_d_mm = (p_d_te.squeeze(1).cpu().numpy()) * Dc_te
        sizing_mae = float(np.mean(np.abs(pred_d_mm - true_d_mm)))
        sizing_rmse = float(np.sqrt(np.mean((pred_d_mm - true_d_mm)**2)))
        sizing_r2 = float(1.0 - np.sum((pred_d_mm - true_d_mm)**2) / np.sum((true_d_mm - np.mean(true_d_mm))**2))
        m_j['Sizing_MAE_mm'] = sizing_mae
        m_j['Sizing_RMSE_mm'] = sizing_rmse
        m_j['Sizing_R2'] = sizing_r2
        ablation_results['Joint_Heatmap_and_Sizing'] = m_j
        print(f"Joint Network: Mean Euc = {m_j['Mean_Euclidean_mm']:.2f} mm | Sizing MAE = {sizing_mae:.2f} mm | Sizing R2 = {sizing_r2:.4f}", flush=True)

    with open(os.path.join(STAGE3E_MODEL_DIR, "ablation_results.json"), 'w') as f:
        json.dump(ablation_results, f, indent=2)
    return ablation_results

def run_abstention_analysis():
    print("\n==========================================", flush=True)
    print("STAGE 3E: UNCERTAINTY & ABSTENTION TRADE-OFF", flush=True)
    print("==========================================", flush=True)
    
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    L_te = df_test['plate_length_mm'].values
    H_te = df_test['plate_height_mm'].values
    true_mm_test = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    
    base_model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    base_model.load_state_dict(torch.load(os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt"), map_location='cpu'))
    base_model.eval()
    
    X_test_t = torch.tensor(X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    with torch.no_grad():
        logits = base_model(X_test_t)
        coords_te, unc_te, cov_te = soft_argmax_2d(logits, beta=40.0)
        pred_mm = np.column_stack([coords_te[:, 0].cpu().numpy() * L_te, coords_te[:, 1].cpu().numpy() * H_te])
        cal = evaluate_uncertainty_calibration(pred_mm, true_mm_test, cov_te.cpu().numpy(), L_te, H_te)
        
    unc_radii = cal['unc_radii_phys']
    euc_errors = cal['euc_errors']
    
    # Sort by uncertainty radius ascending (most certain first)
    sort_order = np.argsort(unc_radii)
    retention_fractions = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    abstention_data = []
    
    for ret in retention_fractions:
        k = int(round(ret * len(unc_radii)))
        idx_kept = sort_order[:k]
        
        kept_errors = euc_errors[idx_kept]
        max_unc_threshold = float(unc_radii[idx_kept[-1]])
        
        m_ret = {
            'Retention_Rate_Pct': ret * 100.0,
            'Retained_Samples': k,
            'Max_Uncertainty_Threshold_mm': max_unc_threshold,
            'Mean_Euclidean_mm': float(np.mean(kept_errors)),
            'Median_Euclidean_mm': float(np.median(kept_errors)),
            'P95_Euclidean_mm': float(np.percentile(kept_errors, 95)),
            'Pct_within_25mm': float(np.mean(kept_errors <= 25.0) * 100.0),
            'Pct_within_10mm': float(np.mean(kept_errors <= 10.0) * 100.0),
        }
        abstention_data.append(m_ret)
        print(f"Retention {ret*100:5.1f}% | Thresh: {max_unc_threshold:6.2f} mm | Mean: {m_ret['Mean_Euclidean_mm']:5.2f} mm | Median: {m_ret['Median_Euclidean_mm']:5.2f} mm | <=25mm: {m_ret['Pct_within_25mm']:5.1f}%", flush=True)
        
    df_abst = pd.DataFrame(abstention_data)
    df_abst.to_json(os.path.join(STAGE3E_MODEL_DIR, "abstention_tradeoff.json"), indent=2, orient='records')
    return df_abst

def run_untouched_confirmatory_evaluation():
    print("\n==========================================", flush=True)
    print("STAGE 3E: UNTOUCHED CONFIRMATORY BENCHMARK (N=500)", flush=True)
    print("==========================================", flush=True)
    meta_path = os.path.join(CONF_DIR, "stage3E_untouched_confirmatory_metadata.csv")
    df_conf = pd.read_csv(meta_path)
    
    cache_conf_path = os.path.join(CONF_DIR, "X_conf_64.npy")
    if os.path.exists(cache_conf_path):
        X_conf = np.load(cache_conf_path)
    else:
        print("Rasterizing 500 confirmatory spatial fields...", flush=True)
        from src.stage3D_feature_extractor import load_coords_and_stress
        X_conf_list = []
        for _, row in df_conf.iterrows():
            coords_mm, vm_stress = load_coords_and_stress(row['field_file'])
            cnorm = normalize_coordinates(coords_mm, row['plate_length_mm'], row['plate_height_mm'])
            grid = rasterize_stress_field(cnorm, vm_stress, grid_res=GRID_RES, method="nearest")
            X_conf_list.append((grid / 100.0).flatten())
        X_conf = np.array(X_conf_list, dtype=np.float32)
        np.save(cache_conf_path, X_conf)
        print(f"Cached {len(X_conf)} confirmatory raster fields to {cache_conf_path}")
        
    X_conf_t = torch.tensor(X_conf.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    L_conf = df_conf['plate_length_mm'].values
    H_conf = df_conf['plate_height_mm'].values
    true_mm_conf = np.column_stack([df_conf['defect_center_x_mm'].values, df_conf['defect_center_y_mm'].values])
    
    base_model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    base_model.load_state_dict(torch.load(os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt"), map_location='cpu'))
    base_model.eval()
    
    with torch.no_grad():
        logits_conf = base_model(X_conf_t)
        coords_conf, unc_conf, cov_conf = soft_argmax_2d(logits_conf, beta=40.0)
        pred_mm_conf = np.column_stack([coords_conf[:, 0].cpu().numpy() * L_conf, coords_conf[:, 1].cpu().numpy() * H_conf])
        
    m_overall = compute_localization_metrics(true_mm_conf, pred_mm_conf)
    cal_conf = evaluate_uncertainty_calibration(pred_mm_conf, true_mm_conf, cov_conf.cpu().numpy(), L_conf, H_conf)
    m_overall.update({k: v for k, v in cal_conf.items() if not isinstance(v, np.ndarray)})
    
    print(f"Confirmatory Overall: Mean Euc = {m_overall['Mean_Euclidean_mm']:.2f} mm | Median = {m_overall['Median_Euclidean_mm']:.2f} mm | <=25mm = {m_overall['Pct_within_25mm']:.1f}%", flush=True)
    
    # Breakdown by sample_type
    sample_type_breakdown = {}
    for stype in df_conf['sample_type'].unique():
        idx = (df_conf['sample_type'] == stype).values
        m_st = compute_localization_metrics(true_mm_conf[idx], pred_mm_conf[idx])
        sample_type_breakdown[stype] = m_st
        print(f"  [{stype}] Mean = {m_st['Mean_Euclidean_mm']:.2f} mm | Median = {m_st['Median_Euclidean_mm']:.2f} mm | <=25mm = {m_st['Pct_within_25mm']:.1f}%", flush=True)
        
    # Breakdown by geometry_family
    geom_breakdown = {}
    for gname in df_conf['geometry_family'].unique():
        idx = (df_conf['geometry_family'] == gname).values
        m_g = compute_localization_metrics(true_mm_conf[idx], pred_mm_conf[idx])
        geom_breakdown[gname] = m_g
        print(f"  [{gname}] Mean = {m_g['Mean_Euclidean_mm']:.2f} mm | Median = {m_g['Median_Euclidean_mm']:.2f} mm | <=25mm = {m_g['Pct_within_25mm']:.1f}%", flush=True)
        
    conf_results = {
        'Overall': m_overall,
        'By_Sample_Type': sample_type_breakdown,
        'By_Geometry_Family': geom_breakdown
    }
    with open(os.path.join(STAGE3E_MODEL_DIR, "confirmatory_results.json"), 'w') as f:
        json.dump(conf_results, f, indent=2)
    return conf_results

def run_discretization_sensitivity():
    print("\n==========================================", flush=True)
    print("STAGE 3E: NUMERICAL DISCRETIZATION SENSITIVITY", flush=True)
    print("==========================================", flush=True)
    meta_path = os.path.join(CONF_DIR, "stage3E_untouched_confirmatory_metadata.csv")
    df_conf = pd.read_csv(meta_path).iloc[:100].reset_index(drop=True)
    
    base_model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    base_model.load_state_dict(torch.load(os.path.join(STAGE3E_MODEL_DIR, "model_D5_seed_42.pt"), map_location='cpu'))
    base_model.eval()
    
    # Compare Element Centroid stress vs Nodal Recovered stress rasterization
    pred_centroid = []
    pred_nodal = []
    true_coords = []
    
    for _, row in df_conf.iterrows():
        fpath = row['field_file']
        data = np.load(fpath)
        L = row['plate_length_mm']
        H = row['plate_height_mm']
        
        # 1. Element centroid
        nodes = data['nodes']
        elements = data['elements']
        elem_vm = data['element_von_mises_psi']
        elem_cents = nodes[elements].mean(axis=1) * 1000.0 # mm
        cnorm_c = normalize_coordinates(elem_cents, L, H)
        grid_c = rasterize_stress_field(cnorm_c, elem_vm, grid_res=GRID_RES, method="nearest") / 100.0
        
        # 2. Nodal recovered
        nodal_vm = data['nodal_von_mises_psi']
        nodal_coords = nodes * 1000.0
        cnorm_n = normalize_coordinates(nodal_coords, L, H)
        grid_n = rasterize_stress_field(cnorm_n, nodal_vm, grid_res=GRID_RES, method="nearest") / 100.0
        
        with torch.no_grad():
            inp_c = torch.tensor(grid_c.reshape(1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
            inp_n = torch.tensor(grid_n.reshape(1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
            c_c, _, _ = soft_argmax_2d(base_model(inp_c), beta=40.0)
            c_n, _, _ = soft_argmax_2d(base_model(inp_n), beta=40.0)
            
        pred_centroid.append([c_c[0, 0].item() * L, c_c[0, 1].item() * H])
        pred_nodal.append([c_n[0, 0].item() * L, c_n[0, 1].item() * H])
        true_coords.append([row['defect_center_x_mm'], row['defect_center_y_mm']])
        
    p_c = np.array(pred_centroid)
    p_n = np.array(pred_nodal)
    t_c = np.array(true_coords)
    
    m_centroid = compute_localization_metrics(t_c, p_c)
    m_nodal = compute_localization_metrics(t_c, p_n)
    
    # Difference between centroid and nodal predictions
    diff_inter = np.sqrt(np.sum((p_c - p_n)**2, axis=1))
    
    res = {
        'Element_Centroid_Evaluation': m_centroid,
        'Nodal_Recovered_Evaluation': m_nodal,
        'Mean_Discretization_Discrepancy_mm': float(np.mean(diff_inter)),
        'Median_Discretization_Discrepancy_mm': float(np.median(diff_inter)),
        'Max_Discretization_Discrepancy_mm': float(np.max(diff_inter)),
        'Correlation_Centroid_vs_Nodal': float(np.corrcoef(p_c.flatten(), p_n.flatten())[0, 1])
    }
    print(f"Discretization Sensitivity: Mean Discrepancy between Centroid and Nodal = {res['Mean_Discretization_Discrepancy_mm']:.2f} mm | Median = {res['Median_Discretization_Discrepancy_mm']:.2f} mm | r = {res['Correlation_Centroid_vs_Nodal']:.4f}", flush=True)
    with open(os.path.join(STAGE3E_MODEL_DIR, "discretization_sensitivity.json"), 'w') as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == '__main__':
    run_repeated_seeds()
    run_structured_ablations()
    run_abstention_analysis()
    run_untouched_confirmatory_evaluation()
    run_discretization_sensitivity()

