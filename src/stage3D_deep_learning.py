"""
Stage 3D Deep Learning Training Pipeline (Optimized for Fast CPU Convergence):
- Phase G: Model D4 — CNN Spatial-Field Inverse Model
- Phase H: Model D5 — Spatial-Field U-Net Heatmap & Uncertainty Model
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.stage3D_cnn_unet import DefectInverseCNN, DefectHeatmapUNet, soft_argmax_2d
from src.stage3D_metrics import compute_localization_metrics

torch.set_num_threads(min(os.cpu_count() or 4, 8))
MODEL_DIR = "models/stage3D_spatial_inverse"
CACHE_DIR = "data/stage3D_field_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

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

def run_deep_learning_pipeline():
    print("\n=======================================================", flush=True)
    print("      PHASE G: MODEL D4 CNN SPATIAL-FIELD INVERSION    ", flush=True)
    print("=======================================================", flush=True)
    
    train_cache = os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy")
    test_cache = os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy")
    conf_cache = os.path.join(CACHE_DIR, f"X_pca_conf_{GRID_RES}.npy")
    meta_cache = os.path.join(CACHE_DIR, "meta_subsets.parquet")
    
    X_train_raw = np.load(train_cache)
    X_test_raw = np.load(test_cache)
    X_conf_raw = np.load(conf_cache)
    df_meta = pd.read_parquet(meta_cache)
    
    df_train = df_meta[df_meta['subset'] == 'train'].reset_index(drop=True)
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    df_conf = df_meta[df_meta['subset'] == 'conf'].reset_index(drop=True)
    
    # Fast subset of train: 2000 samples
    N_TRAIN = 2000
    idx_sub = np.random.RandomState(42).choice(len(df_train), N_TRAIN, replace=False)
    
    X_train_sub = X_train_raw[idx_sub].reshape(-1, 1, GRID_RES, GRID_RES)
    X_train_field = torch.tensor(X_train_sub, dtype=torch.float32)
    X_test_field = torch.tensor(X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    X_conf_field = torch.tensor(X_conf_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
    
    def make_meta(df):
        ar = (df['plate_length_mm'] / df['plate_height_mm']).values
        d_ratio = df['defect_d_over_Dc'].values
        return torch.tensor(np.column_stack([ar, d_ratio]), dtype=torch.float32)
        
    M_train = make_meta(df_train.iloc[idx_sub].reset_index(drop=True))
    M_test = make_meta(df_test)
    M_conf = make_meta(df_conf)
    
    y_train = torch.tensor(df_train.iloc[idx_sub][['defect_x_over_L', 'defect_y_over_H']].values, dtype=torch.float32)
    y_test = torch.tensor(df_test[['defect_x_over_L', 'defect_y_over_H']].values, dtype=torch.float32)
    y_conf = torch.tensor(df_conf[['defect_x_over_L', 'defect_y_over_H']].values, dtype=torch.float32)
    
    y_test_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    y_conf_mm = np.column_stack([df_conf['defect_center_x_mm'].values, df_conf['defect_center_y_mm'].values])
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    L_conf = df_conf['plate_length_mm'].values
    H_conf = df_conf['plate_height_mm'].values
    
    print(f"Training Model D4 on {N_TRAIN} samples...", flush=True)
    cnn_model = DefectInverseCNN(in_channels=1, meta_dim=2, out_dim=2).to(DEVICE)
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train_field, M_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    cnn_model.train()
    n_epochs = 6
    t0 = time.time()
    for ep in range(n_epochs):
        ep_loss = 0.0
        for f_batch, m_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = cnn_model(f_batch, m_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(f_batch)
        print(f"  CNN Epoch {ep+1:02d}/{n_epochs:02d} - Loss: {ep_loss / len(train_dataset):.6f}", flush=True)
            
    print(f"Model D4 trained in {time.time() - t0:.1f} s", flush=True)
    torch.save(cnn_model.state_dict(), os.path.join(MODEL_DIR, "model_D4_cnn.pt"))
    
    cnn_model.eval()
    with torch.no_grad():
        pred_test_norm = cnn_model(X_test_field, M_test).cpu().numpy()
        pred_test_mm = np.column_stack([pred_test_norm[:, 0] * L_test, pred_test_norm[:, 1] * H_test])
        m_test_d4 = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        pred_conf_norm = cnn_model(X_conf_field, M_conf).cpu().numpy()
        pred_conf_mm = np.column_stack([pred_conf_norm[:, 0] * L_conf, pred_conf_norm[:, 1] * H_conf])
        m_conf_d4 = compute_localization_metrics(y_conf_mm, pred_conf_mm)
        
    print(f"  [Model D4 CNN] Test Mean Euc: {m_test_d4['Mean_Euclidean_mm']:.2f} mm | Conf: {m_conf_d4['Mean_Euclidean_mm']:.2f} mm", flush=True)
    
    with open(os.path.join(MODEL_DIR, "model_D4_metrics.json"), 'w') as f:
        json.dump({'test': m_test_d4, 'confirmatory': m_conf_d4}, f, indent=2)
        
    # -----------------------------------------------------------------
    # PHASE H: MODEL D5 U-NET HEATMAP & UNCERTAINTY MODEL
    # -----------------------------------------------------------------
    print("\n=======================================================", flush=True)
    print("   PHASE H: MODEL D5 U-NET HEATMAP & UNCERTAINTY MODEL  ", flush=True)
    print("=======================================================", flush=True)
    
    H_train_np = generate_gaussian_heatmaps(y_train.numpy(), grid_res=GRID_RES, sigma=0.04)
    H_train = torch.tensor(H_train_np, dtype=torch.float32)
    
    unet_model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    unet_opt = torch.optim.AdamW(unet_model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()
    
    unet_dataset = TensorDataset(X_train_field, H_train)
    unet_loader = DataLoader(unet_dataset, batch_size=32, shuffle=True)
    
    print("Training U-Net Heatmap predictor...", flush=True)
    t0 = time.time()
    n_epochs_unet = 5
    for ep in range(n_epochs_unet):
        ep_loss = 0.0
        for f_batch, h_batch in unet_loader:
            unet_opt.zero_grad()
            logits = unet_model(f_batch)
            loss = bce_loss(logits, h_batch)
            loss.backward()
            unet_opt.step()
            ep_loss += loss.item() * len(f_batch)
        print(f"  U-Net Epoch {ep+1:02d}/{n_epochs_unet:02d} - Loss: {ep_loss / len(unet_dataset):.6f}", flush=True)
            
    print(f"Model D5 U-Net trained in {time.time() - t0:.1f} s", flush=True)
    torch.save(unet_model.state_dict(), os.path.join(MODEL_DIR, "model_D5_unet.pt"))
    
    unet_model.eval()
    with torch.no_grad():
        logits_test = unet_model(X_test_field)
        coords_test_norm, unc_test_norm, cov_test = soft_argmax_2d(logits_test, beta=40.0)
        
        pred_test_mm = np.column_stack([
            coords_test_norm[:, 0].cpu().numpy() * L_test,
            coords_test_norm[:, 1].cpu().numpy() * H_test
        ])
        unc_test_mm = unc_test_norm.cpu().numpy() * np.sqrt(L_test * H_test)
        m_test_d5 = compute_localization_metrics(y_test_mm, pred_test_mm)
        m_test_d5['Mean_Uncertainty_Radius_mm'] = float(np.mean(unc_test_mm))
        
        logits_conf = unet_model(X_conf_field)
        coords_conf_norm, unc_conf_norm, cov_conf = soft_argmax_2d(logits_conf, beta=40.0)
        
        pred_conf_mm = np.column_stack([
            coords_conf_norm[:, 0].cpu().numpy() * L_conf,
            coords_conf_norm[:, 1].cpu().numpy() * H_conf
        ])
        unc_conf_mm = unc_conf_norm.cpu().numpy() * np.sqrt(L_conf * H_conf)
        m_conf_d5 = compute_localization_metrics(y_conf_mm, pred_conf_mm)
        m_conf_d5['Mean_Uncertainty_Radius_mm'] = float(np.mean(unc_conf_mm))
        
    print(f"  [Model D5 U-Net] Test Mean Euc: {m_test_d5['Mean_Euclidean_mm']:.2f} mm | Uncertainty: {m_test_d5['Mean_Uncertainty_Radius_mm']:.2f} mm", flush=True)
    print(f"  [Model D5 U-Net] Conf Mean Euc: {m_conf_d5['Mean_Euclidean_mm']:.2f} mm | Uncertainty: {m_conf_d5['Mean_Uncertainty_Radius_mm']:.2f} mm", flush=True)
    
    with open(os.path.join(MODEL_DIR, "model_D5_metrics.json"), 'w') as f:
        json.dump({'test': m_test_d5, 'confirmatory': m_conf_d5}, f, indent=2)
        
    print("All Deep Learning experiments complete!", flush=True)

if __name__ == '__main__':
    run_deep_learning_pipeline()
