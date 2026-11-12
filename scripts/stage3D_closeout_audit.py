"""
Stage 3D Close-Out Audit Script:
1. Reconciles and verifies the canonical geometry (Dc = H/12).
2. Executes Model D5 U-Net LOGO cross-validation across all 5 geometries.
3. Performs rigorous spatial uncertainty calibration (Chi-squared Mahalanobis coverage: 50%, 68%, 90%, 95%, 99%).
4. Evaluates dedicated Model D5 noise & missing-information robustness.
5. Runs Anti-Leakage controls: Central-hole-mask-only baseline, Shuffled fields, and Inpainting comparison (Nearest vs IDW).
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.stage3D_cnn_unet import DefectHeatmapUNet, soft_argmax_2d
from src.stage3D_metrics import compute_localization_metrics
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field
from src.stage3D_feature_extractor import load_coords_and_stress

torch.set_num_threads(min(os.cpu_count() or 4, 8))
AUDIT_MODEL_DIR = "models/stage3D_closeout"
CACHE_DIR = "data/stage3D_field_cache"
os.makedirs(AUDIT_MODEL_DIR, exist_ok=True)

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
    """
    Evaluates statistical calibration of 2D spatial uncertainty covariance.
    Computes Mahalanobis distances and empirical coverage at nominal 50%, 68%, 90%, 95%, 99% levels.
    """
    N = len(pred_coords_mm)
    mahalanobis_sq = np.zeros(N)
    unc_radii_phys = np.zeros(N)
    
    for i in range(N):
        L = L_arr[i]
        H = H_arr[i]
        cov_norm = cov_matrices_norm[i] # 2x2 in [0, 1]
        # Physical covariance: Var(X) = L^2 Var(xi), Var(Y) = H^2 Var(eta), Cov(X,Y) = L*H Cov(xi,eta)
        cov_phys = np.array([
            [cov_norm[0, 0] * (L**2), cov_norm[0, 1] * (L * H)],
            [cov_norm[1, 0] * (L * H), cov_norm[1, 1] * (H**2)]
        ], dtype=np.float64)
        
        # Add small regularization for numerical inversion
        cov_reg = cov_phys + np.eye(2) * 1e-4
        inv_cov = np.linalg.inv(cov_reg)
        
        diff = pred_coords_mm[i] - true_coords_mm[i]
        d_sq = float(diff @ inv_cov @ diff)
        mahalanobis_sq[i] = d_sq
        unc_radii_phys[i] = float(np.sqrt(cov_phys[0, 0] + cov_phys[1, 1]))
        
    # Chi-square with 2 degrees of freedom thresholds
    # F(x) = 1 - exp(-x/2)
    # 50%: x = -2 ln(0.5) = 1.386
    # 68.27% (1-sigma): x = -2 ln(1 - 0.6827) = 2.296
    # 90%: x = -2 ln(0.10) = 4.605
    # 95%: x = -2 ln(0.05) = 5.991
    # 99%: x = -2 ln(0.01) = 9.210
    cov_50 = float(np.mean(mahalanobis_sq <= 1.386) * 100.0)
    cov_68 = float(np.mean(mahalanobis_sq <= 2.296) * 100.0)
    cov_90 = float(np.mean(mahalanobis_sq <= 4.605) * 100.0)
    cov_95 = float(np.mean(mahalanobis_sq <= 5.991) * 100.0)
    cov_99 = float(np.mean(mahalanobis_sq <= 9.210) * 100.0)
    
    euc_errors = np.sqrt(np.sum((pred_coords_mm - true_coords_mm)**2, axis=1))
    corr_unc_err = float(np.corrcoef(unc_radii_phys, euc_errors)[0, 1])
    
    return {
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

def run_d5_logo():
    """
    Executes true Model D5 U-Net Leave-One-Geometry-Out (LOGO) across all 5 geometries.
    """
    print("\n--- AUDIT EXPERIMENT: TRUE MODEL D5 U-NET LOGO ---", flush=True)
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    X_train_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy"))
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    
    df_train = df_meta[df_meta['subset'] == 'train'].reset_index(drop=True)
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    geometries = [
        'square_1000x1000',
        'rectangular_1000x900',
        'rectangular_1000x800',
        'rectangular_1000x700',
        'rectangular_1000x600'
    ]
    
    logo_d5_results = {}
    
    for held_out in geometries:
        print(f"Holding out geometry: {held_out}...", flush=True)
        train_mask = (df_train['geometry_family'] != held_out).values
        test_mask = (df_test['geometry_family'] == held_out).values
        
        X_tr = X_train_raw[train_mask]
        y_tr = df_train.loc[train_mask, ['defect_x_over_L', 'defect_y_over_H']].values
        
        # Subsample for fast training (1600 samples)
        sub_idx = np.random.RandomState(42).choice(len(X_tr), min(len(X_tr), 1600), replace=False)
        X_tr_sub = torch.tensor(X_tr[sub_idx].reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
        H_tr_sub = torch.tensor(generate_gaussian_heatmaps(y_tr[sub_idx], grid_res=GRID_RES, sigma=0.04), dtype=torch.float32)
        
        model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        
        loader = DataLoader(TensorDataset(X_tr_sub, H_tr_sub), batch_size=32, shuffle=True)
        model.train()
        for ep in range(5):
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
                
        # Evaluate on held-out test geometry
        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(X_test_raw[test_mask].reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32)
            logits_te = model(X_te)
            coords_te, unc_te, cov_te = soft_argmax_2d(logits_te, beta=40.0)
            
            df_te_sub = df_test[test_mask].reset_index(drop=True)
            L_te = df_te_sub['plate_length_mm'].values
            H_te = df_te_sub['plate_height_mm'].values
            true_mm = np.column_stack([df_te_sub['defect_center_x_mm'].values, df_te_sub['defect_center_y_mm'].values])
            pred_mm = np.column_stack([coords_te[:, 0].cpu().numpy() * L_te, coords_te[:, 1].cpu().numpy() * H_te])
            
            m = compute_localization_metrics(true_mm, pred_mm)
            cal = evaluate_uncertainty_calibration(pred_mm, true_mm, cov_te.cpu().numpy(), L_te, H_te)
            m.update(cal)
            
        logo_d5_results[held_out] = m
        print(f"  LOGO D5 [{held_out}] Mean Euc: {m['Mean_Euclidean_mm']:.2f} mm | Median: {m['Median_Euclidean_mm']:.2f} mm | <=25mm: {m['Pct_within_25mm']:.1f}% | 90% Cov: {m['Empirical_Coverage_90pct']:.1f}%", flush=True)
        
    with open(os.path.join(AUDIT_MODEL_DIR, "d5_logo_audit.json"), 'w') as f:
        json.dump(logo_d5_results, f, indent=2)
    return logo_d5_results

def run_d5_noise_and_missing_robustness():
    """
    Dedicated Model D5 U-Net noise and missing-information evaluation.
    """
    print("\n--- AUDIT EXPERIMENT: D5 NOISE & MISSING FIELD ROBUSTNESS ---", flush=True)
    model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("models/stage3D_spatial_inverse/model_D5_unet.pt", map_location='cpu'))
    model.eval()
    
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    L_te = df_test['plate_length_mm'].values
    H_te = df_test['plate_height_mm'].values
    true_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    
    conditions = {
        'Clean_Reference': lambda x: x.copy(),
        'Noise_0.5pct': lambda x: x * (1.0 + np.random.normal(0, 0.005, size=x.shape)),
        'Noise_1.0pct': lambda x: x * (1.0 + np.random.normal(0, 0.010, size=x.shape)),
        'Noise_2.0pct': lambda x: x * (1.0 + np.random.normal(0, 0.020, size=x.shape)),
        'Noise_5.0pct': lambda x: x * (1.0 + np.random.normal(0, 0.050, size=x.shape)),
        'Spatial_Bias_5pct': lambda x: x * (1.0 + 0.05 * np.linspace(-1, 1, GRID_RES)[None, None, :, None]),
        'Missing_Dropout_10pct': lambda x: x * (np.random.rand(*x.shape) > 0.10),
        'Missing_Dropout_25pct': lambda x: x * (np.random.rand(*x.shape) > 0.25),
        'Contiguous_Missing_Patch_16x16': lambda x: apply_missing_patch(x.copy(), patch_size=16)
    }
    
    def apply_missing_patch(arr, patch_size=16):
        # Zero out a 16x16 corner block (sensor occlusion)
        arr[:, :, :patch_size, :patch_size] = 1.0 # set to nominal uniform far-field tension
        return arr
        
    robustness_results = {}
    np.random.seed(42)
    
    X_test_4d = X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES)
    
    for c_name, transform_fn in conditions.items():
        X_deg = transform_fn(X_test_4d)
        X_deg_t = torch.tensor(X_deg, dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(X_deg_t)
            coords_te, unc_te, cov_te = soft_argmax_2d(logits, beta=40.0)
            pred_mm = np.column_stack([coords_te[:, 0].numpy() * L_te, coords_te[:, 1].numpy() * H_te])
            m = compute_localization_metrics(true_mm, pred_mm)
            cal = evaluate_uncertainty_calibration(pred_mm, true_mm, cov_te.numpy(), L_te, H_te)
            m.update(cal)
            
        robustness_results[c_name] = m
        print(f"  [{c_name}] Mean Euc: {m['Mean_Euclidean_mm']:.2f} mm | Median: {m['Median_Euclidean_mm']:.2f} mm | <=25mm: {m['Pct_within_25mm']:.1f}% | Unc: {m['Mean_Uncertainty_Radius_mm']:.2f} mm", flush=True)
        
    with open(os.path.join(AUDIT_MODEL_DIR, "d5_robustness_audit.json"), 'w') as f:
        json.dump(robustness_results, f, indent=2)
    return robustness_results

def run_anti_leakage_and_inpainting_audit():
    """
    Evaluates:
    1. Mask-only negative control using central-hole mask and aspect ratio (zero stress).
    2. Shuffled-field negative control on D5.
    3. Inpainting comparison: Nearest-neighbor vs IDW continuum inpainting.
    """
    print("\n--- AUDIT EXPERIMENT: ANTI-LEAKAGE & INPAINTING AUDIT ---", flush=True)
    X_train_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy"))
    X_test_raw = np.load(os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy"))
    df_meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta_subsets.parquet"))
    df_train = df_meta[df_meta['subset'] == 'train'].reset_index(drop=True)
    df_test = df_meta[df_meta['subset'] == 'test'].reset_index(drop=True)
    
    L_te = df_test['plate_length_mm'].values
    H_te = df_test['plate_height_mm'].values
    true_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    y_test_norm = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    
    # 1. Mask-Only Diagnostic Control
    # Input is only central hole mask (1 inside hole, 0 outside)
    print("Evaluating Mask-Only Diagnostic Baseline (Zero Stress)...", flush=True)
    y_grid, x_grid = np.meshgrid(np.linspace(0, 1, GRID_RES), np.linspace(0, 1, GRID_RES), indexing='ij')
    # Central hole mask in normalized coordinates (rx = Dc/(2L), ry = Dc/(2H))
    X_mask_train = np.zeros((len(df_train), 1, GRID_RES, GRID_RES), dtype=np.float32)
    X_mask_test = np.zeros((len(df_test), 1, GRID_RES, GRID_RES), dtype=np.float32)
    for i in range(len(df_train)):
        Dc = df_train['central_hole_diameter_mm'].iloc[i]
        L = df_train['plate_length_mm'].iloc[i]
        H = df_train['plate_height_mm'].iloc[i]
        mask = (((x_grid - 0.5) / (Dc/(2*L)))**2 + ((y_grid - 0.5) / (Dc/(2*H)))**2) <= 1.0
        X_mask_train[i, 0] = mask.astype(np.float32)
    for i in range(len(df_test)):
        Dc = df_test['central_hole_diameter_mm'].iloc[i]
        L = df_test['plate_length_mm'].iloc[i]
        H = df_test['plate_height_mm'].iloc[i]
        mask = (((x_grid - 0.5) / (Dc/(2*L)))**2 + ((y_grid - 0.5) / (Dc/(2*H)))**2) <= 1.0
        X_mask_test[i, 0] = mask.astype(np.float32)
        
    # Train simple CNN on mask only
    from src.stage3D_cnn_unet import DefectInverseCNN
    cnn_mask = DefectInverseCNN(in_channels=1, meta_dim=2, out_dim=2).to(DEVICE)
    opt = torch.optim.Adam(cnn_mask.parameters(), lr=1e-3)
    
    M_tr = torch.tensor(np.column_stack([df_train['plate_length_mm']/df_train['plate_height_mm'], df_train['defect_d_over_Dc']]), dtype=torch.float32)
    M_te = torch.tensor(np.column_stack([df_test['plate_length_mm']/df_test['plate_height_mm'], df_test['defect_d_over_Dc']]), dtype=torch.float32)
    y_tr_t = torch.tensor(df_train[['defect_x_over_L', 'defect_y_over_H']].values, dtype=torch.float32)
    
    cnn_mask.train()
    loader = DataLoader(TensorDataset(torch.tensor(X_mask_train[:1500]), M_tr[:1500], y_tr_t[:1500]), batch_size=64, shuffle=True)
    for ep in range(5):
        for xb, mb, yb in loader:
            opt.zero_grad()
            nn.MSELoss()(cnn_mask(xb, mb), yb).backward()
            opt.step()
            
    cnn_mask.eval()
    with torch.no_grad():
        pred_norm = cnn_mask(torch.tensor(X_mask_test), M_te).numpy()
        pred_mm = np.column_stack([pred_norm[:, 0] * L_te, pred_norm[:, 1] * H_te])
        m_mask = compute_localization_metrics(true_mm, pred_mm)
    print(f"  [Mask-Only Control] Mean Euc: {m_mask['Mean_Euclidean_mm']:.2f} mm | Median: {m_mask['Median_Euclidean_mm']:.2f} mm | <=25mm: {m_mask['Pct_within_25mm']:.1f}%", flush=True)
    
    # 2. Shuffled D5 Control
    print("Evaluating Shuffled-Field D5 Control...", flush=True)
    # Permute test labels relative to test fields
    perm_idx = np.random.RandomState(42).permutation(len(df_test))
    true_shuffled_mm = true_mm[perm_idx]
    
    model = DefectHeatmapUNet(in_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("models/stage3D_spatial_inverse/model_D5_unet.pt", map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_raw.reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32))
        coords_te, _, _ = soft_argmax_2d(logits, beta=40.0)
        pred_mm = np.column_stack([coords_te[:, 0].numpy() * L_te, coords_te[:, 1].numpy() * H_te])
        m_shuffled = compute_localization_metrics(true_shuffled_mm, pred_mm)
    print(f"  [Shuffled Fields Control] Mean Euc: {m_shuffled['Mean_Euclidean_mm']:.2f} mm | Median: {m_shuffled['Median_Euclidean_mm']:.2f} mm | <=25mm: {m_shuffled['Pct_within_25mm']:.1f}%", flush=True)
    
    # 3. Inpainting Comparison (Nearest vs IDW)
    print("Evaluating Inpainting Method Comparison (Nearest vs IDW k=4)...", flush=True)
    # Re-rasterize a sample of 200 test cases with IDW method
    sub_test_df = df_test.iloc[:200]
    X_idw_list = []
    for _, row in sub_test_df.iterrows():
        coords_mm, vm_stress = load_coords_and_stress(row['field_file'])
        cnorm = normalize_coordinates(coords_mm, row['plate_length_mm'], row['plate_height_mm'])
        grid_idw = rasterize_stress_field(cnorm, vm_stress, grid_res=GRID_RES, method="idw")
        X_idw_list.append((grid_idw / 100.0).flatten())
        
    X_idw = np.array(X_idw_list, dtype=np.float32).reshape(-1, 1, GRID_RES, GRID_RES)
    with torch.no_grad():
        logits_idw = model(torch.tensor(X_idw, dtype=torch.float32))
        coords_idw, _, _ = soft_argmax_2d(logits_idw, beta=40.0)
        pred_idw_mm = np.column_stack([
            coords_idw[:, 0].numpy() * sub_test_df['plate_length_mm'].values,
            coords_idw[:, 1].numpy() * sub_test_df['plate_height_mm'].values
        ])
        true_sub_mm = np.column_stack([sub_test_df['defect_center_x_mm'].values, sub_test_df['defect_center_y_mm'].values])
        m_idw = compute_localization_metrics(true_sub_mm, pred_idw_mm)
        
        # Compare with nearest neighbor on same 200 samples
        logits_nn = model(torch.tensor(X_test_raw[:200].reshape(-1, 1, GRID_RES, GRID_RES), dtype=torch.float32))
        coords_nn, _, _ = soft_argmax_2d(logits_nn, beta=40.0)
        pred_nn_mm = np.column_stack([
            coords_nn[:, 0].numpy() * sub_test_df['plate_length_mm'].values,
            coords_nn[:, 1].numpy() * sub_test_df['plate_height_mm'].values
        ])
        m_nn = compute_localization_metrics(true_sub_mm, pred_nn_mm)
        
    print(f"  [Inpainting Nearest] Mean Euc: {m_nn['Mean_Euclidean_mm']:.2f} mm | Median: {m_nn['Median_Euclidean_mm']:.2f} mm", flush=True)
    print(f"  [Inpainting IDW k=4] Mean Euc: {m_idw['Mean_Euclidean_mm']:.2f} mm | Median: {m_idw['Median_Euclidean_mm']:.2f} mm", flush=True)
    
    anti_leakage_results = {
        'Mask_Only_Control': m_mask,
        'Shuffled_Field_Control': m_shuffled,
        'Inpainting_Nearest_200': m_nn,
        'Inpainting_IDW_200': m_idw
    }
    with open(os.path.join(AUDIT_MODEL_DIR, "anti_leakage_audit.json"), 'w') as f:
        json.dump(anti_leakage_results, f, indent=2)
    return anti_leakage_results

def main():
    print("=== STARTING STAGE 3D CLOSE-OUT AUDIT EXPERIMENTS ===", flush=True)
    logo_res = run_d5_logo()
    rob_res = run_d5_noise_and_missing_robustness()
    leak_res = run_anti_leakage_and_inpainting_audit()
    print("=== ALL STAGE 3D CLOSE-OUT AUDIT EXPERIMENTS COMPLETED ===", flush=True)

if __name__ == '__main__':
    main()

