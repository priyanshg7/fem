"""
Stage 3D Phase F: Model D3 — PCA-Reduced Full Stress Field Inversion.
Evaluates dimension reduction with k in [10, 20, 40, 80, 120] principal components.
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from concurrent.futures import ProcessPoolExecutor
from src.stage3D_feature_extractor import load_coords_and_stress
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field
from src.stage3D_metrics import compute_localization_metrics

MODEL_DIR = "models/stage3D_spatial_inverse"
CACHE_DIR = "data/stage3D_field_cache"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

GRID_RES = 64

def rasterize_row(row_dict):
    field_file = row_dict['field_file']
    L = float(row_dict['plate_length_mm'])
    H = float(row_dict['plate_height_mm'])
    try:
        coords_mm, vm_stress = load_coords_and_stress(field_file)
        coords_norm = normalize_coordinates(coords_mm, L, H)
        # Continuum inpainting rasterization
        grid = rasterize_stress_field(coords_norm, vm_stress, grid_res=GRID_RES, method="nearest")
        # Normalize stress to Kt = stress / 100.0 psi
        grid_kt = grid / 100.0
        return grid_kt.flatten()
    except Exception as e:
        print(f"Error rasterizing {field_file}: {e}")
        return np.zeros(GRID_RES * GRID_RES, dtype=np.float32)

def get_or_create_raster_dataset():
    train_cache = os.path.join(CACHE_DIR, f"X_pca_train_{GRID_RES}.npy")
    test_cache = os.path.join(CACHE_DIR, f"X_pca_test_{GRID_RES}.npy")
    conf_cache = os.path.join(CACHE_DIR, f"X_pca_conf_{GRID_RES}.npy")
    meta_cache = os.path.join(CACHE_DIR, "meta_subsets.parquet")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache) and os.path.exists(conf_cache):
        print("Loading precomputed raster field cache...")
        X_train = np.load(train_cache)
        X_test = np.load(test_cache)
        X_conf = np.load(conf_cache)
        df_subsets = pd.read_parquet(meta_cache)
        df_train_sub = df_subsets[df_subsets['subset'] == 'train'].reset_index(drop=True)
        df_test_sub = df_subsets[df_subsets['subset'] == 'test'].reset_index(drop=True)
        df_conf = df_subsets[df_subsets['subset'] == 'conf'].reset_index(drop=True)
        return X_train, X_test, X_conf, df_train_sub, df_test_sub, df_conf
        
    print("Generating stratified raster dataset for PCA & CNN...")
    df_master = pd.read_csv("data/enriched_dataset_51625.csv")
    df_conf = pd.read_csv("data/stage3D_blind_confirmatory/stage3D_confirmatory_metadata.csv")
    
    # Stratified sample: 1,000 per geometry from train, 200 per geometry from test
    train_rows = []
    test_rows = []
    for g in df_master['geometry_family'].unique():
        g_train = df_master[(df_master['geometry_family'] == g) & (df_master['split'] == 'train')].sample(n=1000, random_state=42)
        g_test = df_master[(df_master['geometry_family'] == g) & (df_master['split'] == 'test')].sample(n=200, random_state=42)
        train_rows.append(g_train)
        test_rows.append(g_test)
        
    df_train_sub = pd.concat(train_rows, ignore_index=True)
    df_test_sub = pd.concat(test_rows, ignore_index=True)
    
    df_train_sub['subset'] = 'train'
    df_test_sub['subset'] = 'test'
    df_conf_sub = df_conf.copy()
    df_conf_sub['subset'] = 'conf'
    
    df_all_sub = pd.concat([df_train_sub, df_test_sub, df_conf_sub], ignore_index=True)
    df_all_sub.to_parquet(meta_cache, index=False)
    
    print(f"Rasterizing: Train={len(df_train_sub)}, Test={len(df_test_sub)}, Conf={len(df_conf_sub)}...")
    n_workers = min(os.cpu_count() or 4, 12)
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        X_train_list = list(executor.map(rasterize_row, df_train_sub.to_dict('records')))
        X_test_list = list(executor.map(rasterize_row, df_test_sub.to_dict('records')))
        X_conf_list = list(executor.map(rasterize_row, df_conf_sub.to_dict('records')))
        
    X_train = np.array(X_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    X_conf = np.array(X_conf_list, dtype=np.float32)
    
    np.save(train_cache, X_train)
    np.save(test_cache, X_test)
    np.save(conf_cache, X_conf)
    print(f"Rasterization complete in {time.time() - t0:.1f} s! Cached to {CACHE_DIR}")
    
    return X_train, X_test, X_conf, df_train_sub, df_test_sub, df_conf_sub

def run_pca_experiments():
    print("\n=======================================================")
    print("      PHASE F: MODEL D3 PCA-REDUCED FULL-FIELD INVERSION ")
    print("=======================================================")
    
    X_train_raw, X_test_raw, X_conf_raw, df_train, df_test, df_conf = get_or_create_raster_dataset()
    
    print(f"Fitting PCA (120 components) on X_train (shape {X_train_raw.shape})...")
    t0 = time.time()
    pca = PCA(n_components=120, svd_solver='randomized', random_state=42)
    pca.fit(X_train_raw)
    print(f"PCA fit in {time.time() - t0:.1f} s. Explained variance ratio (top 10): {pca.explained_variance_ratio_[:10].sum():.4f}, (120): {pca.explained_variance_ratio_.sum():.4f}")
    
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca_field_model.joblib"))
    
    # Pre-transform
    Z_train_120 = pca.transform(X_train_raw)
    Z_test_120 = pca.transform(X_test_raw)
    Z_conf_120 = pca.transform(X_conf_raw)
    
    k_components_list = [10, 20, 40, 80, 120]
    results_d3 = {}
    
    y_train = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_test = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    y_conf = df_conf[['defect_x_over_L', 'defect_y_over_H']].values
    
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    L_conf = df_conf['plate_length_mm'].values
    H_conf = df_conf['plate_height_mm'].values
    
    y_test_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    y_conf_mm = np.column_stack([df_conf['defect_center_x_mm'].values, df_conf['defect_center_y_mm'].values])
    
    ar_train = (df_train['plate_length_mm'] / df_train['plate_height_mm']).values[:, None]
    ar_test = (df_test['plate_length_mm'] / df_test['plate_height_mm']).values[:, None]
    ar_conf = (df_conf['plate_length_mm'] / df_conf['plate_height_mm']).values[:, None]
    
    d_train = df_train['defect_d_over_Dc'].values[:, None]
    d_test = df_test['defect_d_over_Dc'].values[:, None]
    d_conf = df_conf['defect_d_over_Dc'].values[:, None]
    
    for k in k_components_list:
        print(f"\n--- Evaluating PCA Dimension k = {k} ---")
        var_explained = float(pca.explained_variance_ratio_[:k].sum())
        
        # Features: PCA modes + aspect ratio + defect diameter
        feat_train = np.hstack([Z_train_120[:, :k], ar_train, d_train])
        feat_test = np.hstack([Z_test_120[:, :k], ar_test, d_test])
        feat_conf = np.hstack([Z_conf_120[:, :k], ar_conf, d_conf])
        
        # Test ExtraTrees regressor
        model = ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42)
        t_start = time.time()
        model.fit(feat_train, y_train)
        fit_t = time.time() - t_start
        
        pred_test = model.predict(feat_test)
        pred_test_mm = np.column_stack([pred_test[:, 0] * L_test, pred_test[:, 1] * H_test])
        m_test = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        pred_conf = model.predict(feat_conf)
        pred_conf_mm = np.column_stack([pred_conf[:, 0] * L_conf, pred_conf[:, 1] * H_conf])
        m_conf = compute_localization_metrics(y_conf_mm, pred_conf_mm)
        
        results_d3[k] = {
            'variance_explained': var_explained,
            'fit_time_s': fit_t,
            'test': m_test,
            'confirmatory': m_conf
        }
        
        print(f"  k={k} (Var: {var_explained*100:.1f}%) | Test Mean Euc: {m_test['Mean_Euclidean_mm']:.2f} mm (Median: {m_test['Median_Euclidean_mm']:.2f} mm, <=25mm: {m_test['Pct_within_25mm']:.1f}%) | Conf: {m_conf['Mean_Euclidean_mm']:.2f} mm")
        
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_D3_pca_k{k}.joblib"))
        
    with open(os.path.join(MODEL_DIR, "model_D3_metrics.json"), 'w') as f:
        json.dump(results_d3, f, indent=2)
        
    print("Phase F PCA experiments complete!")
    return results_d3

if __name__ == '__main__':
    run_pca_experiments()

