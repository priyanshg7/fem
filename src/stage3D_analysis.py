"""
Stage 3D Specialized Scientific Analysis:
- Phase J: Leave-One-Geometry-Out (LOGO) Cross-Validation
- Phase M: Sensor Noise & Missing-Sensor Sensitivity Analysis
- Phase N: Negative Controls & Anti-Leakage Verification
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from src.stage3D_metrics import compute_localization_metrics

OUTPUT_DIR = "models/stage3D_spatial_inverse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_logo_analysis(df_master):
    """
    Phase J: Leave-One-Geometry-Out (LOGO) cross-validation across 5 geometries.
    """
    print("\n=======================================================")
    print("      PHASE J: LEAVE-ONE-GEOMETRY-OUT (LOGO) STUDY      ")
    print("=======================================================")
    
    geometries = [
        'square_1000x1000',
        'rectangular_1000x900',
        'rectangular_1000x800',
        'rectangular_1000x700',
        'rectangular_1000x600'
    ]
    
    d1_cols = [
        'D1_Kt_max', 'D1_Kt_Q1', 'D1_Kt_Q2', 'D1_Kt_Q3', 'D1_Kt_Q4',
        'D1_mean_Q1', 'D1_mean_Q2', 'D1_mean_Q3', 'D1_mean_Q4',
        'D1_p90_Q1', 'D1_p90_Q2', 'D1_p90_Q3', 'D1_p90_Q4',
        'D1_Kt_ring', 'D1_mean_ring', 'D1_p90_ring', 'aspect_ratio', 'defect_d_over_Dc'
    ]
    
    logo_results = {}
    
    for held_out in geometries:
        print(f"\nHolding out geometry: {held_out}...")
        df_train = df_master[df_master['geometry_family'] != held_out]
        df_test = df_master[df_master['geometry_family'] == held_out]
        
        # Subsample train for fast execution (e.g. 15,000 samples)
        df_train_sub = df_train.sample(n=min(len(df_train), 15000), random_state=42)
        df_test_sub = df_test.sample(n=min(len(df_test), 2000), random_state=42)
        
        X_train = df_train_sub[d1_cols].values
        y_train = df_train_sub[['defect_x_over_L', 'defect_y_over_H']].values
        
        X_test = df_test_sub[d1_cols].values
        y_test_mm = np.column_stack([df_test_sub['defect_center_x_mm'].values, df_test_sub['defect_center_y_mm'].values])
        L_test = df_test_sub['plate_length_mm'].values
        H_test = df_test_sub['plate_height_mm'].values
        
        model = ExtraTreesRegressor(n_estimators=80, max_depth=16, min_samples_split=5, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        pred_test = model.predict(X_test)
        pred_test_mm = np.column_stack([pred_test[:, 0] * L_test, pred_test[:, 1] * H_test])
        m = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        logo_results[held_out] = m
        print(f"  LOGO [{held_out}] Mean Euc: {m['Mean_Euclidean_mm']:.2f} mm | Median: {m['Median_Euclidean_mm']:.2f} mm | <=25mm: {m['Pct_within_25mm']:.1f}%")
        
    with open(os.path.join(OUTPUT_DIR, "logo_results.json"), 'w') as f:
        json.dump(logo_results, f, indent=2)
        
    return logo_results

def run_noise_sensitivity(df_master):
    """
    Phase M: Sensor noise perturbation study (0.5%, 1%, 2%, 5%).
    """
    print("\n=======================================================")
    print("      PHASE M: SENSOR NOISE SENSITIVITY ANALYSIS       ")
    print("=======================================================")
    
    d1_cols = [
        'D1_Kt_max', 'D1_Kt_Q1', 'D1_Kt_Q2', 'D1_Kt_Q3', 'D1_Kt_Q4',
        'D1_mean_Q1', 'D1_mean_Q2', 'D1_mean_Q3', 'D1_mean_Q4',
        'D1_p90_Q1', 'D1_p90_Q2', 'D1_p90_Q3', 'D1_p90_Q4',
        'D1_Kt_ring', 'D1_mean_ring', 'D1_p90_ring', 'aspect_ratio', 'defect_d_over_Dc'
    ]
    
    stress_cols = d1_cols[:-2]
    
    df_train = df_master[df_master['split'] == 'train'].sample(n=10000, random_state=42)
    df_test = df_master[df_master['split'] == 'test'].sample(n=2000, random_state=42)
    
    X_train = df_train[d1_cols].values
    y_train = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    
    model = ExtraTreesRegressor(n_estimators=80, max_depth=16, min_samples_split=5, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    y_test_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    noise_results = {}
    
    np.random.seed(42)
    for nl in noise_levels:
        df_noisy = df_test.copy()
        if nl > 0:
            noise = np.random.normal(0.0, nl, size=df_noisy[stress_cols].shape)
            df_noisy[stress_cols] = df_noisy[stress_cols] * (1.0 + noise)
            
        X_test_noisy = df_noisy[d1_cols].values
        pred_test = model.predict(X_test_noisy)
        pred_test_mm = np.column_stack([pred_test[:, 0] * L_test, pred_test[:, 1] * H_test])
        m = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        noise_results[f"noise_{nl*100:.1f}pct"] = m
        print(f"  Noise {nl*100:.1f}% | Mean Euc: {m['Mean_Euclidean_mm']:.2f} mm | Median: {m['Median_Euclidean_mm']:.2f} mm | <=25mm: {m['Pct_within_25mm']:.1f}%")
        
    with open(os.path.join(OUTPUT_DIR, "noise_sensitivity.json"), 'w') as f:
        json.dump(noise_results, f, indent=2)
        
    return noise_results

def run_negative_controls(df_master):
    """
    Phase N: Negative controls & Anti-leakage verification.
    """
    print("\n=======================================================")
    print("      PHASE N: NEGATIVE CONTROLS & ANTI-LEAKAGE         ")
    print("=======================================================")
    
    df_train = df_master[df_master['split'] == 'train'].sample(n=10000, random_state=42)
    df_test = df_master[df_master['split'] == 'test'].sample(n=2000, random_state=42)
    
    y_train = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_test_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    
    # Negative Control 1: Geometry-only (Aspect Ratio + Defect Diameter, zero stress information)
    print("\nEvaluating Negative Control 1: Specimen Geometry Only (Zero Stress Information)...")
    X_train_nc1 = df_train[['aspect_ratio', 'defect_d_over_Dc']].values
    X_test_nc1 = df_test[['aspect_ratio', 'defect_d_over_Dc']].values
    
    m1 = ExtraTreesRegressor(n_estimators=80, max_depth=16, random_state=42, n_jobs=-1)
    m1.fit(X_train_nc1, y_train)
    pred_nc1 = m1.predict(X_test_nc1)
    pred_nc1_mm = np.column_stack([pred_nc1[:, 0] * L_test, pred_nc1[:, 1] * H_test])
    res_nc1 = compute_localization_metrics(y_test_mm, pred_nc1_mm)
    print(f"  [NC1 Geometry Only] Mean Euc: {res_nc1['Mean_Euclidean_mm']:.2f} mm | Median: {res_nc1['Median_Euclidean_mm']:.2f} mm | <=25mm: {res_nc1['Pct_within_25mm']:.1f}%")
    
    # Negative Control 2: Shuffled stress fields (Destroys spatial correlation)
    print("\nEvaluating Negative Control 2: Shuffled Stress Features (Random Permutation)...")
    d1_cols = [
        'D1_Kt_max', 'D1_Kt_Q1', 'D1_Kt_Q2', 'D1_Kt_Q3', 'D1_Kt_Q4',
        'D1_mean_Q1', 'D1_mean_Q2', 'D1_mean_Q3', 'D1_mean_Q4',
        'D1_p90_Q1', 'D1_p90_Q2', 'D1_p90_Q3', 'D1_p90_Q4',
        'D1_Kt_ring', 'D1_mean_ring', 'D1_p90_ring', 'aspect_ratio', 'defect_d_over_Dc'
    ]
    X_train_nc2 = df_train[d1_cols].copy().values
    # Shuffle each column independently
    for j in range(X_train_nc2.shape[1] - 2):
        np.random.shuffle(X_train_nc2[:, j])
        
    m2 = ExtraTreesRegressor(n_estimators=80, max_depth=16, random_state=42, n_jobs=-1)
    m2.fit(X_train_nc2, y_train)
    pred_nc2 = m2.predict(df_test[d1_cols].values)
    pred_nc2_mm = np.column_stack([pred_nc2[:, 0] * L_test, pred_nc2[:, 1] * H_test])
    res_nc2 = compute_localization_metrics(y_test_mm, pred_nc2_mm)
    print(f"  [NC2 Shuffled Stresses] Mean Euc: {res_nc2['Mean_Euclidean_mm']:.2f} mm | Median: {res_nc2['Median_Euclidean_mm']:.2f} mm | <=25mm: {res_nc2['Pct_within_25mm']:.1f}%")
    
    results_nc = {'NC1_geometry_only': res_nc1, 'NC2_shuffled_stresses': res_nc2}
    with open(os.path.join(OUTPUT_DIR, "negative_controls.json"), 'w') as f:
        json.dump(results_nc, f, indent=2)
        
    return results_nc

def main():
    print("Loading master features dataset for Phase J, M, N...")
    df_master = pd.read_parquet("data/stage3D_features_master.parquet")
    run_logo_analysis(df_master)
    run_noise_sensitivity(df_master)
    run_negative_controls(df_master)
    print("\nAll scientific analysis phases (J, M, N) successfully finished!")

if __name__ == '__main__':
    main()

