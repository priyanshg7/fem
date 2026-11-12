"""
Stage 3D Model Development and Evaluation:
Implements and evaluates:
- Model D1: Regional Peak & Statistical Stress Features
- Model D2: Fixed Sensor Array Observations (4, 8, 16, 32, 64 sensors)
For both Experiment E1 (Known Diameter -> [X, Y]) and Experiment E2 (Unknown Diameter -> [X, Y, D]).
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.stage3D_metrics import compute_localization_metrics

MODEL_DIR = "models/stage3D_spatial_inverse"
os.makedirs(MODEL_DIR, exist_ok=True)

def run_experiment_d1(df_train, df_val, df_test, df_conf):
    """
    Model D1: Regional Peak & Statistical Stress Features.
    Features: 16 regional stress metrics + aspect_ratio.
    """
    print("\n=======================================================")
    print("      PHASE D: MODEL D1 REGIONAL STRESS INVERSION      ")
    print("=======================================================")
    
    d1_cols = [
        'D1_Kt_max', 'D1_Kt_Q1', 'D1_Kt_Q2', 'D1_Kt_Q3', 'D1_Kt_Q4',
        'D1_mean_Q1', 'D1_mean_Q2', 'D1_mean_Q3', 'D1_mean_Q4',
        'D1_p90_Q1', 'D1_p90_Q2', 'D1_p90_Q3', 'D1_p90_Q4',
        'D1_Kt_ring', 'D1_mean_ring', 'D1_p90_ring', 'aspect_ratio'
    ]
    
    # ---------------------------------------------------------
    # Experiment E1: Known Defect Diameter -> [defect_x_over_L, defect_y_over_H]
    # ---------------------------------------------------------
    feat_e1 = d1_cols + ['defect_d_over_Dc']
    X_train_e1 = df_train[feat_e1].values
    X_val_e1 = df_val[feat_e1].values
    X_test_e1 = df_test[feat_e1].values
    X_conf_e1 = df_conf[feat_e1].values
    
    y_train_e1 = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_val_e1 = df_val[['defect_x_over_L', 'defect_y_over_H']].values
    y_test_e1 = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    y_conf_e1 = df_conf[['defect_x_over_L', 'defect_y_over_H']].values
    
    # Truth in mm
    def get_coords_mm(df):
        return np.column_stack([df['defect_center_x_mm'].values, df['defect_center_y_mm'].values])
        
    y_test_mm = get_coords_mm(df_test)
    y_conf_mm = get_coords_mm(df_conf)
    
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    L_conf = df_conf['plate_length_mm'].values
    H_conf = df_conf['plate_height_mm'].values
    
    models = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('reg', MultiOutputRegressor(Ridge(alpha=10.0)))]),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_split=5, n_jobs=-1, random_state=42),
        'HistGradientBoosting': MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42)),
        'ANN-256': Pipeline([('scaler', StandardScaler()), ('reg', MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=42))])
    }
    
    results_e1 = {}
    best_model_name = None
    best_mean_euc = float('inf')
    best_model_obj = None
    
    for name, model in models.items():
        print(f"Training D1 E1 ({name})...")
        t0 = time.time()
        model.fit(X_train_e1, y_train_e1)
        t_fit = time.time() - t0
        
        # Test evaluation
        pred_test_norm = model.predict(X_test_e1)
        pred_test_mm = np.column_stack([pred_test_norm[:, 0] * L_test, pred_test_norm[:, 1] * H_test])
        m_test = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        # Confirmatory evaluation
        pred_conf_norm = model.predict(X_conf_e1)
        pred_conf_mm = np.column_stack([pred_conf_norm[:, 0] * L_conf, pred_conf_norm[:, 1] * H_conf])
        m_conf = compute_localization_metrics(y_conf_mm, pred_conf_mm)
        
        results_e1[name] = {
            'fit_time_s': t_fit,
            'test': m_test,
            'confirmatory': m_conf
        }
        
        print(f"  [{name}] Test Mean Euc: {m_test['Mean_Euclidean_mm']:.2f} mm (Median: {m_test['Median_Euclidean_mm']:.2f} mm, <=25mm: {m_test['Pct_within_25mm']:.1f}%) | Conf Mean: {m_conf['Mean_Euclidean_mm']:.2f} mm")
        
        if m_test['Mean_Euclidean_mm'] < best_mean_euc:
            best_mean_euc = m_test['Mean_Euclidean_mm']
            best_model_name = name
            best_model_obj = model
            
    # Save best D1 E1 model
    joblib.dump(best_model_obj, os.path.join(MODEL_DIR, "model_D1_E1_best.joblib"))
    print(f"Best D1 E1 model: {best_model_name} (Mean Euc = {best_mean_euc:.2f} mm)")
    
    # ---------------------------------------------------------
    # Experiment E2: Unknown Diameter -> [X/L, Y/H, D/Dc]
    # ---------------------------------------------------------
    print("\nTraining D1 Experiment E2 (Unknown Diameter, Joint [X, Y, D])...")
    X_train_e2 = df_train[d1_cols].values
    X_test_e2 = df_test[d1_cols].values
    X_conf_e2 = df_conf[d1_cols].values
    
    y_train_e2 = df_train[['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc']].values
    y_test_e2 = df_test[['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc']].values
    y_conf_e2 = df_conf[['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc']].values
    
    d_true_test_mm = df_test['defect_diameter_mm'].values
    d_true_conf_mm = df_conf['defect_diameter_mm'].values
    Dc = 200.0
    
    results_e2 = {}
    best_e2_model = None
    best_e2_euc = float('inf')
    
    for name, model in models.items():
        model.fit(X_train_e2, y_train_e2)
        
        # Test eval
        pred_test = model.predict(X_test_e2)
        pred_test_mm = np.column_stack([pred_test[:, 0] * L_test, pred_test[:, 1] * H_test])
        pred_test_d_mm = pred_test[:, 2] * Dc
        m_test = compute_localization_metrics(y_test_mm, pred_test_mm, d_true_test_mm, pred_test_d_mm)
        
        # Conf eval
        pred_conf = model.predict(X_conf_e2)
        pred_conf_mm = np.column_stack([pred_conf[:, 0] * L_conf, pred_conf[:, 1] * H_conf])
        pred_conf_d_mm = pred_conf[:, 2] * Dc
        m_conf = compute_localization_metrics(y_conf_mm, pred_conf_mm, d_true_conf_mm, pred_conf_d_mm)
        
        results_e2[name] = {'test': m_test, 'confirmatory': m_conf}
        print(f"  [E2 {name}] Test Mean Euc: {m_test['Mean_Euclidean_mm']:.2f} mm | MAE_D: {m_test['MAE_D_mm']:.2f} mm | R2_D: {m_test['R2_D']:.4f}")
        
        if m_test['Mean_Euclidean_mm'] < best_e2_euc:
            best_e2_euc = m_test['Mean_Euclidean_mm']
            best_e2_model = model
            
    joblib.dump(best_e2_model, os.path.join(MODEL_DIR, "model_D1_E2_best.joblib"))
    
    with open(os.path.join(MODEL_DIR, "model_D1_metrics.json"), 'w') as f:
        json.dump({'E1': results_e1, 'E2': results_e2}, f, indent=2)
        
    return results_e1, results_e2

def run_experiment_d2(df_train, df_val, df_test, df_conf):
    """
    Model D2: Fixed Sensor Observations (4, 8, 16, 32, 64 sensors).
    """
    print("\n=======================================================")
    print("      PHASE E: MODEL D2 SENSOR ARRAY SCALING STUDY      ")
    print("=======================================================")
    
    sensor_counts = [4, 8, 16, 32, 64]
    results_d2_scaling = {}
    
    y_train = df_train[['defect_x_over_L', 'defect_y_over_H']].values
    y_test = df_test[['defect_x_over_L', 'defect_y_over_H']].values
    y_conf = df_conf[['defect_x_over_L', 'defect_y_over_H']].values
    
    L_test = df_test['plate_length_mm'].values
    H_test = df_test['plate_height_mm'].values
    L_conf = df_conf['plate_length_mm'].values
    H_conf = df_conf['plate_height_mm'].values
    
    y_test_mm = np.column_stack([df_test['defect_center_x_mm'].values, df_test['defect_center_y_mm'].values])
    y_conf_mm = np.column_stack([df_conf['defect_center_x_mm'].values, df_conf['defect_center_y_mm'].values])
    
    for ns in sensor_counts:
        cols = [f'D2_sensor_{i:02d}' for i in range(ns)] + ['aspect_ratio', 'defect_d_over_Dc']
        X_train = df_train[cols].values
        X_test = df_test[cols].values
        X_conf = df_conf[cols].values
        
        print(f"\n--- Testing Sensor Array N = {ns} ---")
        model = ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_split=5, n_jobs=-1, random_state=42)
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_t = time.time() - t0
        
        pred_test = model.predict(X_test)
        pred_test_mm = np.column_stack([pred_test[:, 0] * L_test, pred_test[:, 1] * H_test])
        m_test = compute_localization_metrics(y_test_mm, pred_test_mm)
        
        pred_conf = model.predict(X_conf)
        pred_conf_mm = np.column_stack([pred_conf[:, 0] * L_conf, pred_conf[:, 1] * H_conf])
        m_conf = compute_localization_metrics(y_conf_mm, pred_conf_mm)
        
        results_d2_scaling[ns] = {
            'fit_time_s': fit_t,
            'test': m_test,
            'confirmatory': m_conf
        }
        
        print(f"  N={ns} Test Mean Euc: {m_test['Mean_Euclidean_mm']:.2f} mm (Median: {m_test['Median_Euclidean_mm']:.2f} mm, <=25mm: {m_test['Pct_within_25mm']:.1f}%) | Conf: {m_conf['Mean_Euclidean_mm']:.2f} mm")
        
        # Save model checkpoint
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_D2_sensors_{ns}.joblib"))
        
    with open(os.path.join(MODEL_DIR, "model_D2_metrics.json"), 'w') as f:
        json.dump(results_d2_scaling, f, indent=2)
        
    return results_d2_scaling

