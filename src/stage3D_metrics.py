"""
Evaluation metrics suite for Stage 3D Spatial Inverse Defect Localization.
Strict adherence to physical millimeter errors and tolerance thresholds.
"""
import numpy as np
import pandas as pd

def compute_localization_metrics(y_true_coords_mm, y_pred_coords_mm, y_true_d_mm=None, y_pred_d_mm=None):
    """
    Computes Euclidean localization metrics in millimeters.
    
    Parameters:
    -----------
    y_true_coords_mm : np.ndarray of shape (N, 2), columns [X_mm, Y_mm]
    y_pred_coords_mm : np.ndarray of shape (N, 2), columns [X_mm, Y_mm]
    y_true_d_mm      : optional np.ndarray of shape (N,), defect diameter in mm
    y_pred_d_mm      : optional np.ndarray of shape (N,), predicted diameter in mm
    
    Returns:
    --------
    dict of physical performance metrics
    """
    diff_x = y_pred_coords_mm[:, 0] - y_true_coords_mm[:, 0]
    diff_y = y_pred_coords_mm[:, 1] - y_true_coords_mm[:, 1]
    
    mae_x = float(np.mean(np.abs(diff_x)))
    mae_y = float(np.mean(np.abs(diff_y)))
    rmse_x = float(np.sqrt(np.mean(diff_x**2)))
    rmse_y = float(np.sqrt(np.mean(diff_y**2)))
    
    euc_errors = np.sqrt(diff_x**2 + diff_y**2)
    
    mean_euc = float(np.mean(euc_errors))
    median_euc = float(np.median(euc_errors))
    p95_euc = float(np.percentile(euc_errors, 95))
    max_euc = float(np.max(euc_errors))
    
    # Tolerances
    n = len(euc_errors)
    pct_5mm = float(np.sum(euc_errors <= 5.0) / n * 100.0)
    pct_10mm = float(np.sum(euc_errors <= 10.0) / n * 100.0)
    pct_25mm = float(np.sum(euc_errors <= 25.0) / n * 100.0)
    pct_50mm = float(np.sum(euc_errors <= 50.0) / n * 100.0)
    pct_100mm = float(np.sum(euc_errors <= 100.0) / n * 100.0)
    
    res = {
        'MAE_X_mm': mae_x,
        'MAE_Y_mm': mae_y,
        'RMSE_X_mm': rmse_x,
        'RMSE_Y_mm': rmse_y,
        'Mean_Euclidean_mm': mean_euc,
        'Median_Euclidean_mm': median_euc,
        'P95_Euclidean_mm': p95_euc,
        'Max_Euclidean_mm': max_euc,
        'Pct_within_5mm': pct_5mm,
        'Pct_within_10mm': pct_10mm,
        'Pct_within_25mm': pct_25mm,
        'Pct_within_50mm': pct_50mm,
        'Pct_within_100mm': pct_100mm,
        'N_samples': n
    }
    
    if y_true_d_mm is not None and y_pred_d_mm is not None:
        diff_d = y_pred_d_mm - y_true_d_mm
        mae_d = float(np.mean(np.abs(diff_d)))
        rmse_d = float(np.sqrt(np.mean(diff_d**2)))
        ss_res = np.sum(diff_d**2)
        ss_tot = np.sum((y_true_d_mm - np.mean(y_true_d_mm))**2)
        r2_d = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        res.update({
            'MAE_D_mm': mae_d,
            'RMSE_D_mm': rmse_d,
            'R2_D': r2_d
        })
        
    return res

def format_metrics_table(metrics_dict_list, titles):
    """
    Format list of metrics into a clean Markdown table.
    """
    rows = []
    for m, t in zip(metrics_dict_list, titles):
        row = {
            'Model / Configuration': t,
            'Mean Euc (mm)': f"{m['Mean_Euclidean_mm']:.2f}",
            'Median Euc (mm)': f"{m['Median_Euclidean_mm']:.2f}",
            'P95 Euc (mm)': f"{m['P95_Euclidean_mm']:.2f}",
            'Max Euc (mm)': f"{m['Max_Euclidean_mm']:.2f}",
            '≤ 10mm (%)': f"{m['Pct_within_10mm']:.1f}%",
            '≤ 25mm (%)': f"{m['Pct_within_25mm']:.1f}%",
            '≤ 50mm (%)': f"{m['Pct_within_50mm']:.1f}%"
        }
        if 'MAE_D_mm' in m:
            row['MAE_D (mm)'] = f"{m['MAE_D_mm']:.2f}"
            row['R²_D'] = f"{m['R2_D']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)

