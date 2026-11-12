"""
Stage 2V Phase B - Plotting and Residual Analysis
Generates:
1. Predicted vs Actual Kt_element for all 4 models + committee
2. Residual distribution by geometry family
3. Spatial validation error maps across the 5 geometries
4. Committee disagreement heatmaps
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_pilot_plots():
    print("Generating Stage 2V Pilot Model Figures...")
    val_df_path = 'data/stage2V_preliminary_val_predictions.csv'
    if not os.path.exists(val_df_path):
        print(f"Error: {val_df_path} does not exist.")
        return
        
    df = pd.read_csv(val_df_path)
    os.makedirs('output', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # 1. Predicted vs Actual Kt_element
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    models = ['ExtraTrees', 'RandomForest', 'HistGradientBoosting', 'MLP_NeuralNet']
    
    for ax, name in zip(axes.flatten(), models):
        y_true = df['Kt_element']
        y_pred = df[f'pred_{name}']
        
        ax.scatter(y_true, y_pred, alpha=0.35, s=15, c=df['aspect_ratio'], cmap='viridis')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='1:1 Ideal')
        
        r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        mae_psi = np.mean(np.abs(y_true - y_pred)) * 100.0
        
        ax.set_title(f"{name}\n$R^2 = {r2:.4f}$ | MAE = {mae_psi:.2f} psi", fontsize=13, fontweight='bold')
        ax.set_xlabel('Actual $K_{t,\\text{element}}$')
        ax.set_ylabel('Predicted $K_{t,\\text{element}}$')
        ax.legend(loc='upper left')
        
    plt.tight_layout()
    plt.savefig('output/stage2V_preliminary_pred_vs_actual.png', dpi=300)
    plt.close()
    print("Saved output/stage2V_preliminary_pred_vs_actual.png")
    
    # 2. Residual Distribution by Geometry Family
    plt.figure(figsize=(12, 6))
    geom_order = ['rectangular_1000x600', 'rectangular_1000x700', 'rectangular_1000x800', 'rectangular_1000x900', 'square_1000x1000']
    
    df['committee_resid_psi'] = (df['committee_mean_pred'] - df['Kt_element']) * 100.0
    sns.boxplot(data=df, x='geometry_family', y='committee_resid_psi', order=geom_order, palette='Set2')
    plt.axhline(0, color='r', linestyle='--', lw=1.5)
    plt.title('Stage 2V Committee Residual Distribution by Geometry Family', fontsize=14, fontweight='bold')
    plt.xlabel('Geometry Family')
    plt.ylabel('Prediction Residual $\\hat{K}_t - K_t$ (psi)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('output/stage2V_preliminary_residuals.png', dpi=300)
    plt.close()
    print("Saved output/stage2V_preliminary_residuals.png")
    
    # 3. Spatial Validation Error Map (Absolute Error in Normalized Space)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    
    for ax, g in zip(axes, geom_order):
        sub = df[df['geometry_family'] == g]
        sc = ax.scatter(
            sub['defect_x_over_L'], 
            sub['defect_y_over_H'], 
            c=sub['committee_abs_error'] * 100.0, 
            cmap='plasma', 
            s=22, 
            alpha=0.8,
            vmax=25.0
        )
        ax.set_title(f"{g}\nMax Error = {sub['committee_abs_error'].max()*100.0:.1f} psi", fontsize=11, fontweight='bold')
        ax.set_xlabel('$X / L$')
        if ax == axes[0]:
            ax.set_ylabel('$Y / H$')
            
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.85, pad=0.02)
    cbar.set_label('Absolute Error $|\\hat{K}_t - K_t|$ (psi)', fontsize=12)
    plt.suptitle('Spatial Validation Error Distribution Across Plate Geometries', fontsize=15, fontweight='bold', y=1.03)
    plt.savefig('output/stage2V_spatial_val_error_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved output/stage2V_spatial_val_error_map.png")
    
    # 4. Committee Disagreement Map (Std Dev across 4 models)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    
    for ax, g in zip(axes, geom_order):
        sub = df[df['geometry_family'] == g]
        sc = ax.scatter(
            sub['defect_x_over_L'], 
            sub['defect_y_over_H'], 
            c=sub['committee_std_disagreement'] * 100.0, 
            cmap='inferno', 
            s=22, 
            alpha=0.8,
            vmax=15.0
        )
        ax.set_title(f"{g}\nMax Disagree = {sub['committee_std_disagreement'].max()*100.0:.1f} psi", fontsize=11, fontweight='bold')
        ax.set_xlabel('$X / L$')
        if ax == axes[0]:
            ax.set_ylabel('$Y / H$')
            
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.85, pad=0.02)
    cbar.set_label('Model Disagreement $\\sigma_{\\text{committee}}$ (psi)', fontsize=12)
    plt.suptitle('Model Committee Uncertainty / Disagreement Across Plate Geometries', fontsize=15, fontweight='bold', y=1.03)
    plt.savefig('output/stage2V_committee_disagreement_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved output/stage2V_committee_disagreement_map.png")

if __name__ == '__main__':
    generate_pilot_plots()

