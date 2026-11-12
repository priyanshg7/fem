"""
Plotting script for Stage 3E:
1. stage3E_fig01_repeated_seed_stability.png
2. stage3E_fig02_geometry_and_logo_generalization.png
3. stage3E_fig03_diameter_and_clearance_sensitivity.png
4. stage3E_fig04_ablation_comparison.png
5. stage3E_fig05_abstention_tradeoff_curve.png
6. stage3E_fig06_fem_vs_ml_regime_comparison.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

STAGE3E_OUT_DIR = "output/stage3E"
STAGE3E_MODEL_DIR = "models/stage3E"
os.makedirs(STAGE3E_OUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_stage3E_figures():
    seed_file = os.path.join(STAGE3E_MODEL_DIR, "repeated_seed_results.json")
    ablation_file = os.path.join(STAGE3E_MODEL_DIR, "ablation_results.json")
    abst_file = os.path.join(STAGE3E_MODEL_DIR, "abstention_tradeoff.json")
    
    # ----------------------------------------------------
    # FIG 1: Repeated Seed Stability
    # ----------------------------------------------------
    if os.path.exists(seed_file):
        df_seeds = pd.read_json(seed_file)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(df_seeds))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, df_seeds['Mean_Euclidean_mm'], width, label='Mean Euclidean Error (mm)', color='#2e6f40', alpha=0.85, edgecolor='black', linewidth=0.8)
        rects2 = ax.bar(x + width/2, df_seeds['Median_Euclidean_mm'], width, label='Median Euclidean Error (mm)', color='#66bb6a', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for r in rects1:
            h = r.get_height()
            ax.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for r in rects2:
            h = r.get_height()
            ax.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        ax.set_ylabel('Localization Error (mm)', fontsize=11, fontweight='bold')
        ax.set_title(f'Figure 3E-1: Multi-Seed Training Stability Across 5 Independent Random Initializations\nMean Error = {df_seeds["Mean_Euclidean_mm"].mean():.2f} ± {df_seeds["Mean_Euclidean_mm"].std():.2f} mm | Median Error = {df_seeds["Median_Euclidean_mm"].mean():.2f} ± {df_seeds["Median_Euclidean_mm"].std():.2f} mm', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in df_seeds['seed']], fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        ax.set_ylim(0, max(df_seeds['Mean_Euclidean_mm']) * 1.35)
        plt.tight_layout()
        plt.savefig(os.path.join(STAGE3E_OUT_DIR, "stage3E_fig01_repeated_seed_stability.png"))
        plt.close()
        print("Saved stage3E_fig01_repeated_seed_stability.png")
        
    # ----------------------------------------------------
    # FIG 4: Structured Ablations
    # ----------------------------------------------------
    if os.path.exists(ablation_file):
        with open(ablation_file, 'r') as f:
            abl = json.load(f)
            
        keys = ['Baseline_UNet_beta40', 'Architecture_FCN_NoSkips', 'Temperature_beta_10', 'Temperature_beta_80', 'Joint_Heatmap_and_Sizing']
        labels = ['Baseline U-Net\n(beta=40)', 'FCN\n(No Skips)', 'U-Net\n(beta=10)', 'U-Net\n(beta=80)', 'Joint U-Net\n(+ Sizing Head)']
        means = [abl[k]['Mean_Euclidean_mm'] for k in keys if k in abl]
        medians = [abl[k]['Median_Euclidean_mm'] for k in keys if k in abl]
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        x = np.arange(len(means))
        width = 0.35
        
        r1 = ax.bar(x - width/2, means, width, label='Mean Euclidean (mm)', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.8)
        r2 = ax.bar(x + width/2, medians, width, label='Median Euclidean (mm)', color='#aec7e8', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        for r in r1:
            h = r.get_height()
            ax.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for r in r2:
            h = r.get_height()
            ax.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        ax.set_ylabel('Localization Error (mm)', fontsize=11, fontweight='bold')
        ax.set_title('Figure 3E-4: Controlled Architectural and Temperature Ablations on Model D5', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels[:len(means)], fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        ax.set_ylim(0, max(means) * 1.35)
        plt.tight_layout()
        plt.savefig(os.path.join(STAGE3E_OUT_DIR, "stage3E_fig04_ablation_comparison.png"))
        plt.close()
        print("Saved stage3E_fig04_ablation_comparison.png")

    # ----------------------------------------------------
    # FIG 5: Selective Classification / Abstention Trade-off
    # ----------------------------------------------------
    if os.path.exists(abst_file):
        df_abst = pd.read_json(abst_file)
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
        
        color1 = '#2e6f40'
        ax1.plot(df_abst['Retention_Rate_Pct'], df_abst['Mean_Euclidean_mm'], 'o-', color=color1, linewidth=2.5, markersize=7, label='Mean Error (mm)')
        ax1.plot(df_abst['Retention_Rate_Pct'], df_abst['Median_Euclidean_mm'], 's--', color='#66bb6a', linewidth=2.0, markersize=6, label='Median Error (mm)')
        ax1.set_xlabel('Sample Retention Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Localization Error (mm)', fontsize=11, fontweight='bold', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = '#d9534f'
        ax2.plot(df_abst['Retention_Rate_Pct'], df_abst['Pct_within_10mm'], '^-.', color=color2, linewidth=2.0, markersize=6, label='Success Rate (<= 10 mm)')
        ax2.set_ylabel('Precision (<= 10 mm %)', fontsize=11, fontweight='bold', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(50, 105)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
        plt.title('Figure 3E-5: Selective Classification / Abstention Trade-off Curve\nRejecting Highly Uncertain Predictions Enhances High-Precision Guarantee', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(STAGE3E_OUT_DIR, "stage3E_fig05_abstention_tradeoff_curve.png"))
        plt.close()
        print("Saved stage3E_fig05_abstention_tradeoff_curve.png")

if __name__ == '__main__':
    plot_stage3E_figures()

