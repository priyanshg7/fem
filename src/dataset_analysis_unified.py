"""
Publication-Quality Comparative Analysis for Unified Multi-Geometry Dataset (Square vs Rectangular).

Generates:
1. Comparative Spatial Defect Coverage (Normalized X/L vs Y/H for Square and Rectangular plates).
2. Comparative Defect Diameter Distribution (D / H).
3. Comparative Stress Concentration Distribution (Normalized Max Stress Kt = sigma_vM / P).
4. Direct comparison of Fixed-Diameter Comparison subsets (D/H = 1/80).
5. Joint parameter vs peak stress response surfaces.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "figure.autolayout": True,
    "savefig.dpi": 300
})


def plot_comparative_figures(unified_csv_path: str = "data/unified_dataset_20150.csv", output_dir: str = "output"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading unified dataset from {unified_csv_path}...")
    df = pd.read_csv(unified_csv_path)
    df_sq = df[df["geometry_family"] == "square_1000x1000"]
    df_rect = df[df["geometry_family"] == "rectangular_1000x600"]
    
    print(f"Square records: {len(df_sq)}, Rectangular records: {len(df_rect)}")

    # 1. Comparative Normalized Spatial Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Square plate (Normalized)
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=2)
    ax1.add_patch(Circle((0.5, 0.5), 1.0/24.0, color="gray", alpha=0.7))
    sc1 = ax1.scatter(
        df_sq["normalized_defect_x"], df_sq["normalized_defect_y"],
        c=df_sq["normalized_defect_diameter"] * 1000.0, cmap="viridis",
        s=4, alpha=0.35, rasterized=True
    )
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="D / H (x10^-3)")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Normalized Defect X (X / L)")
    ax1.set_ylabel("Normalized Defect Y (Y / H)")
    ax1.set_title("Square Plate (1000 mm x 1000 mm, L/H = 1.0)\nN = 10,075")
    ax1.grid(True, ls=":", alpha=0.5)

    # Rectangular plate (Normalized)
    ax2.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=2)
    ax2.add_patch(Circle((0.5, 0.5), 1.0/24.0, color="gray", alpha=0.7))
    sc2 = ax2.scatter(
        df_rect["normalized_defect_x"], df_rect["normalized_defect_y"],
        c=df_rect["normalized_defect_diameter"] * 1000.0, cmap="viridis",
        s=4, alpha=0.35, rasterized=True
    )
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="D / H (x10^-3)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Normalized Defect X (X / L)")
    ax2.set_ylabel("Normalized Defect Y (Y / H)")
    ax2.set_title("Rectangular Plate (1000 mm x 600 mm, L/H = 1.67)\nN = 10,075")
    ax2.grid(True, ls=":", alpha=0.5)

    fig.savefig(out_dir / "stage2R_comparative_spatial_distribution.png")
    plt.close(fig)

    # 2. Comparative Normalized Defect Diameter (D / H)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.005, 0.026, 50)
    ax.hist(df_sq["normalized_defect_diameter"], bins=bins, color="#2980b9", alpha=0.6, label="Square Plate (L/H = 1.0)", edgecolor="black")
    ax.hist(df_rect["normalized_defect_diameter"], bins=bins, color="#e67e22", alpha=0.6, label="Rectangular Plate (L/H = 1.67)", edgecolor="black")
    ax.axvline(1.0/80.0, color="red", ls="--", lw=2, label="Comparison Baseline D/H = 1/80 (0.0125)")
    ax.set_xlabel("Normalized Defect Diameter (D / H)")
    ax.set_ylabel("Count")
    ax.set_title("Normalized Defect Diameter Distribution Comparison")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2R_comparative_diameter_distribution.png")
    plt.close(fig)

    # 3. Comparative Normalized Peak Stress Kt = max_stress / P
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(2.9, 5.0, 70)
    ax.hist(df_sq["normalized_max_stress_element"], bins=bins, color="#3498db", alpha=0.6, label=f"Square (Mean Kt: {df_sq['normalized_max_stress_element'].mean():.3f})", edgecolor="black")
    ax.hist(df_rect["normalized_max_stress_element"], bins=bins, color="#e74c3c", alpha=0.6, label=f"Rectangular (Mean Kt: {df_rect['normalized_max_stress_element'].mean():.3f})", edgecolor="black")
    ax.set_xlabel(r"Normalized Peak Stress Concentration Factor $K_t = \sigma_{vM}^{\max} / P$")
    ax.set_ylabel("Simulation Frequency")
    ax.set_title("Stress Concentration Distribution: Square vs Rectangular Plates")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2R_comparative_stress_distribution.png")
    plt.close(fig)

    # 4. Comparative Compliance (Normalized Displacement & Energy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.boxplot(
        [df_sq["normalized_displacement_m_per_pa"] * 1e11, df_rect["normalized_displacement_m_per_pa"] * 1e11],
        tick_labels=["Square (1000x1000)", "Rectangular (1000x600)"],
        patch_artist=True,
        boxprops=dict(facecolor="#9b59b6", alpha=0.7)
    )
    ax1.set_ylabel(r"Compliance $u_{\max} / P$ ($\times 10^{-11}\text{ m/Pa}$)")
    ax1.set_title("Maximum Resultant Displacement Compliance")
    ax1.grid(True, ls=":", alpha=0.5)

    ax2.boxplot(
        [df_sq["normalized_strain_energy_j_per_pa2"] * 1e12, df_rect["normalized_strain_energy_j_per_pa2"] * 1e12],
        tick_labels=["Square (1000x1000)", "Rectangular (1000x600)"],
        patch_artist=True,
        boxprops=dict(facecolor="#2ecc71", alpha=0.7)
    )
    ax2.set_ylabel(r"Strain Energy Compliance $U / P^2$ ($\times 10^{-12}\text{ J/Pa}^2$)")
    ax2.set_title("Total Strain Energy Compliance")
    ax2.grid(True, ls=":", alpha=0.5)

    fig.savefig(out_dir / "stage2R_comparative_compliance.png")
    plt.close(fig)

    print("All comparative plots saved to output/")


if __name__ == "__main__":
    plot_comparative_figures()

