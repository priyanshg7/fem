"""
Publication-Quality Analysis and Visualization for 3-Parameter FEM Dataset.

Generates all required figures for Stage 2:
1. Spatial defect-position distribution in the plate (X vs Y).
2. Defect-diameter distribution (D histogram & KDE).
3. Joint (X, Y, D) scatter with stress color-mapping.
4. Maximum Von Mises stress distribution (element-centroid vs nodal).
5. Marginal relationships:
   - X vs Max Stress
   - Y vs Max Stress
   - D vs Max Stress
6. Fixed-diameter comparison subset (D = 1/80) vs Variable-D subset.
7. Representative FEM stress contours for Small (D=1/160), Original (D=1/80), and Large (D=1/40).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

# Style configuration
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


def plot_dataset_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Generate parameter distributions and correlation plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Defect Spatial Distribution (X vs Y)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Plate boundary
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=2, label="Plate Boundary (1.0 x 1.0)")
    # Central hole
    circle_center = Circle((0.5, 0.5), 1.0/24.0, color="gray", alpha=0.6, label="Central Hole (d=1/12)")
    ax.add_patch(circle_center)
    # Exclusion boundary
    circle_excl = Circle((0.5, 0.5), 1.0/24.0 + 0.01, color="red", fill=False, ls="--", lw=1.2, label="Exclusion Zone (+0.01 margin)")
    ax.add_patch(circle_excl)
    
    # Scatter of defect locations
    sc = ax.scatter(
        df["defect_x"], df["defect_y"],
        c=df["defect_diameter"] * 1000.0, cmap="viridis",
        s=4, alpha=0.4, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Defect Diameter D (mm)")
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Defect Center X (m)")
    ax.set_ylabel("Defect Center Y (m)")
    ax.set_title(f"Sampled Defect Locations in Plate (N = {len(df):,})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(output_dir / "stage2_spatial_distribution.png")
    plt.close(fig)

    # 2. Defect Diameter Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(df["defect_diameter"] * 1000.0, bins=50, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax1.axvline(1.0/80.0 * 1000.0, color="red", ls="--", lw=2, label=f"Original Paper D=1/80 ({1.0/80.0*1000:.2f} mm)")
    ax1.set_xlabel("Defect Diameter D (mm)")
    ax1.set_ylabel("Simulation Count")
    ax1.set_title("Defect Diameter Distribution (Total)")
    ax1.legend()
    ax1.grid(True, ls=":", alpha=0.5)

    # Subset breakdown
    sub_counts = df["subset_type"].value_counts()
    colors = ["#2ca02c", "#ff7f0e"]
    ax2.bar(sub_counts.index, sub_counts.values, color=colors, edgecolor="black", alpha=0.85)
    for i, v in enumerate(sub_counts.values):
        ax2.text(i, v + 100, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.set_title("Dataset Composition: Comparison Subset vs Variable-3D")
    ax2.set_ylim(0, max(sub_counts.values) * 1.15)
    ax2.grid(True, ls=":", alpha=0.5, axis="y")
    fig.savefig(output_dir / "stage2_diameter_distribution.png")
    plt.close(fig)

    # 3. Maximum Von Mises Stress Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["max_von_mises_element_psi"], bins=60, color="#3470a3", edgecolor="black", alpha=0.7, label=f"Element Centroid (Mean: {df['max_von_mises_element_psi'].mean():.2f} psi)")
    ax.hist(df["max_von_mises_nodal_psi"], bins=60, color="#e67e22", edgecolor="black", alpha=0.6, label=f"Recovered Nodal (Mean: {df['max_von_mises_nodal_psi'].mean():.2f} psi)")
    ax.axvline(301.49, color="black", ls=":", lw=1.8, label="Peterson Central Hole Baseline (301.49 psi)")
    ax.set_xlabel("Maximum Von Mises Stress (psi)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Maximum Von Mises Stress Distribution (N = {len(df):,})")
    ax.legend(framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(output_dir / "stage2_stress_distribution.png")
    plt.close(fig)

    # 4. Stress vs Input Parameters (X, Y, D)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    # X vs Stress
    axs[0].scatter(df["defect_x"], df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#2980b9", rasterized=True)
    axs[0].set_xlabel("Defect X-coordinate (m)")
    axs[0].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[0].set_title(r"Defect X vs Peak Stress")
    axs[0].grid(True, ls=":", alpha=0.5)

    # Y vs Stress
    axs[1].scatter(df["defect_y"], df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#8e44ad", rasterized=True)
    axs[1].set_xlabel("Defect Y-coordinate (m)")
    axs[1].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[1].set_title(r"Defect Y vs Peak Stress")
    axs[1].grid(True, ls=":", alpha=0.5)

    # D vs Stress
    axs[2].scatter(df["defect_diameter"] * 1000.0, df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#d35400", rasterized=True)
    axs[2].set_xlabel("Defect Diameter D (mm)")
    axs[2].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[2].set_title(r"Defect Diameter D vs Peak Stress")
    axs[2].grid(True, ls=":", alpha=0.5)

    fig.savefig(output_dir / "stage2_stress_correlations.png")
    plt.close(fig)

    # 5. Fixed-Diameter (D=1/80) Spatial Stress Heatmap / Interpolated Response Surface
    df_fixed = df[df["subset_type"] == "fixed_d_1_80"]
    if len(df_fixed) > 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            df_fixed["defect_x"], df_fixed["defect_y"],
            c=df_fixed["max_von_mises_element_psi"], cmap="plasma",
            s=12, alpha=0.85, edgecolors="none"
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Max Element $\sigma_{vM}$ (psi)")

        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.5)
        circle_center = Circle((0.5, 0.5), 1.0/24.0, color="gray", alpha=0.8)
        ax.add_patch(circle_center)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Defect Center X (m)")
        ax.set_ylabel("Defect Center Y (m)")
        ax.set_title(f"Peak Stress Response Map for Fixed D = 1/80 m (N = {len(df_fixed):,})")
        ax.grid(True, ls=":", alpha=0.5)
        fig.savefig(output_dir / "stage2_fixed_diameter_response_surface.png")
        plt.close(fig)


def plot_representative_contours(fields_dir: Path, output_dir: Path):
    """
    Generate representative full-field Von Mises stress contour plots for:
    1. Small defect (D = 1/160 m)
    2. Original defect (D = 1/80 m)
    3. Large defect (D = 1/40 m)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df_params = pd.read_csv("data/sampled_parameters_3param.csv")
    
    # Find representative simulation IDs for small, medium, large
    d_small_idx = int(df_params.loc[df_params["defect_diameter"] < 0.007].index[0])
    d_med_idx = int(df_params.loc[abs(df_params["defect_diameter"] - 1.0/80.0) < 1e-5].index[0])
    d_large_idx = int(df_params.loc[df_params["defect_diameter"] > 0.024].index[0])

    rep_cases = [
        {"sim_id": d_small_idx, "title": "Small Defect (D = 1/160 m ≈ 6.25 mm)"},
        {"sim_id": d_med_idx, "title": "Original Defect (D = 1/80 m = 12.50 mm)"},
        {"sim_id": d_large_idx, "title": "Large Defect (D = 1/40 m = 25.00 mm)"},
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, case in zip(axs, rep_cases):
        sim_id = case["sim_id"]
        title = case["title"]
        file_path = fields_dir / f"sim_{sim_id:05d}.npz"
        if not file_path.exists():
            continue
        
        data = np.load(file_path)
        nodes = data["nodal_coords"]
        stress = data["nodal_von_mises_psi"]
        x_d, y_d = data["defect_coords"]
        d = float(data["defect_diameter"])
        
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
        # Mask triangles that cross circular holes
        centroids = np.mean(nodes[triang.triangles], axis=1)
        dist_c = np.hypot(centroids[:, 0] - 0.5, centroids[:, 1] - 0.5)
        dist_d = np.hypot(centroids[:, 0] - x_d, centroids[:, 1] - y_d)
        mask = (dist_c < 1.0/24.0 * 0.98) | (dist_d < (d/2.0) * 0.98)
        triang.set_mask(mask)

        tcf = ax.tricontourf(triang, stress, levels=40, cmap="jet")
        plt.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04, label=r"Von Mises Stress $\sigma_{vM}$ (psi)")

        # Plate boundary
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.5)
        # Mark hole centers
        ax.plot(0.5, 0.5, "w+", ms=8, mew=2)
        ax.plot(x_d, y_d, "m*", ms=10, label=f"Defect ({x_d:.3f}, {y_d:.3f})")
        
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{title}\nPeak $\sigma_{{vM}}$ = {np.max(stress):.1f} psi")
        ax.legend(loc="lower right")

    fig.savefig(output_dir / "stage2_representative_stress_contours.png")
    plt.close(fig)


if __name__ == "__main__":
    summary_path = Path("data/fem_dataset_10075.csv")
    out_dir = Path("output")
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        print(f"Generating publication plots from {len(df_sum)} records...")
        plot_dataset_distributions(df_sum, out_dir)
        plot_representative_contours(Path("data/fields"), out_dir)
        print("All Stage 2 plots successfully generated in output/")
    else:
        print("fem_dataset_10075.csv not found yet.")

