"""
Detailed Visualizations for Rectangular Plate Dataset (Stage 2R).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
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


def generate_rectangular_visualizations():
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/rectangular_1000x600/fem_dataset_10075_rectangular.csv")

    # 1. Defect Spatial Distribution
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot([0, 1000, 1000, 0, 0], [0, 0, 600, 600, 0], "k-", lw=2, label="Plate Boundary (1000 x 600 mm)")
    circle_center = Circle((500, 300), 25.0, color="gray", alpha=0.7, label="Central Hole (d = 50 mm)")
    ax.add_patch(circle_center)
    circle_excl = Circle((500, 300), 25.0 + 6.0, color="red", fill=False, ls="--", lw=1.2, label="Exclusion Zone (+6 mm margin)")
    ax.add_patch(circle_excl)

    sc = ax.scatter(
        df["defect_center_x_mm"], df["defect_center_y_mm"],
        c=df["defect_diameter_mm"], cmap="viridis",
        s=4, alpha=0.4, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Defect Diameter D (mm)")

    ax.set_xlim(-20, 1020)
    ax.set_ylim(-20, 620)
    ax.set_aspect("equal")
    ax.set_xlabel("Defect Center X (mm)")
    ax.set_ylabel("Defect Center Y (mm)")
    ax.set_title(f"Rectangular Plate Defect Locations (N = {len(df):,})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2R_spatial_distribution.png")
    plt.close(fig)

    # 2. Stress vs Input Parameters (X, Y, D)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].scatter(df["defect_center_x_mm"], df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#2980b9", rasterized=True)
    axs[0].set_xlabel("Defect X (mm)")
    axs[0].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[0].set_title("Defect X vs Peak Stress")
    axs[0].grid(True, ls=":", alpha=0.5)

    axs[1].scatter(df["defect_center_y_mm"], df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#8e44ad", rasterized=True)
    axs[1].set_xlabel("Defect Y (mm)")
    axs[1].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[1].set_title("Defect Y vs Peak Stress")
    axs[1].grid(True, ls=":", alpha=0.5)

    axs[2].scatter(df["defect_diameter_mm"], df["max_von_mises_element_psi"], s=3, alpha=0.3, color="#d35400", rasterized=True)
    axs[2].set_xlabel("Defect Diameter D (mm)")
    axs[2].set_ylabel(r"Max Element $\sigma_{vM}$ (psi)")
    axs[2].set_title("Defect Diameter D vs Peak Stress")
    axs[2].grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2R_stress_correlations.png")
    plt.close(fig)

    # 3. Fixed-Diameter (D = 7.50 mm) Spatial Stress Heatmap
    df_fixed = df[df["subset_type"] == "fixed_d_7_5mm"]
    if len(df_fixed) > 0:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        sc = ax.scatter(
            df_fixed["defect_center_x_mm"], df_fixed["defect_center_y_mm"],
            c=df_fixed["max_von_mises_element_psi"], cmap="plasma",
            s=12, alpha=0.85, edgecolors="none"
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label(r"Max Element $\sigma_{vM}$ (psi)")

        ax.plot([0, 1000, 1000, 0, 0], [0, 0, 600, 600, 0], "k-", lw=1.5)
        ax.add_patch(Circle((500, 300), 25.0, color="gray", alpha=0.8))
        ax.set_xlim(-20, 1020)
        ax.set_ylim(-20, 620)
        ax.set_aspect("equal")
        ax.set_xlabel("Defect Center X (mm)")
        ax.set_ylabel("Defect Center Y (mm)")
        ax.set_title(f"Rectangular Plate Response Map for Fixed D = 7.50 mm (N = {len(df_fixed):,})")
        ax.grid(True, ls=":", alpha=0.5)
        fig.savefig(out_dir / "stage2R_fixed_diameter_response_surface.png")
        plt.close(fig)

    # 4. Representative Full-field Contours
    fields_dir = Path("data/rectangular_1000x600/fields")
    d_small_idx = int(df.loc[df["defect_diameter_mm"] < 4.0].index[0])
    d_med_idx = int(df.loc[abs(df["defect_diameter_mm"] - 7.50) < 1e-4].index[0])
    d_large_idx = int(df.loc[df["defect_diameter_mm"] > 14.5].index[0])

    rep_cases = [
        {"sim_id": d_small_idx, "title": "Small Defect (D = 3.75 mm)"},
        {"sim_id": d_med_idx, "title": "Original Ratio Defect (D = 7.50 mm)"},
        {"sim_id": d_large_idx, "title": "Large Defect (D = 15.00 mm)"},
    ]

    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    for ax, case in zip(axs, rep_cases):
        sim_id = case["sim_id"]
        title = case["title"]
        file_path = fields_dir / f"sim_{sim_id:05d}.npz"
        if not file_path.exists():
            continue
        data = np.load(file_path)
        nodes = data["nodal_coords_mm"]
        stress = data["nodal_von_mises_psi"]
        x_d, y_d = data["defect_coords_mm"]
        d = float(data["defect_diameter_mm"])

        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
        centroids = np.mean(nodes[triang.triangles], axis=1)
        dist_c = np.hypot(centroids[:, 0] - 500.0, centroids[:, 1] - 300.0)
        dist_d = np.hypot(centroids[:, 0] - x_d, centroids[:, 1] - y_d)
        mask = (dist_c < 25.0 * 0.98) | (dist_d < (d/2.0) * 0.98)
        triang.set_mask(mask)

        tcf = ax.tricontourf(triang, stress, levels=40, cmap="jet")
        plt.colorbar(tcf, ax=ax, fraction=0.025, pad=0.03, label=r"Von Mises $\sigma_{vM}$ (psi)")
        ax.plot([0, 1000, 1000, 0, 0], [0, 0, 600, 600, 0], "k-", lw=1.5)
        ax.plot(500.0, 300.0, "w+", ms=8, mew=2)
        ax.plot(x_d, y_d, "m*", ms=10, label=f"Defect ({x_d:.1f}, {y_d:.1f})")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 1020)
        ax.set_ylim(-20, 620)
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{title} | Peak $\sigma_{{vM}}$ = {np.max(stress):.1f} psi")
        ax.legend(loc="lower right")

    axs[-1].set_xlabel("X (mm)")
    fig.savefig(out_dir / "stage2R_representative_stress_contours.png")
    plt.close(fig)
    print("Rectangular figures successfully generated in output/")


if __name__ == "__main__":
    generate_rectangular_visualizations()

