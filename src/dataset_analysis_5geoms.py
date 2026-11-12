"""
Comprehensive Publication-Quality Cross-Geometry Analysis across 5 Plate Geometries (Stage 2S).

Geometries analyzed:
1. square_1000x1000 (AR = 1.000)
2. rectangular_1000x900 (AR = 1.111)
3. rectangular_1000x800 (AR = 1.250)
4. rectangular_1000x700 (AR = 1.429)
5. rectangular_1000x600 (AR = 1.667)
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

GEOM_ORDER = [
    "square_1000x1000",
    "rectangular_1000x900",
    "rectangular_1000x800",
    "rectangular_1000x700",
    "rectangular_1000x600"
]

GEOM_LABELS = {
    "square_1000x1000": "Square (1000x1000, AR=1.00)",
    "rectangular_1000x900": "RECT_900 (1000x900, AR=1.11)",
    "rectangular_1000x800": "RECT_800 (1000x800, AR=1.25)",
    "rectangular_1000x700": "RECT_700 (1000x700, AR=1.43)",
    "rectangular_1000x600": "RECT_600 (1000x600, AR=1.67)",
}

GEOM_COLORS = {
    "square_1000x1000": "#2c3e50",
    "rectangular_1000x900": "#2980b9",
    "rectangular_1000x800": "#27ae60",
    "rectangular_1000x700": "#e67e22",
    "rectangular_1000x600": "#c0392b",
}


def plot_5geoms_figures(unified_csv: str = "data/unified_dataset_50375.csv", output_dir: str = "output"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 5-geometry dataset from {unified_csv}...")
    df = pd.read_csv(unified_csv)
    print(f"Loaded {len(df)} total rows.")

    # 1. Normalized Spatial Coverage (5 Subplots)
    fig, axs = plt.subplots(1, 5, figsize=(22, 5))
    for ax, geom_id in zip(axs, GEOM_ORDER):
        sub = df[df["geometry_family"] == geom_id]
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.5)
        ax.add_patch(Circle((0.5, 0.5), 1.0/24.0, color="gray", alpha=0.7))
        sc = ax.scatter(
            sub["normalized_defect_x"], sub["normalized_defect_y"],
            c=sub["normalized_defect_diameter"] * 1000.0, cmap="viridis",
            s=2, alpha=0.3, rasterized=True
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("X / L")
        ax.set_title(f"{GEOM_LABELS[geom_id]}\n(N = {len(sub):,})")
        ax.grid(True, ls=":", alpha=0.4)
    axs[0].set_ylabel("Y / H")
    fig.savefig(out_dir / "stage2S_5geoms_spatial_coverage.png")
    plt.close(fig)

    # 2. Normalized Defect Diameter Distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.005, 0.026, 60)
    for geom_id in GEOM_ORDER:
        sub = df[df["geometry_family"] == geom_id]
        ax.hist(sub["normalized_defect_diameter"], bins=bins, histtype="step", lw=2, color=GEOM_COLORS[geom_id], label=GEOM_LABELS[geom_id])
    ax.axvline(1.0/80.0, color="black", ls="--", lw=1.5, label="Baseline D/H = 1/80")
    ax.set_xlabel("Normalized Defect Diameter (D / H)")
    ax.set_ylabel("Count")
    ax.set_title("Normalized Defect Diameter Distribution across 5 Plate Geometries")
    ax.legend(framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2S_5geoms_diameter_distribution.png")
    plt.close(fig)

    # 3. Stress Concentration Factor Distribution Kt = sigma_vM / P
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(2.85, 4.9, 70)
    for geom_id in GEOM_ORDER:
        sub = df[df["geometry_family"] == geom_id]
        mean_kt = sub["normalized_max_stress_element"].mean()
        ax.hist(
            sub["normalized_max_stress_element"], bins=bins, histtype="step", lw=2,
            color=GEOM_COLORS[geom_id], label=f"{GEOM_LABELS[geom_id]} (Mean Kt: {mean_kt:.3f})"
        )
    ax.set_xlabel(r"Stress Concentration Factor $K_t = \sigma_{vM}^{\max} / P$")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Stress Concentration Distribution across 5 Aspect Ratios ($N = 50,375$)")
    ax.legend(framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(out_dir / "stage2S_5geoms_stress_distribution.png")
    plt.close(fig)

    # 4. Physical Trends vs Aspect Ratio (L / H)
    ar_stats = []
    for geom_id in GEOM_ORDER:
        sub = df[df["geometry_family"] == geom_id]
        ar = sub["aspect_ratio"].iloc[0]
        ar_stats.append({
            "geom_id": geom_id,
            "aspect_ratio": ar,
            "mean_kt": sub["normalized_max_stress_element"].mean(),
            "max_kt": sub["normalized_max_stress_element"].max(),
            "mean_disp_um": sub["max_displacement_m"].mean() * 1e6,
            "mean_energy_j": sub["strain_energy_j"].mean(),
            "defect_dominated_pct": (sub["peak_stress_region"] == "defect_hole").mean() * 100.0
        })
    df_ar = pd.DataFrame(ar_stats)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # Mean Kt vs AR
    axs[0, 0].plot(df_ar["aspect_ratio"], df_ar["mean_kt"], "o-", color="#2980b9", lw=2, ms=8)
    for _, r in df_ar.iterrows():
        axs[0, 0].text(r["aspect_ratio"], r["mean_kt"] + 0.003, f"{r['mean_kt']:.3f}", ha="center", fontweight="bold")
    axs[0, 0].set_xlabel("Aspect Ratio (L / H)")
    axs[0, 0].set_ylabel(r"Mean $K_t = \sigma_{vM}^{\max} / P$")
    axs[0, 0].set_title("Mean Stress Concentration vs Aspect Ratio")
    axs[0, 0].grid(True, ls=":", alpha=0.5)

    # Max Displacement vs AR
    axs[0, 1].plot(df_ar["aspect_ratio"], df_ar["mean_disp_um"], "s-", color="#8e44ad", lw=2, ms=8)
    for _, r in df_ar.iterrows():
        axs[0, 1].text(r["aspect_ratio"], r["mean_disp_um"] + 0.15, f"{r['mean_disp_um']:.2f} um", ha="center", fontweight="bold")
    axs[0, 1].set_xlabel("Aspect Ratio (L / H)")
    axs[0, 1].set_ylabel(r"Mean Max Displacement ($\mu\text{m}$)")
    axs[0, 1].set_title(r"Displacement vs Aspect Ratio ($u \propto H$)")
    axs[0, 1].grid(True, ls=":", alpha=0.5)

    # Strain Energy vs AR
    axs[1, 0].plot(df_ar["aspect_ratio"], df_ar["mean_energy_j"], "d-", color="#27ae60", lw=2, ms=8)
    for _, r in df_ar.iterrows():
        axs[1, 0].text(r["aspect_ratio"], r["mean_energy_j"] + 0.05, f"{r['mean_energy_j']:.3f} J", ha="center", fontweight="bold")
    axs[1, 0].set_xlabel("Aspect Ratio (L / H)")
    axs[1, 0].set_ylabel("Mean Total Strain Energy (J)")
    axs[1, 0].set_title(r"Strain Energy vs Aspect Ratio ($U \propto \text{Volume}$)")
    axs[1, 0].grid(True, ls=":", alpha=0.5)

    # Defect Dominated Percentage vs AR
    axs[1, 1].bar(df_ar["aspect_ratio"].astype(str), df_ar["defect_dominated_pct"], color="#e74c3c", edgecolor="black", alpha=0.8, width=0.4)
    for i, r in df_ar.iterrows():
        axs[1, 1].text(i, r["defect_dominated_pct"] + 0.2, f"{r['defect_dominated_pct']:.2f}%", ha="center", fontweight="bold")
    axs[1, 1].set_xlabel("Aspect Ratio (L / H)")
    axs[1, 1].set_ylabel("Defect-Dominated Simulations (%)")
    axs[1, 1].set_title("Frequency of Defect Controlling Peak Stress")
    axs[1, 1].set_ylim(0, 10.0)
    axs[1, 1].grid(True, ls=":", alpha=0.5, axis="y")

    fig.savefig(out_dir / "stage2S_5geoms_aspect_ratio_trends.png")
    plt.close(fig)

    print("All 5-geometry comparative figures successfully generated in output/")


if __name__ == "__main__":
    plot_5geoms_figures()

