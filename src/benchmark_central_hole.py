"""
Independent Central-Hole Benchmark Validation Module.

Validates the FEM elasticity solver against classical elasticity theory:
1. Kirsch (1898) infinite-plate analytical solution (Kt = 3.00, sigma_max = 300 psi)
2. Peterson's finite-width plate correction (d/W = 1/12, Kt_net = 2.764, sigma_max = 301.49 psi)
3. Stress decay profile along the transverse symmetry line (y = 0.5)

The configuration contains ONLY the central hole (defect is removed).
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, PSI_TO_PA, PA_TO_PSI


def run_central_hole_benchmark(output_dir: str = "output") -> Dict[str, Any]:
    """
    Execute systematic central-hole benchmark across multiple mesh resolutions
    and compare against Kirsch & Peterson analytical solutions.
    """
    os.makedirs(output_dir, exist_ok=True)

    mat_cfg = MaterialConfig()
    P_nominal = mat_cfg.applied_traction_psi  # 100 psi
    d_W = 1.0 / 12.0                         # d/W ratio = 0.083333
    R_center = d_W / 2.0                     # radius = 1/24

    # Theoretical benchmarks
    kirsch_sigma_max = 3.0 * P_nominal       # 300 psi
    kirsch_kt = 3.00

    # Peterson finite-width correction
    # K_tn = 3.0 - 3.13*(d/W) + 3.66*(d/W)^2 - 1.53*(d/W)^3
    peterson_ktn = 3.0 - 3.13 * d_W + 3.66 * (d_W**2) - 1.53 * (d_W**3)
    sigma_net = P_nominal / (1.0 - d_W)      # 109.091 psi
    peterson_sigma_max = peterson_ktn * sigma_net # ~301.49 psi
    peterson_ktg = peterson_sigma_max / P_nominal # ~3.015

    tiers = [
        {"name": "Coarse",    "h_center": 0.0080, "h_coarse": 0.040},
        {"name": "Medium",    "h_center": 0.0040, "h_coarse": 0.025},
        {"name": "Fine",      "h_center": 0.0020, "h_coarse": 0.018},
        {"name": "Very Fine", "h_center": 0.0010, "h_coarse": 0.012},
        {"name": "Ultra Fine","h_center": 0.0006, "h_coarse": 0.009},
    ]

    print("================================================================================")
    print("INDEPENDENT BENCHMARK: SQUARE PLATE WITH SINGLE CENTRAL CIRCULAR HOLE")
    print("================================================================================")
    print(f"Plate Dimensions:        1.0 x 1.0 (Plane Stress)")
    print(f"Central Hole:            Center = (0.500, 0.500), Diameter d = 1/12 (~0.08333)")
    print(f"Defect Hole:             NONE (Removed for benchmark)")
    print(f"Applied Tensile Load P:  100.0 psi  ({mat_cfg.applied_traction_pa:.2f} Pa)")
    print(f"Theoretical Kirsch (Infinite Plate):  Kt = {kirsch_kt:.2f} | Sigma_max = {kirsch_sigma_max:.2f} psi")
    print(f"Peterson Finite-Width (d/W=1/12):     Kt_net = {peterson_ktn:.4f} | Sigma_net = {sigma_net:.2f} psi | Sigma_max = {peterson_sigma_max:.2f} psi (Ktg = {peterson_ktg:.4f})")
    print("--------------------------------------------------------------------------------")
    print(f"{'Tier':<10} | {'h_center':<8} | {'Nodes':<6} | {'Elems':<6} | {'Nodal vM':<10} | {'Elem vM':<10} | {'Sigma_1':<10} | {'Kt (Nodal)':<10} | {'Peak (X, Y)':<18}")
    print("--------------------------------------------------------------------------------")

    results_table = []
    finest_result = None
    finest_geom = None

    for tier in tiers:
        geom = PlateGeometryConfig(
            has_defect=False,
            h_center_hole=tier["h_center"],
            h_coarse=tier["h_coarse"]
        )
        res = solve_plane_stress_fem(geom, mat_cfg, verbose=False)

        kt_nodal = res.max_von_mises_psi / P_nominal
        kt_elem = res.max_element_von_mises_psi / P_nominal

        row = {
            "tier": tier["name"],
            "h_center": tier["h_center"],
            "h_coarse": tier["h_coarse"],
            "num_nodes": res.num_nodes,
            "num_elements": res.num_elements,
            "nodal_vm_psi": float(res.max_von_mises_psi),
            "elem_vm_psi": float(res.max_element_von_mises_psi),
            "sigma_1_psi": float(res.max_principal_stress_psi),
            "kt_nodal": float(kt_nodal),
            "kt_elem": float(kt_elem),
            "peak_location": [float(res.max_stress_location[0]), float(res.max_stress_location[1])],
            "max_displacement_um": float(res.max_displacement * 1e6),
            "strain_energy_j": float(res.strain_energy)
        }
        results_table.append(row)
        finest_result = res
        finest_geom = geom

        print(
            f"{row['tier']:<10} | {row['h_center']:<8.4f} | {row['num_nodes']:<6} | {row['num_elements']:<6} | "
            f"{row['nodal_vm_psi']:<10.2f} | {row['elem_vm_psi']:<10.2f} | {row['sigma_1_psi']:<10.2f} | "
            f"{row['kt_nodal']:<10.3f} | ({res.max_stress_location[0]:.4f}, {res.max_stress_location[1]:.4f})"
        )

    print("================================================================================")

    # 1. Plot Mesh
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    x = finest_result.nodes[:, 0]
    y = finest_result.nodes[:, 1]
    tri = mtri.Triangulation(x, y, finest_result.elements)
    ax.triplot(tri, color="#34495e", lw=0.35, alpha=0.7)
    c_center = Circle((0.5, 0.5), R_center, color="#e74c3c", fill=False, lw=1.5, label="Central Hole (d=1/12)")
    ax.add_patch(c_center)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Central-Hole Benchmark Mesh\nNodes: {finest_result.num_nodes:,} | Elements: {finest_result.num_elements:,}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    mesh_path = os.path.join(output_dir, "benchmark_central_hole_mesh.png")
    fig.savefig(mesh_path)
    plt.close(fig)

    # 2. Plot Von Mises Stress Contour
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    vm_psi = finest_result.nodal_von_mises * PA_TO_PSI
    levels = np.linspace(np.min(vm_psi), np.max(vm_psi), 60)
    tcf = ax.tricontourf(tri, vm_psi, levels=levels, cmap="turbo")
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Von Mises Stress $\\sigma_{vM}$ (psi)", fontsize=11, fontweight="bold")
    c_mask = Circle((0.5, 0.5), R_center, color="black", fill=True, fc="white", lw=1.2)
    ax.add_patch(c_mask)

    max_idx = np.argmax(vm_psi)
    max_x, max_y = x[max_idx], y[max_idx]
    ax.plot(max_x, max_y, "r*", markersize=14, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(
        f"$\\sigma_{{vM}}^{{max}} = {vm_psi[max_idx]:.2f}$ psi\n$K_t = {vm_psi[max_idx]/100.0:.3f}$\nat ({max_x:.4f}, {max_y:.4f})",
        xy=(max_x, max_y),
        xytext=(max_x + 0.08, max_y + 0.08),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.85)
    )
    ax.set_aspect("equal")
    ax.set_title(
        f"Central-Hole Benchmark: Von Mises Stress\nTheoretical Peterson $K_t \\approx 3.015$ (301.5 psi) | Simulated: {vm_psi[max_idx]:.2f} psi",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    stress_path = os.path.join(output_dir, "benchmark_central_hole_stress.png")
    fig.savefig(stress_path)
    plt.close(fig)

    # 3. Plot Transverse Stress Decay Profile along y = 0.5 (Kirsch Profile)
    # Find nodes close to the horizontal line y = 0.5 for x >= 0.5 + R_center
    eps_line = 0.003
    transverse_mask = (abs(y - 0.5) < eps_line) & (x >= (0.5 + R_center - 1e-4))
    x_fem = x[transverse_mask]
    sig_yy_fem = (finest_result.nodal_stresses[transverse_mask, 1]) * PA_TO_PSI

    # Sort along x
    sort_idx = np.argsort(x_fem)
    x_fem_sorted = x_fem[sort_idx]
    sig_yy_fem_sorted = sig_yy_fem[sort_idx]

    # Analytical Kirsch profile along theta = 0 (horizontal axis)
    r_analytical = np.linspace(R_center, 0.5, 200)
    x_analytical = 0.5 + r_analytical
    sigma_yy_kirsch = P_nominal * (1.0 + 0.5 * (R_center / r_analytical)**2 + 1.5 * (R_center / r_analytical)**4)
    # Peterson scaled profile
    sigma_yy_peterson = sigma_yy_kirsch * (peterson_sigma_max / kirsch_sigma_max)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(x_analytical, sigma_yy_kirsch, "k--", lw=2, label="Classical Kirsch (Infinite Plate, $K_t=3.00$)")
    ax.plot(x_analytical, sigma_yy_peterson, "g-.", lw=2, label="Peterson Correction (Finite Width $d/W=1/12$, $K_t=3.015$)")
    ax.plot(x_fem_sorted, sig_yy_fem_sorted, "ro-", lw=1.5, markersize=5, label=r"FEM Simulation ($\sigma_{yy}$ along $y=0.5$)")
    ax.axhline(100.0, color="gray", linestyle=":", label="Far-Field Nominal Load $P = 100$ psi")

    ax.set_xlabel("X Coordinate along Transverse Symmetry Line $Y = 0.5$ (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normal Stress $\\sigma_{yy}$ (psi)", fontsize=11, fontweight="bold")
    ax.set_title("Stress Concentration Decay Profile: FEM vs Classical Kirsch Solution", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    plt.tight_layout()
    profile_path = os.path.join(output_dir, "benchmark_central_hole_kirsch_profile.png")
    fig.savefig(profile_path)
    plt.close(fig)

    benchmark_data = {
        "theoretical": {
            "kirsch_kt": kirsch_kt,
            "kirsch_sigma_max_psi": kirsch_sigma_max,
            "peterson_ktn": peterson_ktn,
            "peterson_ktg": peterson_ktg,
            "peterson_sigma_max_psi": peterson_sigma_max,
            "nominal_stress_psi": P_nominal
        },
        "convergence_table": results_table,
        "finest_simulation": results_table[-1],
        "plots": {
            "mesh": mesh_path,
            "stress_contour": stress_path,
            "kirsch_profile": profile_path
        }
    }

    with open(os.path.join(output_dir, "benchmark_central_hole_metadata.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2)

    return benchmark_data


if __name__ == "__main__":
    run_central_hole_benchmark()
