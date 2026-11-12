"""
Scientific Visualization Module for Plate with Defect FEM Simulation.

Generates high-resolution publication-quality plots:
- Generated Gmsh triangular mesh with zoom on central and defect holes
- Von Mises stress contour field (in psi and MPa)
- Displacement magnitude field
- Mesh convergence comparison
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle

from src.fem_solver import FEMResult, PlateGeometryConfig


def plot_simulation_results(
    result: FEMResult,
    config: PlateGeometryConfig,
    output_dir: str = "output",
    save_prefix: str = "simulation"
) -> dict:
    """
    Generate and save complete suite of visualization figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    x = result.nodes[:, 0]
    y = result.nodes[:, 1]
    triangles = result.elements
    triangulation = mtri.Triangulation(x, y, triangles)

    saved_paths = {}

    # 1. Mesh Visualization Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.triplot(triangulation, color="#2c3e50", lw=0.35, alpha=0.7)
    
    # Highlight holes
    circle_center = Circle(
        (config.center_hole_x, config.center_hole_y),
        config.center_hole_radius,
        color="#e74c3c", fill=False, lw=1.5, label="Central Hole (a=1/12)"
    )
    circle_defect = Circle(
        (config.defect_hole_x, config.defect_hole_y),
        config.defect_hole_radius,
        color="#2980b9", fill=False, lw=1.5, label=f"Defect Hole ({config.defect_hole_x:.2f}, {config.defect_hole_y:.2f})"
    )
    ax.add_patch(circle_center)
    ax.add_patch(circle_defect)

    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("X (Normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Gmsh Adaptive Triangular Mesh\nNodes: {result.num_nodes:,} | Elements: {result.num_elements:,} (Local Hole Refinement)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    mesh_path = os.path.join(output_dir, f"{save_prefix}_mesh.png")
    fig.savefig(mesh_path)
    plt.close(fig)
    saved_paths["mesh_plot"] = mesh_path

    # 2. Von Mises Stress Contour Plot (psi)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    # Convert nodal stress to psi for direct comparison with paper
    vm_psi = result.nodal_von_mises / 6894.757293168361
    
    # Levels: 40 contour levels for smooth representation
    levels = np.linspace(np.min(vm_psi), np.max(vm_psi), 60)
    tcf = ax.tricontourf(triangulation, vm_psi, levels=levels, cmap="turbo")
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Von Mises Stress $\\sigma_{vM}$ (psi)", fontsize=11, fontweight="bold")
    
    # Outline holes cleanly
    c_center = Circle((config.center_hole_x, config.center_hole_y), config.center_hole_radius, color="black", fill=True, fc="white", lw=1.2)
    c_defect = Circle((config.defect_hole_x, config.defect_hole_y), config.defect_hole_radius, color="black", fill=True, fc="white", lw=1.2)
    ax.add_patch(c_center)
    ax.add_patch(c_defect)

    # Annotate Maximum Stress Location
    max_idx = np.argmax(vm_psi)
    max_x, max_y = x[max_idx], y[max_idx]
    max_val = vm_psi[max_idx]
    ax.plot(max_x, max_y, "r*", markersize=14, markeredgecolor="white", markeredgewidth=1.2, label=f"Max Stress: {max_val:.2f} psi")
    ax.annotate(
        f"$\\sigma_{{vM}}^{{max}} = {max_val:.1f}$ psi\nat ({max_x:.3f}, {max_y:.3f})",
        xy=(max_x, max_y),
        xytext=(max_x + 0.08, max_y + 0.05 if max_y < 0.8 else max_y - 0.12),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.85)
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("X (Normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Von Mises Stress Distribution (Plane Stress)\nApplied Tensile Load P = 100 psi | Defect at ({config.defect_hole_x:.2f}, {config.defect_hole_y:.2f})",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout()
    stress_path = os.path.join(output_dir, f"{save_prefix}_stress_contour.png")
    fig.savefig(stress_path)
    plt.close(fig)
    saved_paths["stress_plot"] = stress_path

    # 3. Displacement Magnitude Field
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    disp_mag = np.linalg.norm(result.displacements, axis=1) * 1e6  # in micrometers (um)
    levels_u = np.linspace(np.min(disp_mag), np.max(disp_mag), 50)
    tcf_u = ax.tricontourf(triangulation, disp_mag, levels=levels_u, cmap="viridis")
    cbar_u = fig.colorbar(tcf_u, ax=ax, fraction=0.046, pad=0.04)
    cbar_u.set_label("Displacement Magnitude $|\\mathbf{u}|$ ($\\mu$m)", fontsize=11, fontweight="bold")

    c_center2 = Circle((config.center_hole_x, config.center_hole_y), config.center_hole_radius, color="black", fill=True, fc="white", lw=1.2)
    c_defect2 = Circle((config.defect_hole_x, config.defect_hole_y), config.defect_hole_radius, color="black", fill=True, fc="white", lw=1.2)
    ax.add_patch(c_center2)
    ax.add_patch(c_defect2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("X (Normalized)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (Normalized)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Resultant Displacement Field\nMax Disp: {result.max_displacement * 1e6:.3f} $\\mu$m (Linear Elasticity)",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    disp_path = os.path.join(output_dir, f"{save_prefix}_displacement.png")
    fig.savefig(disp_path)
    plt.close(fig)
    saved_paths["displacement_plot"] = disp_path

    return saved_paths


def plot_mesh_convergence(
    mesh_sizes: list,
    nodal_stresses: list,
    num_elements: list,
    elem_stresses: Optional[list] = None,
    energies: Optional[list] = None,
    output_path: str = "output/mesh_convergence.png"
):
    """
    Plot stress and energy convergence versus mesh resolution and element count.
    Compares recovered nodal maximum against element-centroid maximum.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Stress Convergence vs Element Size h
    ax1.plot(mesh_sizes, nodal_stresses, "o-b", lw=2, markersize=7, label="Recovered Nodal $\\sigma_{vM}^{\\max}$")
    if elem_stresses is not None:
        ax1.plot(mesh_sizes, elem_stresses, "s--r", lw=2, markersize=7, label="Element Centroid $\\sigma_{vM}^{\\max}$")
    ax1.axhline(300.0, color="gray", linestyle=":", label="Theoretical Kirsch Baseline (300 psi)")
    ax1.set_xlabel("Hole Refinement Size $h_{defect}$ (Normalized)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Max Von Mises Stress $\\sigma_{vM}^{\\max}$ (psi)", fontsize=11, fontweight="bold")
    ax1.set_title("Stress Convergence vs Element Characteristic Size", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", frameon=True)
    ax1.invert_xaxis()  # Smaller h on right

    # Plot 2: Stress Convergence vs Number of Elements (with secondary axis for strain energy if available)
    ax2.plot(num_elements, nodal_stresses, "o-b", lw=2, markersize=7, label="Nodal $\\sigma_{vM}^{\\max}$ (psi)")
    if elem_stresses is not None:
        ax2.plot(num_elements, elem_stresses, "s--r", lw=2, markersize=7, label="Element $\\sigma_{vM}^{\\max}$ (psi)")
    ax2.set_xlabel("Number of Triangular Elements", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Max Stress $\\sigma_{vM}^{\\max}$ (psi)", fontsize=11, fontweight="bold", color="black")
    ax2.tick_params(axis="y")
    ax2.grid(True, linestyle=":", alpha=0.6)

    if energies is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(num_elements, energies, "^:g", lw=1.8, markersize=6, label="Strain Energy (J)")
        ax2_twin.set_ylabel("Total Strain Energy (J)", fontsize=11, fontweight="bold", color="green")
        ax2_twin.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax2.get_legend_handles_labels()
    if energies is not None:
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True)
    else:
        ax2.legend(loc="lower right", frameon=True)

    ax2.set_title("Stress & Energy Convergence vs Element Count", fontsize=12, fontweight="bold")
    plt.suptitle("Extended FEM Mesh Convergence Study (Plate with Defect)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

