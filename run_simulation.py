"""
Main Executable Script for FEM Simulation of Plate with Defect.

Reproduces the physical experiment from Singh & Pamnani (2025):
- Plate dimensions: 1.0 x 1.0 (square isotropic plate in plane stress)
- Central hole: fixed at (0.5, 0.5), diameter a = 1/12
- Defect hole: variable (X, Y), diameter c = 1/80
- Material: E = 10^7 psi, nu = 0.3
- Loading: Uniaxial tension P = 100 psi (689,475.7 Pa) along top edge
- Boundary conditions: Roller at bottom (u_y = 0), midpoint pinned in x (u_x = 0)
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, PSI_TO_PA, PA_TO_PSI
from src.visualization import plot_simulation_results
from src.convergence import run_convergence_study


def run_single_simulation(
    defect_x: float = 0.35,
    defect_y: float = 0.65,
    h_defect: float = 0.0010,
    h_center: float = 0.0030,
    h_coarse: float = 0.020,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Execute single validated FEM simulation and output all artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("================================================================================")
    print("PLATE WITH DEFECT: FINITE ELEMENT METHOD (FEM) SIMULATION")
    print("Scientific Reference: Singh & Pamnani (2025), MNIT Jaipur / ISME 2025")
    print("================================================================================")
    print(f"Plate Dimensions:        1.0 x 1.0 [normalized units]")
    print(f"Central Hole:            Center = (0.500, 0.500), Diameter a = 1/12 (~0.0833)")
    print(f"Defect Hole:             Center = ({defect_x:.4f}, {defect_y:.4f}), Diameter c = 1/80 (0.0125)")
    print(f"Young's Modulus E:       1.00e+07 psi  ({1e7 * PSI_TO_PA / 1e9:.3f} GPa)")
    print(f"Poisson's Ratio nu:      0.30")
    print(f"Applied Tensile Load P:  100.0 psi  ({100.0 * PSI_TO_PA:.2f} Pa)")
    print(f"Constitutive Model:      Isotropic Linear Elasticity (Plane Stress Formulation)")
    print("--------------------------------------------------------------------------------")

    geom_cfg = PlateGeometryConfig(
        defect_hole_x=defect_x,
        defect_hole_y=defect_y,
        h_coarse=h_coarse,
        h_center_hole=h_center,
        h_defect_hole=h_defect
    )
    mat_cfg = MaterialConfig()

    print("Generating geometry & adaptive mesh with Gmsh...")
    result = solve_plane_stress_fem(geom_cfg, mat_cfg, verbose=False)

    print("FEM Solution completed successfully.")
    print("--------------------------------------------------------------------------------")
    print(f"Mesh Statistics:         Nodes: {result.num_nodes:,} | Elements: {result.num_elements:,}")
    print(f"Max Nodal Von Mises:     {result.max_von_mises_psi:.2f} psi  ({result.max_von_mises_pa / 1e6:.4f} MPa) at ({result.max_stress_location[0]:.4f}, {result.max_stress_location[1]:.4f})")
    print(f"Max Element Von Mises:   {result.max_element_von_mises_psi:.2f} psi  ({result.max_element_von_mises_pa / 1e6:.4f} MPa) at ({result.max_element_location[0]:.4f}, {result.max_element_location[1]:.4f})")
    print(f"Max Principal Stress s1: {result.max_principal_stress_psi:.2f} psi  ({result.max_principal_stress_pa / 1e6:.4f} MPa)")
    print(f"Max Displacement:        {result.max_displacement * 1e6:.4f} um ({result.max_displacement:.6e} m)")
    print(f"Total Strain Energy:     {result.strain_energy:.6f} J")
    print(f"Applied Tensile Force:   {result.applied_force_total_n:.2f} N")
    print(f"Reaction Force (Bottom): {result.reaction_force_bottom_n:.2f} N  (Delta = {result.equilibrium_error_n:.2e} N)")

    # Analytical Reference (Kirsch solution for infinite plate, Kt = 3.0)
    kirsch_stress_psi = 3.0 * mat_cfg.applied_traction_psi
    # Peterson's finite-width plate with central hole (d/W = 1/12 ~ 0.0833):
    peterson_kt = 3.0 - 3.13 * (1.0 / 12.0) + 3.66 * (1.0 / 12.0)**2 - 1.53 * (1.0 / 12.0)**3
    peterson_nominal = mat_cfg.applied_traction_psi / (1.0 - 1.0 / 12.0)
    peterson_peak_psi = peterson_kt * peterson_nominal
    print("--------------------------------------------------------------------------------")
    print("THEORETICAL BENCHMARK COMPARISONS (Central Hole Baseline):")
    print(f"  Kirsch Infinite-Plate Solution (Kt=3.0):       300.00 psi")
    print(f"  Peterson Finite-Width Hole Solution (Kt*nom):   {peterson_peak_psi:.2f} psi")
    print(f"  FEA Simulated Peak Stress in Plate:            {result.max_von_mises_psi:.2f} psi")
    print(f"  Paper Reported Range (Figs 7-12):              160.0 - 300.0 psi")
    print("================================================================================")

    # Generate Visualization Figures
    print("Rendering publication-quality visual contours...")
    plot_paths = plot_simulation_results(result, geom_cfg, output_dir=output_dir, save_prefix="plate_defect")
    for k, v in plot_paths.items():
        print(f"  Saved {k}: {v}")

    # Save metadata JSON
    metadata = {
        "scientific_reference": "Singh & Pamnani (2025), MNIT Jaipur",
        "parameters": {
            "plate_length": geom_cfg.plate_length,
            "plate_width": geom_cfg.plate_width,
            "center_hole_diameter": geom_cfg.center_hole_diameter,
            "center_hole_center": [geom_cfg.center_hole_x, geom_cfg.center_hole_y],
            "defect_hole_diameter": geom_cfg.defect_hole_diameter,
            "defect_hole_center": [geom_cfg.defect_hole_x, geom_cfg.defect_hole_y],
            "youngs_modulus_psi": mat_cfg.youngs_modulus_psi,
            "youngs_modulus_pa": mat_cfg.youngs_modulus_pa,
            "poisson_ratio": mat_cfg.poisson_ratio,
            "applied_traction_psi": mat_cfg.applied_traction_psi,
            "applied_traction_pa": mat_cfg.applied_traction_pa
        },
        "mesh": {
            "num_nodes": result.num_nodes,
            "num_elements": result.num_elements,
            "h_defect": h_defect,
            "h_center": h_center,
            "h_coarse": h_coarse
        },
        "simulation_results": {
            "max_nodal_von_mises_psi": result.max_von_mises_psi,
            "max_nodal_von_mises_pa": result.max_von_mises_pa,
            "max_element_von_mises_psi": result.max_element_von_mises_psi,
            "max_element_von_mises_pa": result.max_element_von_mises_pa,
            "max_principal_stress_psi": result.max_principal_stress_psi,
            "max_principal_stress_pa": result.max_principal_stress_pa,
            "peak_nodal_location": list(result.max_stress_location),
            "peak_element_location": list(result.max_element_location),
            "max_displacement_m": result.max_displacement,
            "max_displacement_um": result.max_displacement * 1e6,
            "strain_energy_j": result.strain_energy,
            "applied_force_n": result.applied_force_total_n,
            "reaction_force_bottom_n": result.reaction_force_bottom_n,
            "horizontal_reaction_n": result.horizontal_reaction_n,
            "equilibrium_error_n": result.equilibrium_error_n
        },
        "artifacts": plot_paths
    }

    meta_file = os.path.join(output_dir, "simulation_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved simulation metadata to {meta_file}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEM Simulation of Plate with Defect")
    parser.add_argument("--x", type=float, default=0.35, help="Defect X coordinate")
    parser.add_argument("--y", type=float, default=0.65, help="Defect Y coordinate")
    parser.add_argument("--convergence", action="store_true", help="Run mesh convergence study")
    args = parser.parse_args()

    if args.convergence:
        run_convergence_study(defect_x=args.x, defect_y=args.y)
    
    run_single_simulation(defect_x=args.x, defect_y=args.y)

