"""
Extended Mesh Convergence and Numerical Verification Study.

Conducts systematic refinement of triangular elements around both hole boundaries
to evaluate asymptotic convergence of the maximum Von Mises stress (both recovered nodal
and element centroid values), maximum displacement, and total strain energy.
"""

import os
import json
from dataclasses import asdict
from typing import List, Dict, Any
import numpy as np

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem
from src.visualization import plot_mesh_convergence


def run_convergence_study(
    defect_x: float = 0.35,
    defect_y: float = 0.65,
    output_json: str = "output/convergence_report.json",
    output_plot: str = "output/extended_mesh_convergence.png"
) -> Dict[str, Any]:
    """
    Executes simulations over 5 progressive refinement tiers and computes
    relative changes and convergence indicators.
    """
    refinements = [
        {"name": "Tier 1 (Coarse)",    "h_defect": 0.0030, "h_center": 0.0080, "h_coarse": 0.040},
        {"name": "Tier 2 (Medium)",    "h_defect": 0.0018, "h_center": 0.0050, "h_coarse": 0.030},
        {"name": "Tier 3 (Fine)",      "h_defect": 0.0010, "h_center": 0.0030, "h_coarse": 0.020},
        {"name": "Tier 4 (Very Fine)", "h_defect": 0.0006, "h_center": 0.0020, "h_coarse": 0.015},
        {"name": "Tier 5 (Ultra Fine)","h_defect": 0.00035,"h_center": 0.0012, "h_coarse": 0.010},
    ]

    results_table = []
    mat = MaterialConfig()

    print("====================================================================================================")
    print("EXTENDED MESH CONVERGENCE STUDY (2D Plane Stress FEM)")
    print(f"Defect Location: X={defect_x:.3f}, Y={defect_y:.3f} | Nominal Load P = 100 psi")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Tier':<18} | {'h_defect':<8} | {'Nodes':<6} | {'Elems':<6} | {'Nodal vM':<10} | {'Elem vM':<10} | {'Disp (um)':<10} | {'Energy (J)':<10} | {'Rel dSigma':<10}")
    print("----------------------------------------------------------------------------------------------------")

    for i, tier in enumerate(refinements):
        geom = PlateGeometryConfig(
            defect_hole_x=defect_x,
            defect_hole_y=defect_y,
            h_coarse=tier["h_coarse"],
            h_center_hole=tier["h_center"],
            h_defect_hole=tier["h_defect"]
        )

        res = solve_plane_stress_fem(geom, mat, verbose=False)

        row = {
            "tier": tier["name"],
            "h_defect": tier["h_defect"],
            "h_center": tier["h_center"],
            "h_coarse": tier["h_coarse"],
            "num_nodes": res.num_nodes,
            "num_elements": res.num_elements,
            "num_dofs": 2 * res.num_nodes,
            "nodal_vm_psi": float(res.max_von_mises_psi),
            "nodal_vm_pa": float(res.max_von_mises_pa),
            "elem_vm_psi": float(res.max_element_von_mises_psi),
            "elem_vm_pa": float(res.max_element_von_mises_pa),
            "sigma_1_psi": float(res.max_principal_stress_psi),
            "peak_nodal_loc": [float(res.max_stress_location[0]), float(res.max_stress_location[1])],
            "peak_elem_loc": [float(res.max_element_location[0]), float(res.max_element_location[1])],
            "max_displacement_um": float(res.max_displacement * 1e6),
            "strain_energy_j": float(res.strain_energy),
            "applied_force_n": float(res.applied_force_total_n),
            "reaction_force_bottom_n": float(res.reaction_force_bottom_n),
            "equilibrium_error_n": float(res.equilibrium_error_n),
            "rel_change_nodal_pct": 0.0,
            "rel_change_elem_pct": 0.0,
            "rel_change_energy_pct": 0.0
        }

        if i > 0:
            prev = results_table[i - 1]
            row["rel_change_nodal_pct"] = float(abs(row["nodal_vm_psi"] - prev["nodal_vm_psi"]) / prev["nodal_vm_psi"] * 100.0)
            row["rel_change_elem_pct"] = float(abs(row["elem_vm_psi"] - prev["elem_vm_psi"]) / prev["elem_vm_psi"] * 100.0)
            row["rel_change_energy_pct"] = float(abs(row["strain_energy_j"] - prev["strain_energy_j"]) / prev["strain_energy_j"] * 100.0)

        results_table.append(row)

        rel_str = f"{row['rel_change_nodal_pct']:.2f}%" if i > 0 else "Base"
        print(
            f"{row['tier']:<18} | {row['h_defect']:<8.4f} | {row['num_nodes']:<6} | {row['num_elements']:<6} | "
            f"{row['nodal_vm_psi']:<10.2f} | {row['elem_vm_psi']:<10.2f} | {row['max_displacement_um']:<10.4f} | "
            f"{row['strain_energy_j']:<10.4f} | {rel_str:<10}"
        )

    print("====================================================================================================")

    # Convergence Criterion Evaluation:
    # Criterion 1: Strain energy change < 0.1% between final two tiers
    # Criterion 2: Displacement change < 0.1%
    # Criterion 3: Stress change < 3.0%
    last = results_table[-1]
    second_last = results_table[-2]
    d_energy = last["rel_change_energy_pct"]
    d_nodal_stress = last["rel_change_nodal_pct"]
    d_elem_stress = last["rel_change_elem_pct"]
    d_disp = abs(last["max_displacement_um"] - second_last["max_displacement_um"]) / second_last["max_displacement_um"] * 100.0

    energy_converged = bool(d_energy < 0.1)
    disp_converged = bool(d_disp < 0.1)
    elem_stress_converged = bool(d_elem_stress < 3.0)

    print(f"Convergence Evaluation (Tier 4 -> Tier 5):")
    print(f"  Strain Energy Rel Change:   {d_energy:.4f}%  {'[SATISFIED < 0.1%]' if energy_converged else '[NOT SATISFIED]'}")
    print(f"  Max Displacement Rel Change: {d_disp:.4f}%  {'[SATISFIED < 0.1%]' if disp_converged else '[NOT SATISFIED]'}")
    print(f"  Element Max Stress Rel Change: {d_elem_stress:.2f}%  {'[SATISFIED < 3.0%]' if elem_stress_converged else '[NOT SATISFIED]'}")
    print(f"  Nodal Max Stress Rel Change:   {d_nodal_stress:.2f}%")
    print("====================================================================================================")

    # Generate and save plots
    h_sizes = [r["h_defect"] for r in results_table]
    nodal_stresses = [r["nodal_vm_psi"] for r in results_table]
    elem_stresses = [r["elem_vm_psi"] for r in results_table]
    elems = [r["num_elements"] for r in results_table]
    energies = [r["strain_energy_j"] for r in results_table]

    plot_mesh_convergence(
        h_sizes,
        nodal_stresses,
        elems,
        elem_stresses=elem_stresses,
        energies=energies,
        output_path=output_plot
    )
    # Also save standard name
    plot_mesh_convergence(
        h_sizes,
        nodal_stresses,
        elems,
        elem_stresses=elem_stresses,
        energies=energies,
        output_path="output/mesh_convergence.png"
    )

    report = {
        "defect_location": [defect_x, defect_y],
        "refinement_tiers": results_table,
        "criteria": {
            "energy_rel_change_pct": d_energy,
            "energy_converged": energy_converged,
            "disp_rel_change_pct": d_disp,
            "disp_converged": disp_converged,
            "elem_stress_rel_change_pct": d_elem_stress,
            "elem_stress_converged": elem_stress_converged,
            "nodal_stress_rel_change_pct": d_nodal_stress
        },
        "plot_path": output_plot
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_convergence_study()
