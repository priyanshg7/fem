"""
Stage 3D Phase A: Physical Symmetry and Boundary-Value Problem Audit
====================================================================
Tests the exact FEM boundary-value problem on paired defect locations:
  1. (X, Y)
  2. (L - X, Y)    [Horizontal reflection across vertical centerline X = L/2]
  3. (X, H - Y)    [Vertical reflection across horizontal centerline Y = H/2]
  4. (L - X, H - Y) [Dual reflection / 180 deg rotation]

Examines whether reflection symmetries are:
  - Exact physical symmetries
  - Approximate symmetries
  - Not symmetries at all (broken by asymmetric boundary conditions)

Boundary Conditions of Singh & Pamnani (2025):
  - Top edge (y = H): Uniaxial tensile traction sigma_yy = +P
  - Bottom edge (y = 0): Roller constraint (u_y = 0)
  - Bottom midpoint (x = L/2, y = 0): Pin constraint (u_x = 0)
  - Sides & Hole perimeters: Traction-free
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, PA_TO_PSI


def run_symmetry_experiment(
    L: float = 1.0,
    H: float = 1.0,
    Dc: float = 1.0 / 12.0,
    d: float = 1.0 / 80.0,
    base_x: float = 0.35,
    base_y: float = 0.65,
    case_name: str = "Square_Plate_MidLigament"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Running Symmetry Experiment: {case_name}")
    print(f"Plate Dimensions: L = {L:.3f}, H = {H:.3f}")
    print(f"Hole Diameters:   Dc = {Dc:.4f}, d = {d:.4f}")
    print(f"Base Defect:      (X0, Y0) = ({base_x:.4f}, {base_y:.4f})")
    print(f"=======================================================")

    configs = [
        ("Base_Q2", base_x, base_y, "Original Location (Quadrant 2)"),
        ("Horiz_Reflect_Q1", L - base_x, base_y, "Horizontal Reflection X <-> L - X (Quadrant 1)"),
        ("Vert_Reflect_Q3", base_x, H - base_y, "Vertical Reflection Y <-> H - Y (Quadrant 3)"),
        ("Dual_Reflect_Q4", L - base_x, H - base_y, "Dual Reflection X <-> L-X, Y <-> H-Y (Quadrant 4)")
    ]

    mat_cfg = MaterialConfig(applied_traction_psi=100.0)
    results = []

    for label, x_pos, y_pos, desc in configs:
        geom_cfg = PlateGeometryConfig(
            plate_width=L,
            plate_length=H,
            center_hole_diameter=Dc,
            center_hole_x=L / 2.0,
            center_hole_y=H / 2.0,
            defect_hole_diameter=d,
            defect_hole_x=x_pos,
            defect_hole_y=y_pos,
            h_coarse=0.020,
            h_center_hole=0.003,
            h_defect_hole=0.001
        )
        fem_res = solve_plane_stress_fem(geom_cfg, mat_cfg, verbose=False)
        
        elem_vm_psi = fem_res.max_element_von_mises_psi
        nodal_vm_psi = fem_res.max_von_mises_psi
        elem_loc = fem_res.max_element_location
        u_max_mm = fem_res.max_displacement * 1000.0
        se = fem_res.strain_energy
        rx_pin = fem_res.horizontal_reaction_n
        ry_bottom = fem_res.reaction_force_bottom_n
        top_force = fem_res.applied_force_total_n
        eq_err = fem_res.equilibrium_error_n
        
        res_entry = {
            'label': label,
            'description': desc,
            'x': float(x_pos),
            'y': float(y_pos),
            'elem_max_vm_psi': float(elem_vm_psi),
            'nodal_max_vm_psi': float(nodal_vm_psi),
            'peak_elem_loc': [float(elem_loc[0]), float(elem_loc[1])],
            'max_displacement_mm': float(u_max_mm),
            'strain_energy_j': float(se),
            'rx_pin_n': float(rx_pin),
            'ry_bottom_n': float(ry_bottom),
            'top_force_n': float(top_force),
            'equilibrium_err_n': float(eq_err),
            'num_nodes': int(fem_res.num_nodes),
            'num_elements': int(fem_res.num_elements)
        }
        results.append(res_entry)
        print(f"  [{label}] pos=({x_pos:.3f}, {y_pos:.3f}) -> Elem VM: {elem_vm_psi:.3f} psi, Nodal VM: {nodal_vm_psi:.3f} psi, SE: {se:.4f} J, Rx Pin: {rx_pin:.2e} N")

    # Comparisons against Base_Q2
    base = results[0]
    comp_dict = {}
    for r in results[1:]:
        d_elem_vm = r['elem_max_vm_psi'] - base['elem_max_vm_psi']
        d_nodal_vm = r['nodal_max_vm_psi'] - base['nodal_max_vm_psi']
        d_se = r['strain_energy_j'] - base['strain_energy_j']
        rel_elem_diff_pct = abs(d_elem_vm) / base['elem_max_vm_psi'] * 100.0
        comp_dict[r['label']] = {
            'delta_elem_vm_psi': float(d_elem_vm),
            'delta_nodal_vm_psi': float(d_nodal_vm),
            'delta_strain_energy_j': float(d_se),
            'rel_elem_diff_pct': float(rel_elem_diff_pct)
        }
        print(f"  --> vs Base ({r['label']}): Delta Elem VM = {d_elem_vm:+.4f} psi ({rel_elem_diff_pct:.4f}%), Delta SE = {d_se:+.6f} J")

    return {
        'case_name': case_name,
        'plate_dims': {'L': L, 'H': H},
        'holes': {'Dc': Dc, 'd': d},
        'results': results,
        'comparisons_to_base': comp_dict
    }


def run_all_symmetry_audits():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)

    cases = [
        # 1. Square plate, interior defect
        run_symmetry_experiment(L=1.0, H=1.0, Dc=1.0/12.0, d=1.0/80.0, base_x=0.35, base_y=0.65, case_name="Square_Interior_0.35_0.65"),
        # 2. Square plate, near-boundary defect
        run_symmetry_experiment(L=1.0, H=1.0, Dc=1.0/12.0, d=1.0/80.0, base_x=0.15, base_y=0.80, case_name="Square_NearBoundary_0.15_0.80"),
        # 3. Rectangular plate (1000x600 mm)
        run_symmetry_experiment(L=1.0, H=0.6, Dc=1.0/12.0, d=1.0/80.0, base_x=0.30, base_y=0.45, case_name="Rectangular_1000x600_0.30_0.45")
    ]

    out_path = output_dir / "stage3D_phaseA_symmetry_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\nSaved symmetry audit results to {out_path}")


if __name__ == "__main__":
    run_all_symmetry_audits()

