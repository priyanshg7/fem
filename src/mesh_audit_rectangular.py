"""
Rectangular Plate Mesh Sensitivity Audit (Stage 2R).

Evaluates the proposed adaptive mesh rule (h_defect = D / 12.0, h_center = 0.003 m, h_coarse = 0.020 m)
against a substantially refined audit mesh (h_defect = D / 18.0, h_center = 0.002 m, h_coarse = 0.014 m)
across 6 representative configurations of the 1000 mm x 600 mm rectangular plate:
1. Minimum defect diameter: D = 3.75 mm (0.00375 m)
2. Maximum defect diameter: D = 15.00 mm (0.01500 m)
3. Defect close to central hole clearance: X = 500 + 25 + 3.75 + 6.5 = 535.25 mm, Y = 300 mm (D = 7.50 mm)
4. Defect near external boundary clearance: X = 6 + 3.75 + 1.25 = 11.0 mm, Y = 300 mm (D = 7.50 mm)
5. Representative mid-range defect: X = 350 mm, Y = 450 mm, D = 7.50 mm
6. High baseline stress configuration: Large defect near central hole shoulder (X = 540 mm, Y = 300 mm, D = 15.00 mm)
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem

AUDIT_CONFIGS_RECT = [
    {
        "name": "1. Minimum Defect Diameter (D = 3.75 mm)",
        "x": 0.350, "y": 0.450, "d": 0.00375
    },
    {
        "name": "2. Maximum Defect Diameter (D = 15.00 mm)",
        "x": 0.350, "y": 0.450, "d": 0.01500
    },
    {
        "name": "3. Close to Central Hole Clearance",
        "x": 0.5355, "y": 0.300, "d": 0.00750
    },
    {
        "name": "4. Near External Boundary Clearance",
        "x": 0.0110, "y": 0.300, "d": 0.00750
    },
    {
        "name": "5. Representative Mid-Range Defect",
        "x": 0.350, "y": 0.450, "d": 0.00750
    },
    {
        "name": "6. High Stress Interaction (Large D near Shoulder)",
        "x": 0.540, "y": 0.300, "d": 0.01500
    },
]


def run_rectangular_audit():
    mat = MaterialConfig()
    results = []

    print("=" * 85)
    print("RECTANGULAR PLATE MESH SENSITIVITY AUDIT (1000 mm x 600 mm)")
    print("=" * 85)

    for case in AUDIT_CONFIGS_RECT:
        name = case["name"]
        x, y, d = case["x"], case["y"], case["d"]
        print(f"\n--- {name} ---")
        print(f"X={x*1000:.2f} mm, Y={y*1000:.2f} mm, D={d*1000:.2f} mm")

        # 1. Production Adaptive Mesh (h_defect = D / 12)
        hd_prod = d / 12.0
        hc_prod = 0.0030
        hg_prod = 0.020
        cfg_prod = PlateGeometryConfig(
            plate_width=1.0, plate_length=0.6,
            center_hole_x=0.5, center_hole_y=0.3, center_hole_diameter=0.05,
            defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
            h_defect_hole=hd_prod, h_center_hole=hc_prod, h_coarse=hg_prod,
            clearance_margin=0.006
        )
        t0 = time.perf_counter()
        res_prod = solve_plane_stress_fem(cfg_prod, mat)
        t_prod = time.perf_counter() - t0

        # 2. Finer Audit Mesh (h_defect = D / 18, h_center = 0.0020, h_coarse = 0.014)
        hd_audit = d / 18.0
        hc_audit = 0.0020
        hg_audit = 0.014
        cfg_audit = PlateGeometryConfig(
            plate_width=1.0, plate_length=0.6,
            center_hole_x=0.5, center_hole_y=0.3, center_hole_diameter=0.05,
            defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
            h_defect_hole=hd_audit, h_center_hole=hc_audit, h_coarse=hg_audit,
            clearance_margin=0.006
        )
        t0 = time.perf_counter()
        res_audit = solve_plane_stress_fem(cfg_audit, mat)
        t_audit = time.perf_counter() - t0

        diff_elem_vm = abs(res_audit.max_element_von_mises_psi - res_prod.max_element_von_mises_psi) / res_audit.max_element_von_mises_psi * 100.0
        diff_node_vm = abs(res_audit.max_von_mises_psi - res_prod.max_von_mises_psi) / res_audit.max_von_mises_psi * 100.0
        diff_disp = abs(res_audit.max_displacement - res_prod.max_displacement) / res_audit.max_displacement * 100.0
        diff_energy = abs(res_audit.strain_energy - res_prod.strain_energy) / res_audit.strain_energy * 100.0

        rec = {
            "name": name,
            "x_mm": x * 1000.0,
            "y_mm": y * 1000.0,
            "d_mm": d * 1000.0,
            "prod_elements": res_prod.num_elements,
            "audit_elements": res_audit.num_elements,
            "prod_elem_vm_psi": res_prod.max_element_von_mises_psi,
            "audit_elem_vm_psi": res_audit.max_element_von_mises_psi,
            "diff_elem_vm_pct": diff_elem_vm,
            "prod_node_vm_psi": res_prod.max_von_mises_psi,
            "audit_node_vm_psi": res_audit.max_von_mises_psi,
            "diff_node_vm_pct": diff_node_vm,
            "prod_disp_um": res_prod.max_displacement * 1e6,
            "audit_disp_um": res_audit.max_displacement * 1e6,
            "diff_disp_pct": diff_disp,
            "diff_energy_pct": diff_energy,
            "time_prod_s": t_prod,
            "time_audit_s": t_audit,
        }
        results.append(rec)

        print(f"  Production: {res_prod.num_elements} elems | Elem VM: {res_prod.max_element_von_mises_psi:.2f} psi | Node VM: {res_prod.max_von_mises_psi:.2f} psi | Disp: {res_prod.max_displacement*1e6:.3f} um")
        print(f"  Audit Mesh: {res_audit.num_elements} elems | Elem VM: {res_audit.max_element_von_mises_psi:.2f} psi | Node VM: {res_audit.max_von_mises_psi:.2f} psi | Disp: {res_audit.max_displacement*1e6:.3f} um")
        print(f"  -> Delta: Elem VM = {diff_elem_vm:.2f}%, Node VM = {diff_node_vm:.2f}%, Disp = {diff_disp:.4f}%, Energy = {diff_energy:.4f}%")

    out_path = Path("output/mesh_sensitivity_audit_rectangular.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print("RECTANGULAR MESH SENSITIVITY AUDIT SUMMARY TABLE")
    print("=" * 85)
    cols = ["name", "d_mm", "prod_elements", "audit_elements", "prod_elem_vm_psi", "audit_elem_vm_psi", "diff_elem_vm_pct", "diff_energy_pct"]
    print(df[cols].to_string(index=False))
    print(f"\nAudit results successfully saved to {out_path}")


if __name__ == "__main__":
    run_rectangular_audit()
