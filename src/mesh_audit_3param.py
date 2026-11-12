"""
Three-Parameter Adaptive Mesh Sensitivity Audit.

Evaluates the proposed adaptive mesh rule (h_defect = D / 12.0, h_center = 0.0030, h_coarse = 0.020)
against a highly refined audit mesh (h_defect = D / 18.0, h_center = 0.0020, h_coarse = 0.014)
across 6 critical configurations:
1. Original baseline: (X=0.35, Y=0.65, D=1/80)
2. Smallest defect diameter: (X=0.35, Y=0.65, D=1/160)
3. Largest defect diameter: (X=0.35, Y=0.65, D=1/40)
4. Critical central hole clearance: (X=0.575, Y=0.50, D=1/80)
5. Critical boundary clearance: (X=0.022, Y=0.50, D=1/80)
6. Top tensile boundary with large defect: (X=0.50, Y=0.965, D=1/40)
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem

AUDIT_CONFIGS = [
    {
        "name": "1. Baseline (Original D=1/80)",
        "x": 0.35, "y": 0.65, "d": 1.0 / 80.0
    },
    {
        "name": "2. Smallest Defect (D=1/160)",
        "x": 0.35, "y": 0.65, "d": 1.0 / 160.0
    },
    {
        "name": "3. Largest Defect (D=1/40)",
        "x": 0.35, "y": 0.65, "d": 1.0 / 40.0
    },
    {
        "name": "4. Near Central Hole Clearance",
        "x": 0.575, "y": 0.50, "d": 1.0 / 80.0
    },
    {
        "name": "5. Near External Boundary Clearance",
        "x": 0.022, "y": 0.50, "d": 1.0 / 80.0
    },
    {
        "name": "6. Top Traction Boundary (Large D=1/40)",
        "x": 0.50, "y": 0.965, "d": 1.0 / 40.0
    },
]


def run_3param_audit():
    mat = MaterialConfig()
    results = []

    print("=" * 85)
    print("3-PARAMETER ADAPTIVE MESH SENSITIVITY AUDIT (PRODUCTION vs FINER AUDIT)")
    print("=" * 85)

    for case in AUDIT_CONFIGS:
        name = case["name"]
        x, y, d = case["x"], case["y"], case["d"]
        print(f"\n--- {name} ---")
        print(f"Coordinates: X={x:.4f}, Y={y:.4f}, Diameter D={d:.5f} (Radius={d/2:.5f})")

        # 1. Production Adaptive Mesh (h_defect = D / 12)
        hd_prod = d / 12.0
        hc_prod = 0.0030
        hg_prod = 0.020
        cfg_prod = PlateGeometryConfig(
            defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
            h_defect_hole=hd_prod, h_center_hole=hc_prod, h_coarse=hg_prod
        )
        t0 = time.perf_counter()
        res_prod = solve_plane_stress_fem(cfg_prod, mat)
        t_prod = time.perf_counter() - t0

        # 2. Finer Audit Mesh (h_defect = D / 18, h_center = 0.0020, h_coarse = 0.014)
        hd_audit = d / 18.0
        hc_audit = 0.0020
        hg_audit = 0.014
        cfg_audit = PlateGeometryConfig(
            defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
            h_defect_hole=hd_audit, h_center_hole=hc_audit, h_coarse=hg_audit
        )
        t0 = time.perf_counter()
        res_audit = solve_plane_stress_fem(cfg_audit, mat)
        t_audit = time.perf_counter() - t0

        # Differences
        diff_elem_vm = abs(res_audit.max_element_von_mises_psi - res_prod.max_element_von_mises_psi) / res_audit.max_element_von_mises_psi * 100.0
        diff_node_vm = abs(res_audit.max_von_mises_psi - res_prod.max_von_mises_psi) / res_audit.max_von_mises_psi * 100.0
        diff_disp = abs(res_audit.max_displacement - res_prod.max_displacement) / res_audit.max_displacement * 100.0
        diff_energy = abs(res_audit.strain_energy - res_prod.strain_energy) / res_audit.strain_energy * 100.0

        rec = {
            "name": name,
            "x": x,
            "y": y,
            "d": d,
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
            "t_prod_s": t_prod,
            "t_audit_s": t_audit,
        }
        results.append(rec)

        print(f"  Production: {res_prod.num_elements} elems (hd={hd_prod:.6f}) | Elem VM: {res_prod.max_element_von_mises_psi:.2f} psi | Node VM: {res_prod.max_von_mises_psi:.2f} psi | Disp: {res_prod.max_displacement*1e6:.3f} um")
        print(f"  Audit Mesh: {res_audit.num_elements} elems (hd={hd_audit:.6f}) | Elem VM: {res_audit.max_element_von_mises_psi:.2f} psi | Node VM: {res_audit.max_von_mises_psi:.2f} psi | Disp: {res_audit.max_displacement*1e6:.3f} um")
        print(f"  -> Delta: Elem VM = {diff_elem_vm:.2f}%, Node VM = {diff_node_vm:.2f}%, Disp = {diff_disp:.4f}%, Energy = {diff_energy:.4f}%")

    out_path = Path("output/mesh_sensitivity_audit_3param.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print("3-PARAMETER MESH SENSITIVITY AUDIT SUMMARY TABLE")
    print("=" * 85)
    cols = ["name", "d", "prod_elements", "audit_elements", "prod_elem_vm_psi", "audit_elem_vm_psi", "diff_elem_vm_pct", "diff_energy_pct"]
    print(df[cols].to_string(index=False))
    print(f"\nAudit results successfully saved to {out_path}")


if __name__ == "__main__":
    run_3param_audit()

