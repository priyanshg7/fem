"""
Mesh Sensitivity Audit for Production Dataset Generation.

Compares Production Tier 3 mesh settings against Finer Tier 4 across 5 representative
defect positions spanning different interaction regimes:
1. Baseline position (0.35, 0.65) - Intermediate quadrant
2. High-interaction shoulder (0.58, 0.50) - Near central hole transverse stress concentration
3. Near outer edge (0.10, 0.10) - Far-field corner
4. Near tensile boundary (0.50, 0.90) - Top tension zone
5. Asymmetric quadrant (0.75, 0.25) - Bottom-right interior
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem


TEST_POSITIONS = [
    {"name": "Baseline (0.35, 0.65)", "x": 0.35, "y": 0.65},
    {"name": "Near Central Shoulder (0.58, 0.50)", "x": 0.58, "y": 0.50},
    {"name": "Near Corner (0.10, 0.10)", "x": 0.10, "y": 0.10},
    {"name": "Near Top Traction (0.50, 0.90)", "x": 0.50, "y": 0.90},
    {"name": "Asymmetric Quadrant (0.75, 0.25)", "x": 0.75, "y": 0.25},
]

TIER3 = {
    "tier_name": "Tier 3 (Production)",
    "h_defect": 0.0010,
    "h_center": 0.0030,
    "h_coarse": 0.020,
}

TIER4 = {
    "tier_name": "Tier 4 (Finer Audit)",
    "h_defect": 0.0007,
    "h_center": 0.0020,
    "h_coarse": 0.015,
}


def run_audit():
    mat = MaterialConfig()
    results = []

    print("=" * 80)
    print("STARTING REPRESENTATIVE MESH SENSITIVITY AUDIT (TIER 3 vs TIER 4)")
    print("=" * 80)

    for pos in TEST_POSITIONS:
        name = pos["name"]
        x, y = pos["x"], pos["y"]
        print(f"\nEvaluating Position: {name} at ({x}, {y})")

        # Tier 3 solve
        cfg3 = PlateGeometryConfig(
            defect_hole_x=x, defect_hole_y=y,
            h_defect_hole=TIER3["h_defect"],
            h_center_hole=TIER3["h_center"],
            h_coarse=TIER3["h_coarse"]
        )
        t0 = time.perf_counter()
        res3 = solve_plane_stress_fem(cfg3, mat)
        t_tier3 = time.perf_counter() - t0

        # Tier 4 solve
        cfg4 = PlateGeometryConfig(
            defect_hole_x=x, defect_hole_y=y,
            h_defect_hole=TIER4["h_defect"],
            h_center_hole=TIER4["h_center"],
            h_coarse=TIER4["h_coarse"]
        )
        t0 = time.perf_counter()
        res4 = solve_plane_stress_fem(cfg4, mat)
        t_tier4 = time.perf_counter() - t0

        # Relative differences
        diff_elem_vm = abs(res4.max_element_von_mises_psi - res3.max_element_von_mises_psi) / res4.max_element_von_mises_psi * 100.0
        diff_node_vm = abs(res4.max_von_mises_psi - res3.max_von_mises_psi) / res4.max_von_mises_psi * 100.0
        diff_disp = abs(res4.max_displacement - res3.max_displacement) / res4.max_displacement * 100.0
        diff_energy = abs(res4.strain_energy - res3.strain_energy) / res4.strain_energy * 100.0

        rec = {
            "name": name,
            "x": x,
            "y": y,
            "tier3_elements": res3.num_elements,
            "tier4_elements": res4.num_elements,
            "tier3_elem_vm_psi": res3.max_element_von_mises_psi,
            "tier4_elem_vm_psi": res4.max_element_von_mises_psi,
            "diff_elem_vm_pct": diff_elem_vm,
            "tier3_node_vm_psi": res3.max_von_mises_psi,
            "tier4_node_vm_psi": res4.max_von_mises_psi,
            "diff_node_vm_pct": diff_node_vm,
            "tier3_disp_um": res3.max_displacement * 1e6,
            "tier4_disp_um": res4.max_displacement * 1e6,
            "diff_disp_pct": diff_disp,
            "diff_energy_pct": diff_energy,
            "time_tier3_s": t_tier3,
            "time_tier4_s": t_tier4,
        }
        results.append(rec)
        print(
            f"  Tier 3: {res3.num_elements} elems | Elem VM: {res3.max_element_von_mises_psi:.2f} psi | "
            f"Node VM: {res3.max_von_mises_psi:.2f} psi | Disp: {res3.max_displacement*1e6:.3f} um"
        )
        print(
            f"  Tier 4: {res4.num_elements} elems | Elem VM: {res4.max_element_von_mises_psi:.2f} psi | "
            f"Node VM: {res4.max_von_mises_psi:.2f} psi | Disp: {res4.max_displacement*1e6:.3f} um"
        )
        print(
            f"  Delta: Elem VM = {diff_elem_vm:.2f}% | Node VM = {diff_node_vm:.2f}% | "
            f"Disp = {diff_disp:.4f}% | Energy = {diff_energy:.4f}%"
        )

    out_path = Path("output/mesh_sensitivity_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY AUDIT TABLE")
    print("=" * 80)
    print(df[["name", "tier3_elements", "tier4_elements", "tier3_elem_vm_psi", "tier4_elem_vm_psi", "diff_elem_vm_pct", "diff_energy_pct"]].to_string(index=False))
    print(f"\nAudit results successfully saved to {out_path}")


if __name__ == "__main__":
    run_audit()

