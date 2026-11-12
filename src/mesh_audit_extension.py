"""
Multi-Geometry Extension Mesh Sensitivity Audit (Stage 2S).

Evaluates the proposed adaptive mesh rule (h_defect = D / 12.0, h_center = 0.003 m, h_coarse = 0.020 m)
against a 2x finer audit mesh (h_defect = D / 18.0, h_center = 0.002 m, h_coarse = 0.014 m)
across 6 representative configurations for EACH of the three new geometries:
- RECT_700: 1000 mm x 700 mm
- RECT_800: 1000 mm x 800 mm
- RECT_900: 1000 mm x 900 mm

Representative configurations per geometry:
1. Minimum defect diameter: D_min = H / 160
2. Maximum defect diameter: D_max = H / 40
3. Defect close to central hole clearance: X = 500 + Rc + r + 1.2*g, Y = H/2
4. Defect near external boundary clearance: X = r + 1.2*g, Y = H/2
5. Representative intermediate configuration: X = 350 mm, Y = H*0.7, D = H/80
6. High baseline stress configuration: Large defect near central hole shoulder (X = 500 + Rc + r_max + 1.2*g, Y = H/2, D_max)
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem


GEOMETRIES_CONFIG = {
    "RECT_700": {
        "H": 0.700, "Dc": 0.700 / 12.0, "g": 0.007,
        "D_min": 0.700 / 160.0, "D_fixed": 0.700 / 80.0, "D_max": 0.700 / 40.0
    },
    "RECT_800": {
        "H": 0.800, "Dc": 0.800 / 12.0, "g": 0.008,
        "D_min": 0.800 / 160.0, "D_fixed": 0.800 / 80.0, "D_max": 0.800 / 40.0
    },
    "RECT_900": {
        "H": 0.900, "Dc": 0.900 / 12.0, "g": 0.009,
        "D_min": 0.900 / 160.0, "D_fixed": 0.900 / 80.0, "D_max": 0.900 / 40.0
    },
}


def run_extension_audit():
    mat = MaterialConfig()
    all_results = {}

    print("=" * 85)
    print("STAGE 2S MULTI-GEOMETRY MESH SENSITIVITY AUDIT (RECT_700, RECT_800, RECT_900)")
    print("=" * 85)

    for geom_id, gcfg in GEOMETRIES_CONFIG.items():
        H = gcfg["H"]
        Dc = gcfg["Dc"]
        Rc = Dc / 2.0
        g = gcfg["g"]
        d_min = gcfg["D_min"]
        d_fixed = gcfg["D_fixed"]
        d_max = gcfg["D_max"]

        cases = [
            {"name": "1. Minimum Defect Diameter", "x": 0.350, "y": H * 0.70, "d": d_min},
            {"name": "2. Maximum Defect Diameter", "x": 0.350, "y": H * 0.70, "d": d_max},
            {"name": "3. Close to Central Hole Clearance", "x": 0.500 + Rc + (d_fixed / 2.0) + (1.2 * g), "y": H / 2.0, "d": d_fixed},
            {"name": "4. Near External Boundary Clearance", "x": (d_fixed / 2.0) + (1.2 * g), "y": H / 2.0, "d": d_fixed},
            {"name": "5. Representative Intermediate", "x": 0.350, "y": H * 0.70, "d": d_fixed},
            {"name": "6. High Stress Interaction (Large D near Shoulder)", "x": 0.500 + Rc + (d_max / 2.0) + (1.2 * g), "y": H / 2.0, "d": d_max},
        ]

        print(f"\n{'='*30} GEOMETRY: {geom_id} (1000 mm x {H*1000:.0f} mm, g={g*1000:.1f} mm) {'='*30}")
        geom_results = []

        for c in cases:
            name = c["name"]
            x, y, d = c["x"], c["y"], c["d"]

            # Production Mesh
            hd_prod = d / 12.0
            cfg_prod = PlateGeometryConfig(
                plate_width=1.0, plate_length=H,
                center_hole_x=0.50, center_hole_y=H / 2.0, center_hole_diameter=Dc,
                defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
                h_defect_hole=hd_prod, h_center_hole=0.0030, h_coarse=0.020,
                clearance_margin=g
            )
            t0 = time.perf_counter()
            res_prod = solve_plane_stress_fem(cfg_prod, mat)
            t_prod = time.perf_counter() - t0

            # Audit Mesh (finer)
            hd_audit = d / 18.0
            cfg_audit = PlateGeometryConfig(
                plate_width=1.0, plate_length=H,
                center_hole_x=0.50, center_hole_y=H / 2.0, center_hole_diameter=Dc,
                defect_hole_x=x, defect_hole_y=y, defect_hole_diameter=d,
                h_defect_hole=hd_audit, h_center_hole=0.0020, h_coarse=0.014,
                clearance_margin=g
            )
            t0 = time.perf_counter()
            res_audit = solve_plane_stress_fem(cfg_audit, mat)
            t_audit = time.perf_counter() - t0

            diff_elem = abs(res_audit.max_element_von_mises_psi - res_prod.max_element_von_mises_psi) / res_audit.max_element_von_mises_psi * 100.0
            diff_node = abs(res_audit.max_von_mises_psi - res_prod.max_von_mises_psi) / res_audit.max_von_mises_psi * 100.0
            diff_disp = abs(res_audit.max_displacement - res_prod.max_displacement) / res_audit.max_displacement * 100.0
            diff_energy = abs(res_audit.strain_energy - res_prod.strain_energy) / res_audit.strain_energy * 100.0

            rec = {
                "case": name,
                "x_mm": round(x * 1000.0, 2),
                "y_mm": round(y * 1000.0, 2),
                "d_mm": round(d * 1000.0, 2),
                "prod_elements": res_prod.num_elements,
                "audit_elements": res_audit.num_elements,
                "prod_elem_vm_psi": round(res_prod.max_element_von_mises_psi, 2),
                "audit_elem_vm_psi": round(res_audit.max_element_von_mises_psi, 2),
                "diff_elem_vm_pct": round(diff_elem, 3),
                "prod_node_vm_psi": round(res_prod.max_von_mises_psi, 2),
                "audit_node_vm_psi": round(res_audit.max_von_mises_psi, 2),
                "diff_node_vm_pct": round(diff_node, 3),
                "diff_disp_pct": round(diff_disp, 4),
                "diff_energy_pct": round(diff_energy, 4),
                "time_prod_s": round(t_prod, 3),
                "time_audit_s": round(t_audit, 3)
            }
            geom_results.append(rec)
            print(f"  {name} (d={d*1000:.2f}mm): Prod={res_prod.num_elements} el, Audit={res_audit.num_elements} el | Delta: Elem={diff_elem:.2f}%, Disp={diff_disp:.4f}%, Energy={diff_energy:.4f}%")

        all_results[geom_id] = geom_results

    out_path = Path("output/mesh_sensitivity_audit_extension.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll extension audit results successfully saved to {out_path}")


if __name__ == "__main__":
    run_extension_audit()

