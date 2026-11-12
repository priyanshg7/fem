"""
Stage 2V Phase E - High-Resolution Mesh Refinement Audit
Selects 10 representative configurations per geometry (50 cases total) from the
Stage 2V adaptive batch and tests mesh convergence against a 2x refined mesh.
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.geometry import PlateGeometryConfig, validate_defect_position
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, FEMResult

PA_TO_PSI = 1.4503773773020923e-4

def select_audit_cases(df_adaptive: pd.DataFrame) -> pd.DataFrame:
    """Select 10 critical cases per geometry from the adaptive batch."""
    selected_rows = []
    
    for g_name in df_adaptive['geometry_family'].unique():
        sub = df_adaptive[df_adaptive['geometry_family'] == g_name].copy()
        
        idx_cases = [
            sub['max_von_mises_element_psi'].idxmax(),                      # 1. Max stress
            sub['model_disagreement_sigma'].idxmax(),                       # 2. Max disagreement
            sub['hole_to_hole_clearance_mm'].idxmin(),                      # 3. Smallest hole clearance
            sub['boundary_clearance_mm'].idxmin(),                          # 4. Smallest boundary clearance
            sub['defect_diameter_mm'].idxmin(),                             # 5. Smallest defect diameter
            sub['defect_diameter_mm'].idxmax(),                             # 6. Largest defect diameter
            sub[sub['acquisition_strategy'] == 'Strategy_A_Model_Disagreement']['model_disagreement_sigma'].idxmax(), # 7. Top Strategy A
            sub[sub['acquisition_strategy'] == 'Strategy_B_High_Error_Neighborhood']['model_disagreement_sigma'].idxmax(), # 8. Top Strategy B
            sub[sub['acquisition_strategy'] == 'Strategy_C_Physical_Risk_Coverage']['hole_to_hole_clearance_mm'].idxmin(), # 9. Top Strategy C
            sub['max_von_mises_element_psi'].sub(sub['max_von_mises_element_psi'].median()).abs().idxmin() # 10. Median stress
        ]
        
        # Deduplicate indices
        unique_indices = list(dict.fromkeys(idx_cases))
        for idx in unique_indices[:10]:
            r = sub.loc[idx].to_dict()
            selected_rows.append(r)
            
    df_audit = pd.DataFrame(selected_rows)
    print(f"Selected {len(df_audit)} audit configurations across {df_adaptive['geometry_family'].nunique()} geometries.")
    return df_audit

def run_mesh_refinement_audit(df_audit: pd.DataFrame) -> List[Dict[str, Any]]:
    """Simulate selected cases with 2x refined mesh and compare results."""
    print("Running high-resolution mesh convergence audit on selected adaptive cases...")
    
    audit_results = []
    mat_config = MaterialConfig()
    
    for i, row in df_audit.iterrows():
        sim_id = row['adaptive_simulation_id']
        g_family = row['geometry_family']
        
        L = float(row['plate_length_mm']) / 1000.0
        H = float(row['plate_height_mm']) / 1000.0
        Dc = float(row['central_hole_diameter_mm']) / 1000.0
        
        x = float(row['defect_center_x_mm']) / 1000.0
        y = float(row['defect_center_y_mm']) / 1000.0
        d = float(row['defect_diameter_mm']) / 1000.0
        g_m = 0.01 * H
        
        # Production mesh values (from previous solve)
        prod_elem_vm = float(row['max_von_mises_element_psi'])
        prod_node_vm = float(row['max_von_mises_nodal_psi'])
        prod_disp_m = float(row['max_displacement_m'])
        prod_energy_j = float(row['strain_energy_j'])
        prod_elems = int(row['element_count'])
        prod_nodes = int(row['node_count'])
        
        # 2x Refined mesh: h_defect = d / 24.0, h_center = 1.5 mm, h_coarse = 10.0 mm
        ref_h_defect = float(d / 24.0)
        ref_h_center = 0.0015
        ref_h_coarse = 0.0100
        
        geom_ref = PlateGeometryConfig(
            plate_length=H,
            plate_width=L,
            center_hole_diameter=Dc,
            center_hole_x=L / 2.0,
            center_hole_y=H / 2.0,
            has_defect=True,
            defect_hole_diameter=float(d),
            defect_hole_x=float(x),
            defect_hole_y=float(y),
            h_coarse=ref_h_coarse,
            h_center_hole=ref_h_center,
            h_defect_hole=ref_h_defect,
            clearance_margin=g_m
        )
        
        t0 = time.perf_counter()
        res_ref: FEMResult = solve_plane_stress_fem(geom_ref, mat_config, verbose=False)
        t_ref = time.perf_counter() - t0
        
        ref_elem_vm = float(res_ref.max_element_von_mises_psi)
        ref_node_vm = float(res_ref.max_von_mises_psi)
        ref_disp_m = float(res_ref.max_displacement)
        ref_energy_j = float(res_ref.strain_energy)
        ref_elems = int(res_ref.num_elements)
        ref_nodes = int(res_ref.num_nodes)
        
        # Discrepancies (%)
        diff_elem_vm_pct = abs(ref_elem_vm - prod_elem_vm) / ref_elem_vm * 100.0
        diff_node_vm_pct = abs(ref_node_vm - prod_node_vm) / ref_node_vm * 100.0
        diff_disp_pct = abs(ref_disp_m - prod_disp_m) / ref_disp_m * 100.0
        diff_energy_pct = abs(ref_energy_j - prod_energy_j) / ref_energy_j * 100.0
        
        audit_results.append({
            'adaptive_simulation_id': sim_id,
            'geometry_family': g_family,
            'x_mm': round(x * 1000.0, 2),
            'y_mm': round(y * 1000.0, 2),
            'd_mm': round(d * 1000.0, 2),
            'acquisition_strategy': row.get('acquisition_strategy', ''),
            'prod_elements': prod_elems,
            'ref_elements': ref_elems,
            'mesh_multiplier': round(ref_elems / prod_elems, 2),
            'prod_elem_vm_psi': round(prod_elem_vm, 2),
            'ref_elem_vm_psi': round(ref_elem_vm, 2),
            'diff_elem_vm_pct': round(diff_elem_vm_pct, 3),
            'prod_node_vm_psi': round(prod_node_vm, 2),
            'ref_node_vm_psi': round(ref_node_vm, 2),
            'diff_node_vm_pct': round(diff_node_vm_pct, 3),
            'prod_disp_um': round(prod_disp_m * 1e6, 3),
            'ref_disp_um': round(ref_disp_m * 1e6, 3),
            'diff_disp_pct': round(diff_disp_pct, 4),
            'prod_energy_j': round(prod_energy_j, 4),
            'ref_energy_j': round(ref_energy_j, 4),
            'diff_energy_pct': round(diff_energy_pct, 4),
            'time_ref_s': round(t_ref, 2)
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(df_audit)} refined audit simulations.")
            
    return audit_results

if __name__ == '__main__':
    adapt_path = 'data/adaptive_enrichment_stage2V/master_adaptive_simulations_1250.csv'
    if not os.path.exists(adapt_path):
        print(f"Waiting for {adapt_path}...")
    else:
        df_adapt = pd.read_csv(adapt_path)
        df_audit = select_audit_cases(df_adapt)
        results = run_mesh_refinement_audit(df_audit)
        
        os.makedirs('output', exist_ok=True)
        with open('output/stage2V_adaptive_mesh_refinement_audit.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved output/stage2V_adaptive_mesh_refinement_audit.json")

