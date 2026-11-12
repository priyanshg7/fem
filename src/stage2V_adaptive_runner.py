"""
Stage 2V Phase D - Adaptive FEM Simulation Engine
Executes FEM simulations for the 1,250 adaptively selected candidate configurations.
Calculates full mechanical metrics, verifies equilibrium, and saves .npz spatial fields.
"""

import os
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.geometry import PlateGeometryConfig, validate_defect_position
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, FEMResult

PLATE_WIDTH_M = 1.0  # 1000 mm along X
PA_TO_PSI = 1.4503773773020923e-4
PSI_TO_MPA = 0.006894757293168361

def simulate_adaptive_candidate(cand_dict: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Execute a single adaptive FEM simulation with complete field preservation."""
    t_start = time.perf_counter()
    
    g_family = cand_dict['geometry_family']
    sim_id = cand_dict['adaptive_simulation_id']
    
    L = float(cand_dict['plate_length_mm']) / 1000.0  # m (width along X)
    H = float(cand_dict['plate_height_mm']) / 1000.0  # m (height along Y)
    Dc = float(cand_dict['central_hole_diameter_mm']) / 1000.0  # m
    
    x = float(cand_dict['defect_center_x_mm']) / 1000.0  # m
    y = float(cand_dict['defect_center_y_mm']) / 1000.0  # m
    d = float(cand_dict['defect_diameter_mm']) / 1000.0  # m
    
    g_m = 0.01 * H  # clearance margin
    
    # Adaptive mesh rule: h_defect = d / 12.0, h_center = 3mm, h_coarse = 20mm
    h_defect = float(d / 12.0)
    h_center = 0.0030
    h_coarse = 0.0200
    
    # In PlateGeometryConfig: plate_length = H (along Y), plate_width = L (along X)
    geom_config = PlateGeometryConfig(
        plate_length=H,
        plate_width=L,
        center_hole_diameter=Dc,
        center_hole_x=L / 2.0,
        center_hole_y=H / 2.0,
        has_defect=True,
        defect_hole_diameter=float(d),
        defect_hole_x=float(x),
        defect_hole_y=float(y),
        h_coarse=h_coarse,
        h_center_hole=h_center,
        h_defect_hole=h_defect,
        clearance_margin=g_m
    )
    
    if not validate_defect_position(geom_config, margin=g_m):
        return {
            'adaptive_simulation_id': sim_id,
            'status': 'FAILED',
            'error': f'Clearance violation at ({x:.6f}, {y:.6f}, d={d:.6f})'
        }
        
    mat_config = MaterialConfig()
    result: FEMResult = solve_plane_stress_fem(geom_config, mat_config, verbose=False)
    t_duration = time.perf_counter() - t_start
    
    # Quality checks
    if result.num_nodes <= 0 or result.num_elements <= 0:
        return {'adaptive_simulation_id': sim_id, 'status': 'FAILED', 'error': 'Empty mesh'}
    if np.isnan(result.max_element_von_mises_psi) or np.isinf(result.max_element_von_mises_psi):
        return {'adaptive_simulation_id': sim_id, 'status': 'FAILED', 'error': 'Non-finite element stress'}
    if np.isnan(result.max_von_mises_psi) or np.isinf(result.max_von_mises_psi):
        return {'adaptive_simulation_id': sim_id, 'status': 'FAILED', 'error': 'Non-finite nodal stress'}
    if result.strain_energy <= 0.0:
        return {'adaptive_simulation_id': sim_id, 'status': 'FAILED', 'error': 'Non-positive strain energy'}
        
    rel_eq_err = abs(result.equilibrium_error_n) / abs(result.applied_force_total_n)
    if rel_eq_err > 1e-4:
        return {'adaptive_simulation_id': sim_id, 'status': 'FAILED', 'error': f'Equilibrium residual: {rel_eq_err:.2e}'}
        
    # Regional Stress Identification
    el_coords = result.nodes[result.elements]
    centroids = np.mean(el_coords, axis=1)
    
    rc = Dc / 2.0
    rd = d / 2.0
    dc = np.hypot(centroids[:, 0] - L / 2.0, centroids[:, 1] - H / 2.0)
    dd = np.hypot(centroids[:, 0] - x, centroids[:, 1] - y)
    
    mask_center = (dc <= 1.8 * rc) & (dc < dd)
    mask_defect = (dd <= 2.2 * rd) & (dd < dc)
    
    vm_elem_psi = result.element_von_mises * PA_TO_PSI
    max_vm_center_psi = float(np.max(vm_elem_psi[mask_center])) if np.any(mask_center) else float(result.max_element_von_mises_psi)
    max_vm_defect_psi = float(np.max(vm_elem_psi[mask_defect])) if np.any(mask_defect) else 0.0
    
    global_max_elem_idx = int(np.argmax(vm_elem_psi))
    if mask_defect[global_max_elem_idx]:
        peak_stress_region = "defect_hole"
    elif mask_center[global_max_elem_idx]:
        peak_stress_region = "center_hole"
    else:
        peak_stress_region = "far_field"
        
    # Scale-invariant metrics
    p_applied_psi = mat_config.applied_traction_psi      # 100.0 psi
    p_applied_pa = mat_config.applied_traction_pa        # 689,475.7 Pa
    
    max_elem_vm_psi = float(result.max_element_von_mises_psi)
    max_node_vm_psi = float(result.max_von_mises_psi)
    max_disp_m = float(result.max_displacement)
    strain_energy_j = float(result.strain_energy)
    
    # Save compressed full-field data
    fields_dir = Path(output_dir) / 'fields'
    fields_dir.mkdir(parents=True, exist_ok=True)
    field_abs_path = fields_dir / f"{sim_id}.npz"
    field_rel_path = f"data/adaptive_enrichment_stage2V/{g_family}/fields/{sim_id}.npz"
    
    np.savez_compressed(
        field_abs_path,
        sim_id=sim_id,
        geometry_family=g_family,
        plate_length_mm=np.float32(L * 1000.0),
        plate_height_mm=np.float32(H * 1000.0),
        defect_coords_mm=np.array([x * 1000.0, y * 1000.0], dtype=np.float32),
        defect_diameter_mm=np.float32(d * 1000.0),
        acquisition_strategy=cand_dict.get('acquisition_strategy', ''),
        element_centroids_mm=(centroids * 1000.0).astype(np.float32),
        element_von_mises_psi=vm_elem_psi.astype(np.float32),
        nodal_coords_mm=(result.nodes * 1000.0).astype(np.float32),
        nodal_von_mises_psi=(result.nodal_von_mises * PA_TO_PSI).astype(np.float32),
        displacements_m=result.displacements.astype(np.float32)
    )
    
    res_dict = dict(cand_dict)
    res_dict.update({
        'status': 'SUCCESS',
        'max_von_mises_element_psi': max_elem_vm_psi,
        'max_von_mises_element_mpa': max_elem_vm_psi * PSI_TO_MPA,
        'max_von_mises_nodal_psi': max_node_vm_psi,
        'max_von_mises_nodal_mpa': max_node_vm_psi * PSI_TO_MPA,
        'max_principal_stress_psi': float(result.max_principal_stress_psi),
        'max_displacement_m': max_disp_m,
        'strain_energy_j': strain_energy_j,
        'Kt_element': max_elem_vm_psi / p_applied_psi,
        'normalized_max_stress_element': max_elem_vm_psi / p_applied_psi,
        'normalized_max_stress_nodal': max_node_vm_psi / p_applied_psi,
        'normalized_displacement_m_per_pa': max_disp_m / p_applied_pa,
        'normalized_strain_energy_j_per_pa2': strain_energy_j / (p_applied_pa ** 2),
        'applied_force_n': float(result.applied_force_total_n),
        'reaction_force_y_n': float(result.reaction_force_bottom_n),
        'reaction_force_x_n': float(result.horizontal_reaction_n),
        'reaction_force_residual_n': float(result.equilibrium_error_n),
        'max_vm_center_psi': max_vm_center_psi,
        'max_vm_defect_psi': max_vm_defect_psi,
        'peak_stress_region': peak_stress_region,
        'node_count': int(result.num_nodes),
        'element_count': int(result.num_elements),
        'mesh_generation_time_s': round(result.mesh_generation_time, 4),
        'solve_time_s': round(result.solve_time, 4),
        'simulation_time_s': round(t_duration, 4),
        'solver_source': 'scipy_sparse_cst_plane_stress',
        'mesh_resolution': 'adaptive_hd_D_over_12_hc3mm_hg20mm',
        'field_file': field_rel_path
    })
    
    return res_dict

def run_adaptive_simulations():
    print("Starting Stage 2V Phase D Adaptive FEM Simulation Batch...")
    master_cand_path = 'data/stage2V_adaptive_selection/master_adaptive_candidates_1250.csv'
    if not os.path.exists(master_cand_path):
        print(f"Error: {master_cand_path} not found.")
        return
        
    df_cands = pd.read_csv(master_cand_path)
    total_cands = len(df_cands)
    print(f"Loaded {total_cands} adaptive candidate configurations.")
    
    base_out_dir = 'data/adaptive_enrichment_stage2V'
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Process geometry by geometry
    all_results = []
    for g_name in df_cands['geometry_family'].unique():
        sub_cands = df_cands[df_cands['geometry_family'] == g_name].to_dict(orient='records')
        geom_out_dir = os.path.join(base_out_dir, g_name)
        os.makedirs(geom_out_dir, exist_ok=True)
        
        print(f"\n--- Simulating {len(sub_cands)} adaptive cases for {g_name} ---")
        
        geom_results = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(simulate_adaptive_candidate, cand, geom_out_dir): cand['adaptive_simulation_id']
                for cand in sub_cands
            }
            
            completed = 0
            for fut in as_completed(futures):
                sim_id = futures[fut]
                try:
                    res = fut.result()
                    geom_results.append(res)
                    completed += 1
                    if completed % 50 == 0 or completed == len(sub_cands):
                        print(f"  [{g_name}] Progress: {completed}/{len(sub_cands)} completed.")
                except Exception as e:
                    print(f"  [{g_name}] Simulation {sim_id} raised exception: {e}")
                    geom_results.append({
                        'adaptive_simulation_id': sim_id,
                        'status': 'FAILED',
                        'error': str(e)
                    })
                    
        # Save geometry specific CSV
        df_geom = pd.DataFrame(geom_results)
        df_geom.to_csv(os.path.join(geom_out_dir, f'adaptive_results_{g_name}.csv'), index=False)
        all_results.extend(geom_results)
        
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(base_out_dir, 'master_adaptive_simulations_1250.csv'), index=False)
    
    succeeded = (df_all['status'] == 'SUCCESS').sum()
    failed = len(df_all) - succeeded
    print(f"\n==================================================================")
    print(f"Adaptive Simulation Campaign Completed:")
    print(f"  Total Attempted: {len(df_all)}")
    print(f"  Successfully Solved: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"==================================================================")

if __name__ == '__main__':
    run_adaptive_simulations()
