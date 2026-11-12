"""
Stage 3C Phase H: Dedicated Inverse Confirmatory FEM Dataset Generator.
Generates 100 valid configurations for each of the 5 geometry families (500 total):
  - 60 space-filling configurations (variable diameter).
  - 20 physically challenging configurations (tight ligament or boundary).
  - 20 controlled fixed-diameter configurations (d/Dc = 0.150 for reference evaluation).
Executes high-fidelity FEM simulations with Gmsh mesh refinement and saves full fields.
Strictly isolated in data/stage3C_inverse_confirmatory/
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.geometry import PlateGeometryConfig
from src.fem_solver import MaterialConfig, solve_plane_stress_fem

PA_TO_PSI = 1.4503773773020923e-4

def generate_stage3C_confirmatory_configs() -> List[Dict[str, Any]]:
    # Dedicated independent seed for Stage 3C inverse confirmatory set
    np.random.seed(554433)
    
    geometries = [
        ('rectangular_1000x600', 1000.0, 600.0, 50.0000),
        ('rectangular_1000x700', 1000.0, 700.0, 58.3333),
        ('rectangular_1000x800', 1000.0, 800.0, 66.6667),
        ('rectangular_1000x900', 1000.0, 900.0, 75.0000),
        ('square_1000x1000', 1000.0, 1000.0, 83.3333),
    ]
    
    all_configs = []
    global_id = 1
    
    for g_name, L, H, Dc in geometries:
        Xc = L / 2.0
        Yc = H / 2.0
        gm = 0.01 * H
        
        # 1. 60 Space-filling samples (variable diameter d/Dc in [0.075, 0.300])
        space_samples = []
        while len(space_samples) < 60:
            d_ratio = np.random.uniform(0.075, 0.300)
            d = d_ratio * Dc
            x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
            y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            
            dist_c = np.hypot(x - Xc, y - Yc)
            clearance_hole = dist_c - (Dc + d) / 2.0
            
            if clearance_hole >= gm:
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                
                space_samples.append({
                    'confirmatory_id': f"STAGE3C_CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'space_filling',
                    'sub_type': 'variable_diameter_space_filling',
                    'plate_length_mm': L,
                    'plate_height_mm': H,
                    'aspect_ratio': L / H,
                    'central_hole_diameter_mm': Dc,
                    'defect_diameter_mm': d,
                    'defect_center_x_mm': x,
                    'defect_center_y_mm': y,
                    'defect_x_over_L': x / L,
                    'defect_y_over_H': y / H,
                    'defect_d_over_Dc': d / Dc,
                    'defect_d_over_H': d / H,
                    'hole_to_hole_clearance_mm': clearance_hole,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        # 2. 20 Physically challenging samples (tight ligament < 25mm or boundary < 25mm)
        challenging_samples = []
        while len(challenging_samples) < 20:
            d_ratio = np.random.uniform(0.10, 0.25)
            d = d_ratio * Dc
            target_clearance = np.random.uniform(gm + 0.5, 25.0)
            target_dist = target_clearance + (Dc + d) / 2.0
            
            angle = np.random.uniform(0, 2 * np.pi)
            x = Xc + target_dist * np.cos(angle)
            y = Yc + target_dist * np.sin(angle)
            
            c_left = x - d / 2.0
            c_right = L - x - d / 2.0
            c_bot = y - d / 2.0
            c_top = H - y - d / 2.0
            c_bnd = min(c_left, c_right, c_bot, c_top)
            
            if c_bnd >= gm:
                challenging_samples.append({
                    'confirmatory_id': f"STAGE3C_CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging',
                    'sub_type': 'tight_ligament',
                    'plate_length_mm': L,
                    'plate_height_mm': H,
                    'aspect_ratio': L / H,
                    'central_hole_diameter_mm': Dc,
                    'defect_diameter_mm': d,
                    'defect_center_x_mm': x,
                    'defect_center_y_mm': y,
                    'defect_x_over_L': x / L,
                    'defect_y_over_H': y / H,
                    'defect_d_over_Dc': d / Dc,
                    'defect_d_over_H': d / H,
                    'hole_to_hole_clearance_mm': target_clearance,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        # 3. 20 Controlled Fixed-Diameter samples (d/Dc = 0.150 exactly, matching reference paper)
        fixed_samples = []
        while len(fixed_samples) < 20:
            d = 0.150 * Dc # exactly 0.150 matching Singh & Pamnani
            x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
            y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            
            dist_c = np.hypot(x - Xc, y - Yc)
            clearance_hole = dist_c - (Dc + d) / 2.0
            
            if clearance_hole >= gm:
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                
                fixed_samples.append({
                    'confirmatory_id': f"STAGE3C_CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'fixed_diameter_reference',
                    'sub_type': 'fixed_diameter_0.150',
                    'plate_length_mm': L,
                    'plate_height_mm': H,
                    'aspect_ratio': L / H,
                    'central_hole_diameter_mm': Dc,
                    'defect_diameter_mm': d,
                    'defect_center_x_mm': x,
                    'defect_center_y_mm': y,
                    'defect_x_over_L': x / L,
                    'defect_y_over_H': y / H,
                    'defect_d_over_Dc': 0.150,
                    'defect_d_over_H': d / H,
                    'hole_to_hole_clearance_mm': clearance_hole,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        all_configs.extend(space_samples)
        all_configs.extend(challenging_samples)
        all_configs.extend(fixed_samples)
        
    return all_configs

def simulate_case_worker_3c(cand: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sim_id = cand['confirmatory_id']
    g_family = cand['geometry_family']
    
    L = float(cand['plate_length_mm']) / 1000.0
    H = float(cand['plate_height_mm']) / 1000.0
    Dc = float(cand['central_hole_diameter_mm']) / 1000.0
    x = float(cand['defect_center_x_mm']) / 1000.0
    y = float(cand['defect_center_y_mm']) / 1000.0
    d = float(cand['defect_diameter_mm']) / 1000.0
    gm = 0.01 * H
    
    h_defect = float(d / 12.0)
    h_center = 0.0030
    h_coarse = 0.0200
    
    geom_config = PlateGeometryConfig(
        plate_length=H,
        plate_width=L,
        center_hole_diameter=Dc,
        center_hole_x=L / 2.0,
        center_hole_y=H / 2.0,
        has_defect=True,
        defect_hole_diameter=d,
        defect_hole_x=x,
        defect_hole_y=y,
        h_coarse=h_coarse,
        h_center_hole=h_center,
        h_defect_hole=h_defect,
        clearance_margin=gm
    )
    
    mat_config = MaterialConfig()
    result = solve_plane_stress_fem(geom_config, mat_config, verbose=False)
    duration = time.perf_counter() - t0
    
    rel_eq_err = abs(result.equilibrium_error_n) / abs(result.applied_force_total_n)
    
    # Save spatial field .npz
    fields_dir = Path(output_dir) / 'fields'
    fields_dir.mkdir(parents=True, exist_ok=True)
    field_file = fields_dir / f"{sim_id}.npz"
    
    vm_elem_psi = result.element_von_mises * PA_TO_PSI
    np.savez_compressed(
        field_file,
        sim_id=sim_id,
        geometry_family=g_family,
        nodes=result.nodes,
        elements=result.elements,
        displacements=result.displacements,
        element_von_mises_psi=vm_elem_psi,
        nodal_von_mises_psi=result.nodal_von_mises * PA_TO_PSI
    )
    
    res_record = {
        **cand,
        'max_von_mises_element_psi': float(result.max_element_von_mises_psi),
        'max_von_mises_nodal_psi': float(result.max_von_mises_psi),
        'Kt_element': float(result.max_element_von_mises_psi / 100.0),
        'equilibrium_error_relative': float(rel_eq_err),
        'reaction_force_residual_n': float(result.equilibrium_error_n),
        'num_nodes': int(result.num_nodes),
        'num_elements': int(result.num_elements),
        'simulation_time_s': float(duration),
        'field_file': str(field_file)
    }
    return res_record

def run_stage3C_confirmatory_campaign():
    out_dir = Path('data/stage3C_inverse_confirmatory')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_dir / 'confirmatory_dataset_500.csv'
    if csv_file.exists():
        df_ex = pd.read_csv(csv_file)
        if len(df_ex) == 500:
            print(f"Stage 3C Confirmatory dataset already exists with {len(df_ex)} runs. Ready.")
            return
            
    configs = generate_stage3C_confirmatory_configs()
    print(f"Generated {len(configs)} configurations for Stage 3C Confirmatory Set. Starting parallel FEM...")
    
    t0 = time.perf_counter()
    results = []
    max_workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simulate_case_worker_3c, cfg, str(out_dir)): cfg['confirmatory_id'] for cfg in configs}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(configs):
                print(f"  Stage 3C FEM completed {completed}/{len(configs)} ({completed/len(configs)*100:.1f}%)")
                
    elapsed = time.perf_counter() - t0
    print(f"Stage 3C 500 FEM simulations completed in {elapsed:.2f}s ({elapsed/500:.3f} s/sim).")
    
    df_results = pd.DataFrame(results)
    df_results.sort_values(by='confirmatory_id', inplace=True)
    df_results.to_csv(csv_file, index=False)
    print(f"Saved dataset to {csv_file}")
    print(f"Max relative equilibrium error: {df_results['equilibrium_error_relative'].max():.2e}")

if __name__ == "__main__":
    run_stage3C_confirmatory_campaign()

