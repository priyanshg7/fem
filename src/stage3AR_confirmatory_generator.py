"""
Stage 3A-R Phase C: Independent 500 Confirmatory FEM Test Set Generator.
Generates 100 valid configurations for each of the 5 geometry families (500 total):
  - 60 space-filling samples across valid normalized parameter space.
  - 40 physically challenging samples (tight ligament, boundary proximity, large defect).
Executes high-fidelity FEM simulations with Gmsh mesh refinement and saves full .npz fields.
Strictly isolated in data/stage3AR_confirmatory_test/
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

PA_TO_PSI = 1.4503773773020923e-4

def generate_confirmatory_configurations() -> List[Dict[str, Any]]:
    """Sample 500 candidate configurations across 5 geometries."""
    np.random.seed(314159) # Dedicated independent seed for confirmatory set
    
    geometries = [
        ('rectangular_1000x600', 1000.0, 600.0, 100.0),
        ('rectangular_1000x700', 1000.0, 700.0, 100.0),
        ('rectangular_1000x800', 1000.0, 800.0, 100.0),
        ('rectangular_1000x900', 1000.0, 900.0, 100.0),
        ('square_1000x1000', 1000.0, 1000.0, 100.0),
    ]
    
    all_configs = []
    global_id = 1
    
    for g_name, L, H, Dc in geometries:
        Xc = L / 2.0
        Yc = H / 2.0
        gm = 0.01 * H
        
        # 1. 60 Space-filling samples
        space_samples = []
        while len(space_samples) < 60:
            d = np.random.uniform(5.0, 40.0) # D in [5, 40] mm
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
                    'confirmatory_id': f"CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'space_filling',
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
                
        # 2. 40 Physically Challenging samples:
        # 15 close hole interaction (clearance in [gm, 25 mm])
        # 15 boundary proximity (boundary clearance in [gm, 25 mm])
        # 10 large defect diameter (D in [30, 40 mm])
        challenging_samples = []
        
        # 15 close hole interaction
        while len([s for s in challenging_samples if s['sub_type'] == 'tight_ligament']) < 15:
            d = np.random.uniform(10.0, 35.0)
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
                    'confirmatory_id': f"CONF_{g_name}_{global_id:04d}",
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
                
        # 15 boundary proximity
        while len([s for s in challenging_samples if s.get('sub_type') == 'boundary_proximity']) < 15:
            d = np.random.uniform(10.0, 30.0)
            edge = np.random.choice(['left', 'right', 'bottom', 'top'])
            b_clear = np.random.uniform(gm + 0.5, 25.0)
            
            if edge == 'left':
                x = d / 2.0 + b_clear
                y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            elif edge == 'right':
                x = L - d / 2.0 - b_clear
                y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            elif edge == 'bottom':
                y = d / 2.0 + b_clear
                x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
            else:
                y = H - d / 2.0 - b_clear
                x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
                
            dist_c = np.hypot(x - Xc, y - Yc)
            clearance_hole = dist_c - (Dc + d) / 2.0
            
            if clearance_hole >= gm:
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                
                challenging_samples.append({
                    'confirmatory_id': f"CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging',
                    'sub_type': 'boundary_proximity',
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
                
        # 10 large defect diameter
        while len([s for s in challenging_samples if s.get('sub_type') == 'large_diameter']) < 10:
            d = np.random.uniform(32.0, 40.0)
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
                
                challenging_samples.append({
                    'confirmatory_id': f"CONF_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging',
                    'sub_type': 'large_diameter',
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
                
        all_configs.extend(space_samples)
        all_configs.extend(challenging_samples)
        
    return all_configs

def simulate_confirmatory_case(cand: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Execute high-fidelity FEM simulation for a confirmatory test case."""
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
    
    # Regional stress identification
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
        peak_region = "defect_hole"
    elif mask_center[global_max_elem_idx]:
        peak_region = "center_hole"
    else:
        peak_region = "far_field"
        
    rel_eq_err = abs(result.equilibrium_error_n) / abs(result.applied_force_total_n)
    
    # Save spatial field .npz
    fields_dir = Path(output_dir) / 'fields'
    fields_dir.mkdir(parents=True, exist_ok=True)
    field_file = fields_dir / f"{sim_id}.npz"
    
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
        'max_vm_center_psi': max_vm_center_psi,
        'max_vm_defect_psi': max_vm_defect_psi,
        'peak_stress_region': peak_region,
        'Kt_element': float(result.max_element_von_mises_psi / 100.0),
        'equilibrium_error_relative': float(rel_eq_err),
        'num_nodes': int(result.num_nodes),
        'num_elements': int(result.num_elements),
        'simulation_time_s': float(duration),
        'field_file': str(field_file)
    }
    return res_record

def run_confirmatory_campaign():
    out_dir = Path('data/stage3AR_confirmatory_test')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    configs = generate_confirmatory_configurations()
    print(f"Generated {len(configs)} candidate configurations for confirmatory test set.")
    
    df_init = pd.DataFrame(configs)
    df_init.to_csv(out_dir / 'confirmatory_sample_manifest.csv', index=False)
    print(f"Saved initial configuration manifest: {len(df_init)} cases.")
    
    print("\nStarting Parallel FEM Simulations for 500 Confirmatory Cases...")
    t_start = time.perf_counter()
    completed_records = []
    
    # Run with ProcessPoolExecutor
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing with {max_workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(simulate_confirmatory_case, c, str(out_dir)): c['confirmatory_id'] for c in configs}
        count = 0
        for fut in as_completed(future_map):
            sim_id = future_map[fut]
            try:
                rec = fut.result()
                completed_records.append(rec)
                count += 1
                if count % 50 == 0 or count == len(configs):
                    elapsed = time.perf_counter() - t_start
                    rate = count / elapsed
                    print(f"  [{count:3d}/500] Completed simulations ({rate:.2f} sim/s, elapsed: {elapsed:.1f}s)")
            except Exception as e:
                print(f"  [ERROR] {sim_id}: {e}")
                
    total_time = time.perf_counter() - t_start
    print(f"\nAll {len(completed_records)} FEM simulations completed in {total_time:.1f}s ({total_time/60:.2f} min).")
    
    df_final = pd.DataFrame(completed_records)
    df_final.to_csv(out_dir / 'confirmatory_dataset_500.csv', index=False)
    print(f"Saved final confirmatory dataset to {out_dir / 'confirmatory_dataset_500.csv'}")

if __name__ == '__main__':
    run_confirmatory_campaign()

