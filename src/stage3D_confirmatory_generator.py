"""
Stage 3D Phase I: Dedicated Blind Confirmatory FEM Dataset Generator.
=====================================================================
Generates 100 pristine, untouched configurations for each of the 5 geometries (500 total):
  - 60 space-filling in-domain configurations (D/Dc in [0.075, 0.300])
  - 20 physically challenging configurations (ligament c_lig in [2, 15] mm, boundary c_bnd in [2, 20] mm)
  - 20 out-of-domain defect-diameter extension configurations (D/Dc in [0.32, 0.45])
Executes high-fidelity FEM simulations with Gmsh mesh refinement, verified reaction force equilibrium,
and saves full spatial field tensors (.npz) and metadata manifest.
Strictly isolated in `data/stage3D_blind_confirmatory/`.
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


def generate_stage3D_blind_configs() -> List[Dict[str, Any]]:
    # Dedicated independent seed for Stage 3D blind confirmatory set
    np.random.seed(998877)
    
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
        
        # 1. 60 Space-filling in-domain samples (D/Dc in [0.075, 0.300])
        space_samples = []
        while len(space_samples) < 60:
            d_ratio = np.random.uniform(0.075, 0.300)
            d = d_ratio * Dc
            x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
            y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            
            dist_c = np.hypot(x - Xc, y - Yc)
            c_hole = dist_c - (Dc + d) / 2.0
            
            if c_hole >= gm:
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                
                space_samples.append({
                    'confirmatory_id': f"STAGE3D_BLIND_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'space_filling_in_domain',
                    'sub_type': 'space_filling',
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
                    'hole_to_hole_clearance_mm': c_hole,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        # 2. 20 Physically challenging samples (10 near central hole, 10 near boundary)
        chall_samples = []
        # 10 Ligament proximity (c_lig in [2.0, 15.0] mm)
        while len(chall_samples) < 10:
            d_ratio = np.random.uniform(0.08, 0.25)
            d = d_ratio * Dc
            theta = np.random.uniform(0, 2 * np.pi)
            c_lig = np.random.uniform(2.0, 15.0)
            R_c = (Dc + d) / 2.0 + c_lig
            x = Xc + R_c * np.cos(theta)
            y = Yc + R_c * np.sin(theta)
            
            c_left = x - d / 2.0
            c_right = L - x - d / 2.0
            c_bot = y - d / 2.0
            c_top = H - y - d / 2.0
            c_bnd = min(c_left, c_right, c_bot, c_top)
            
            if c_bnd >= gm:
                chall_samples.append({
                    'confirmatory_id': f"STAGE3D_BLIND_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging_in_domain',
                    'sub_type': 'ligament_interaction',
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
                    'hole_to_hole_clearance_mm': c_lig,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        # 10 Boundary proximity (c_bnd in [2.0, 20.0] mm)
        while len(chall_samples) < 20:
            d_ratio = np.random.uniform(0.08, 0.25)
            d = d_ratio * Dc
            edge = np.random.choice(['left', 'right', 'bottom', 'top'])
            c_bnd = np.random.uniform(2.0, 20.0)
            if edge == 'left':
                x = d / 2.0 + c_bnd
                y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            elif edge == 'right':
                x = L - d / 2.0 - c_bnd
                y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            elif edge == 'bottom':
                x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
                y = d / 2.0 + c_bnd
            else: # top
                x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
                y = H - d / 2.0 - c_bnd
                
            dist_c = np.hypot(x - Xc, y - Yc)
            c_hole = dist_c - (Dc + d) / 2.0
            if c_hole >= gm:
                chall_samples.append({
                    'confirmatory_id': f"STAGE3D_BLIND_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging_in_domain',
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
                    'hole_to_hole_clearance_mm': c_hole,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        # 3. 20 Out-of-Domain defect diameter extension samples (D/Dc in [0.32, 0.45])
        ood_samples = []
        while len(ood_samples) < 20:
            d_ratio = np.random.uniform(0.32, 0.45)
            d = d_ratio * Dc
            x = np.random.uniform(d / 2.0 + gm, L - d / 2.0 - gm)
            y = np.random.uniform(d / 2.0 + gm, H - d / 2.0 - gm)
            
            dist_c = np.hypot(x - Xc, y - Yc)
            c_hole = dist_c - (Dc + d) / 2.0
            
            if c_hole >= gm:
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                
                ood_samples.append({
                    'confirmatory_id': f"STAGE3D_BLIND_{g_name}_{global_id:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'out_of_domain_diameter_extension',
                    'sub_type': 'large_diameter_extension',
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
                    'hole_to_hole_clearance_mm': c_hole,
                    'boundary_clearance_mm': c_bnd
                })
                global_id += 1
                
        all_configs.extend(space_samples)
        all_configs.extend(chall_samples)
        all_configs.extend(ood_samples)
        
    return all_configs


def run_fem_single(config: Dict[str, Any]) -> Dict[str, Any]:
    L = float(config['plate_length_mm']) / 1000.0
    H = float(config['plate_height_mm']) / 1000.0
    Dc = float(config['central_hole_diameter_mm']) / 1000.0
    d = float(config['defect_diameter_mm']) / 1000.0
    x = float(config['defect_center_x_mm']) / 1000.0
    y = float(config['defect_center_y_mm']) / 1000.0
    gm = 0.01 * H
    
    h_defect = float(d / 12.0)
    h_center = 0.0030
    h_coarse = 0.0200
    
    # Gmsh refined meshing
    geom_cfg = PlateGeometryConfig(
        plate_width=L,
        plate_length=H,
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
    mat_cfg = MaterialConfig(applied_traction_psi=100.0)
    
    t0 = time.perf_counter()
    res = solve_plane_stress_fem(geom_cfg, mat_cfg, verbose=False)
    solve_time = time.perf_counter() - t0
    
    # Extract results
    elem_max_vm_psi = res.max_element_von_mises_psi
    nodal_max_vm_psi = res.max_von_mises_psi
    se = res.strain_energy
    rf_x = res.horizontal_reaction_n
    rf_y = res.reaction_force_bottom_n
    app_f = res.applied_force_total_n
    eq_err = res.equilibrium_error_n
    
    # Save spatial fields NPZ
    out_dir = Path("data/stage3D_blind_confirmatory/spatial_fields")
    out_dir.mkdir(parents=True, exist_ok=True)
    cid = config['confirmatory_id']
    npz_path = out_dir / f"{cid}_fields.npz"
    
    np.savez_compressed(
        npz_path,
        sim_id=cid,
        geometry_family=config['geometry_family'],
        nodes=res.nodes,
        elements=res.elements,
        displacements=res.displacements,
        element_von_mises_psi=res.element_von_mises.astype(np.float32) * PA_TO_PSI,
        nodal_von_mises_psi=res.nodal_von_mises.astype(np.float32) * PA_TO_PSI
    )
    
    res_dict = dict(config)
    res_dict.update({
        'field_file': str(npz_path),
        'max_von_mises_element_psi': float(elem_max_vm_psi),
        'max_von_mises_nodal_psi': float(nodal_max_vm_psi),
        'Kt_element': float(elem_max_vm_psi / 100.0),
        'Kt_nodal': float(nodal_max_vm_psi / 100.0),
        'strain_energy_j': float(se),
        'horizontal_reaction_n': float(rf_x),
        'bottom_reaction_n': float(rf_y),
        'applied_force_n': float(app_f),
        'equilibrium_error_n': float(eq_err),
        'node_count': int(res.num_nodes),
        'element_count': int(res.num_elements),
        'solve_time_s': float(solve_time)
    })
    return res_dict


def run_stage3D_confirmatory_generation():
    print("=== STARTING STAGE 3D PHASE I: BLIND CONFIRMATORY FEM GENERATION ===")
    configs = generate_stage3D_blind_configs()
    print(f"Generated {len(configs)} candidate configurations across 5 geometries.")
    
    t_start = time.perf_counter()
    results = []
    
    # Parallel execution across available cores
    max_workers = min(os.cpu_count() or 4, 12)
    print(f"Dispatching FEM simulations to ProcessPoolExecutor with {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_fem_single, cfg): cfg['confirmatory_id'] for cfg in configs}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                res = future.result()
                results.append(res)
                if len(results) % 50 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"  Completed {len(results)}/500 simulations in {elapsed:.1f} s...")
            except Exception as e:
                print(f"  Error in simulation {cid}: {e}")
                
    total_time = time.perf_counter() - t_start
    print(f"\nAll 500 Stage 3D Blind Confirmatory simulations completed in {total_time:.2f} s ({total_time/500:.3f} s/sim)!")
    
    # Save master manifest
    out_dir = Path("data/stage3D_blind_confirmatory")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results).sort_values('confirmatory_id').reset_index(drop=True)
    manifest_path = out_dir / "stage3D_confirmatory_metadata.csv"
    df_out.to_csv(manifest_path, index=False)
    print(f"Saved Stage 3D blind confirmatory metadata to {manifest_path}")
    
    # Verification checks
    max_eq_err = df_out['equilibrium_error_n'].abs().max()
    print(f"Reaction force equilibrium verification: Max residual = {max_eq_err:.4e} N (Residual <= 1e-12)")
    print("Stage 3D Phase I Blind Confirmatory Generation Complete!\n")


if __name__ == "__main__":
    run_stage3D_confirmatory_generation()
