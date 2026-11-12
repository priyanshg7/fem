"""
Stage 3E Untouched Confirmatory FEM Benchmark Generator:
Generates an independent, sealed confirmatory evaluation dataset of N = 500 simulations
(exactly 100 per geometry family) with strictly verified canonical geometry:
    Dc = H / 12.0
    Xc = L / 2.0, Yc = H / 2.0
"""
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.fem_solver import solve_plane_stress_fem, PlateGeometryConfig, MaterialConfig

OUTPUT_DIR = Path("data/stage3E_confirmatory")
FIELDS_DIR = OUTPUT_DIR / "spatial_fields"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIELDS_DIR.mkdir(parents=True, exist_ok=True)

PA_TO_PSI = 1.0 / 6894.757

GEOMETRY_FAMILIES = {
    'square_1000x1000': {'L': 1000.0, 'H': 1000.0, 'Dc': 1000.0 / 12.0},
    'rectangular_1000x900': {'L': 1000.0, 'H': 900.0, 'Dc': 900.0 / 12.0},
    'rectangular_1000x800': {'L': 1000.0, 'H': 800.0, 'Dc': 800.0 / 12.0},
    'rectangular_1000x700': {'L': 1000.0, 'H': 700.0, 'Dc': 700.0 / 12.0},
    'rectangular_1000x600': {'L': 1000.0, 'H': 600.0, 'Dc': 600.0 / 12.0},
}

def generate_stage3E_configs():
    np.random.seed(30303) # distinct seed for untouched confirmatory
    all_configs = []
    
    for g_name, geom in GEOMETRY_FAMILIES.items():
        L, H, Dc = geom['L'], geom['H'], geom['Dc']
        Xc, Yc = L / 2.0, H / 2.0
        gm = 0.012 * H
        
        # 1. 60 Space-filling configurations
        space_samples = []
        gid = 1
        while len(space_samples) < 60:
            d_ratio = np.random.uniform(0.08, 0.28)
            d = d_ratio * Dc
            x = np.random.uniform(d/2.0 + gm, L - d/2.0 - gm)
            y = np.random.uniform(d/2.0 + gm, H - d/2.0 - gm)
            c_hole = np.hypot(x - Xc, y - Yc) - (Dc + d) / 2.0
            if c_hole >= gm:
                c_bnd = min(x - d/2.0, L - x - d/2.0, y - d/2.0, H - y - d/2.0)
                space_samples.append({
                    'confirmatory_id': f"STAGE3E_UNTOUCHED_{g_name}_{gid:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'space_filling_in_domain',
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
                gid += 1
                
        # 2. 20 Physically challenging (10 ligament, 10 boundary)
        chall_samples = []
        while len(chall_samples) < 10:
            d = np.random.uniform(0.08, 0.22) * Dc
            theta = np.random.uniform(0, 2 * np.pi)
            c_lig = np.random.uniform(gm + 1.0, gm + 10.0)
            R_c = (Dc + d) / 2.0 + c_lig
            x = Xc + R_c * np.cos(theta)
            y = Yc + R_c * np.sin(theta)
            c_bnd = min(x - d/2.0, L - x - d/2.0, y - d/2.0, H - y - d/2.0)
            if c_bnd >= gm and 0 < x - d/2 < L and 0 < y - d/2 < H:
                chall_samples.append({
                    'confirmatory_id': f"STAGE3E_UNTOUCHED_{g_name}_{gid:04d}",
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
                gid += 1
                
        while len(chall_samples) < 20:
            d = np.random.uniform(0.08, 0.22) * Dc
            edge = np.random.choice(['left', 'right', 'bottom', 'top'])
            c_bnd = np.random.uniform(gm + 1.0, gm + 12.0)
            if edge == 'left':
                x = d / 2.0 + c_bnd
                y = np.random.uniform(d / 2.0 + 20.0, H - d / 2.0 - 20.0)
            elif edge == 'right':
                x = L - d / 2.0 - c_bnd
                y = np.random.uniform(d / 2.0 + 20.0, H - d / 2.0 - 20.0)
            elif edge == 'bottom':
                x = np.random.uniform(d / 2.0 + 20.0, L - d / 2.0 - 20.0)
                y = d / 2.0 + c_bnd
            else:
                x = np.random.uniform(d / 2.0 + 20.0, L - d / 2.0 - 20.0)
                y = H - d / 2.0 - c_bnd
            c_lig = np.hypot(x - Xc, y - Yc) - (Dc + d) / 2.0
            if c_lig >= gm:
                chall_samples.append({
                    'confirmatory_id': f"STAGE3E_UNTOUCHED_{g_name}_{gid:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'physically_challenging_in_domain',
                    'sub_type': 'boundary_interaction',
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
                gid += 1
                
        # 3. 20 Out-of-domain diameter extensions (D/Dc in [0.32, 0.45])
        ood_samples = []
        while len(ood_samples) < 20:
            d_ratio = np.random.uniform(0.32, 0.45)
            d = d_ratio * Dc
            x = np.random.uniform(d/2.0 + gm, L - d/2.0 - gm)
            y = np.random.uniform(d/2.0 + gm, H - d/2.0 - gm)
            c_hole = np.hypot(x - Xc, y - Yc) - (Dc + d) / 2.0
            if c_hole >= gm:
                c_bnd = min(x - d/2.0, L - x - d/2.0, y - d/2.0, H - y - d/2.0)
                ood_samples.append({
                    'confirmatory_id': f"STAGE3E_UNTOUCHED_{g_name}_{gid:04d}",
                    'geometry_family': g_name,
                    'sample_type': 'out_of_domain_diameter_extension',
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
                gid += 1
                
        all_configs.extend(space_samples)
        all_configs.extend(chall_samples)
        all_configs.extend(ood_samples)
        
    return all_configs

def solve_one(config):
    L = float(config['plate_length_mm']) / 1000.0
    H = float(config['plate_height_mm']) / 1000.0
    Dc = float(config['central_hole_diameter_mm']) / 1000.0
    d = float(config['defect_diameter_mm']) / 1000.0
    x = float(config['defect_center_x_mm']) / 1000.0
    y = float(config['defect_center_y_mm']) / 1000.0
    gm = 0.012 * H
    
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
        h_coarse=0.020,
        h_center_hole=0.0035,
        h_defect_hole=float(d / 10.0),
        clearance_margin=gm
    )
    mat_cfg = MaterialConfig(applied_traction_psi=100.0)
    
    t0 = time.perf_counter()
    res = solve_plane_stress_fem(geom_cfg, mat_cfg, verbose=False)
    solve_time = time.perf_counter() - t0
    
    cid = config['confirmatory_id']
    npz_path = FIELDS_DIR / f"{cid}_fields.npz"
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
    
    out = dict(config)
    out.update({
        'field_file': str(npz_path),
        'max_von_mises_element_psi': float(res.max_element_von_mises_psi),
        'max_von_mises_nodal_psi': float(res.max_von_mises_psi),
        'Kt_element': float(res.max_element_von_mises_psi / 100.0),
        'Kt_nodal': float(res.max_von_mises_psi / 100.0),
        'strain_energy_j': float(res.strain_energy),
        'node_count': int(res.num_nodes),
        'element_count': int(res.num_elements),
        'solve_time_s': float(solve_time)
    })
    return out

def run_generation():
    configs = generate_stage3E_configs()
    print(f"Generated {len(configs)} candidate configs for Stage 3E Untouched Confirmatory benchmark.")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        futures = {executor.submit(solve_one, cfg): cfg['confirmatory_id'] for cfg in configs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if len(results) % 50 == 0:
                print(f"  Solved {len(results)}/500 in {time.time() - t0:.1f} s...")
                
    df = pd.DataFrame(results).sort_values(['geometry_family', 'sample_type', 'confirmatory_id']).reset_index(drop=True)
    manifest = OUTPUT_DIR / "stage3E_untouched_confirmatory_metadata.csv"
    df.to_csv(manifest, index=False)
    print(f"Successfully generated 500 Stage 3E untouched confirmatory simulations in {time.time() - t0:.1f} s!")
    print(f"Saved manifest to {manifest}")

if __name__ == '__main__':
    run_generation()

