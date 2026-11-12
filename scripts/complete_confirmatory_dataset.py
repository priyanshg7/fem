"""
Script to complete the Stage 3D blind confirmatory dataset to exactly 500 samples (100 per geometry).
"""
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.stage3D_confirmatory_generator import run_fem_single

OUTPUT_DIR = "data/stage3D_blind_confirmatory"
META_PATH = os.path.join(OUTPUT_DIR, "stage3D_confirmatory_metadata.csv")

def main():
    df = pd.read_csv(META_PATH)
    geom_dims = {
        'square_1000x1000': (1000.0, 1000.0),
        'rectangular_1000x900': (1000.0, 900.0),
        'rectangular_1000x800': (1000.0, 800.0),
        'rectangular_1000x700': (1000.0, 700.0),
        'rectangular_1000x600': (1000.0, 600.0)
    }
    counts = df.groupby(['geometry_family', 'sample_type']).size()
    
    new_candidates = []
    np.random.seed(12345)
    
    for g_name, (L, H) in geom_dims.items():
        curr = counts.get((g_name, 'physically_challenging_in_domain'), 0)
        needed = 20 - curr
        if needed <= 0:
            continue
        
        Dc = 200.0
        Xc, Yc = L / 2.0, H / 2.0
        gm = 0.012 * H  # strictly above 0.01 * H
        
        generated = 0
        idx = 800
        while generated < needed:
            idx += 1
            is_ligament = (generated % 2 == 0)
            d_ratio = np.random.uniform(0.08, 0.22)
            d = d_ratio * Dc
            
            if is_ligament:
                theta = np.random.uniform(0, 2 * np.pi)
                c_lig = np.random.uniform(gm + 1.0, gm + 10.0)
                R_c = (Dc + d) / 2.0 + c_lig
                x = Xc + R_c * np.cos(theta)
                y = Yc + R_c * np.sin(theta)
                
                c_left = x - d / 2.0
                c_right = L - x - d / 2.0
                c_bot = y - d / 2.0
                c_top = H - y - d / 2.0
                c_bnd = min(c_left, c_right, c_bot, c_top)
                sub_type = 'ligament_interaction'
                if c_bnd < gm or x - d/2 < 0 or x + d/2 > L or y - d/2 < 0 or y + d/2 > H:
                    continue
            else:
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
                
                dist_center = np.sqrt((x - Xc)**2 + (y - Yc)**2)
                c_lig = dist_center - (Dc + d) / 2.0
                sub_type = 'boundary_interaction'
                if c_lig < gm:
                    continue
            
            new_candidates.append({
                'confirmatory_id': f"STAGE3D_BLIND_{g_name}_{idx:04d}",
                'geometry_family': g_name,
                'sample_type': 'physically_challenging_in_domain',
                'sub_type': sub_type,
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
            generated += 1
            
    print(f"Generated {len(new_candidates)} replacement candidates. Solving with ProcessPoolExecutor...")
    results = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
        futures = {executor.submit(run_fem_single, cfg): cfg['confirmatory_id'] for cfg in new_candidates}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Failed {cid}: {e}")
                
    print(f"Successfully solved {len(results)} cases.")
    df_new = pd.DataFrame(results)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined = df_combined.sort_values(['geometry_family', 'sample_type', 'confirmatory_id']).reset_index(drop=True)
    df_combined.to_csv(META_PATH, index=False)
    print(f"Saved total {len(df_combined)} records to {META_PATH}")
    print(df_combined.groupby(['geometry_family', 'sample_type']).size())

if __name__ == '__main__':
    main()
