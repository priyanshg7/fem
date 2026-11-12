"""
Regenerate the 35 replacement confirmatory samples with CANONICAL Dc = H/12
so that 100% of the 500 Stage 3D confirmatory samples match the canonical geometry definition.
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
    print("Initial central hole distribution in confirmatory set:")
    print(df.groupby('geometry_family')['central_hole_diameter_mm'].value_counts())
    
    # Identify the 35 non-canonical rows
    bad_mask = df['central_hole_diameter_mm'] > 100.0
    df_good = df[~bad_mask].copy()
    bad_rows = df[bad_mask].copy()
    print(f"\nFound {len(bad_rows)} non-canonical rows (Dc=200mm) to regenerate.")
    
    new_candidates = []
    np.random.seed(42)
    
    for _, row in bad_rows.iterrows():
        g_name = row['geometry_family']
        L = float(row['plate_length_mm'])
        H = float(row['plate_height_mm'])
        Dc = H / 12.0  # CANONICAL DEFINITION
        Xc, Yc = L / 2.0, H / 2.0
        gm = 0.012 * H
        
        cid = row['confirmatory_id']
        sub_type = row['sub_type']
        
        valid = False
        attempts = 0
        while not valid and attempts < 1000:
            attempts += 1
            d_ratio = np.random.uniform(0.08, 0.22)
            d = d_ratio * Dc
            
            if 'ligament' in str(sub_type):
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
                if c_bnd < gm or x - d/2 < 0 or x + d/2 > L or y - d/2 < 0 or y + d/2 > H:
                    continue
                valid = True
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
                if c_lig < gm:
                    continue
                valid = True
                
        new_candidates.append({
            'confirmatory_id': cid,
            'geometry_family': g_name,
            'sample_type': row['sample_type'],
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
        
    print(f"Generated {len(new_candidates)} replacement configurations. Solving FEM in parallel...")
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
                
    print(f"Successfully solved {len(results)} replacement simulations.")
    df_new = pd.DataFrame(results)
    df_clean = pd.concat([df_good, df_new], ignore_index=True)
    df_clean = df_clean.sort_values(['geometry_family', 'sample_type', 'confirmatory_id']).reset_index(drop=True)
    df_clean.to_csv(META_PATH, index=False)
    print(f"Updated metadata saved to {META_PATH}. Total rows: {len(df_clean)}")
    print("\nVerified canonical central hole distribution in confirmatory set:")
    print(df_clean.groupby('geometry_family')['central_hole_diameter_mm'].value_counts())

if __name__ == '__main__':
    main()

