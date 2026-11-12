"""
Stage 3D Feature Extraction Engine:
Extracts Model D1 (Regional Peak and Statistical Stress Features) and
Model D2 (Fixed Spatial Sensor Array Observations) from raw FEM field solutions.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_sensor_locations():
    # 4 sensors (4 corners / far-field)
    s4 = np.array([
        [0.10, 0.10],
        [0.90, 0.10],
        [0.10, 0.90],
        [0.90, 0.90]
    ], dtype=np.float32)
    
    # 8 sensors (4 corners + 4 edge midpoints)
    s8 = np.vstack([
        s4,
        np.array([
            [0.50, 0.08],
            [0.50, 0.92],
            [0.08, 0.50],
            [0.92, 0.50]
        ], dtype=np.float32)
    ])
    
    # 16 sensors: regular 4x4 grid excluding center hole zone (radius < 0.15)
    gx4 = np.linspace(0.12, 0.88, 4)
    gy4 = np.linspace(0.12, 0.88, 4)
    s16_list = []
    for x in gx4:
        for y in gy4:
            if np.hypot(x - 0.5, y - 0.5) > 0.13:
                s16_list.append([x, y])
            else:
                s16_list.append([x + 0.08 * np.sign(x - 0.5), y + 0.08 * np.sign(y - 0.5)])
    s16 = np.array(s16_list[:16], dtype=np.float32)
    
    # 32 sensors: 6x6 grid excluding center hole
    gx6 = np.linspace(0.08, 0.92, 6)
    gy6 = np.linspace(0.08, 0.92, 6)
    s32_list = []
    for x in gx6:
        for y in gy6:
            if np.hypot(x - 0.5, y - 0.5) > 0.14:
                s32_list.append([x, y])
    s32 = np.array(s32_list[:32], dtype=np.float32)
    
    # 64 sensors: 9x9 grid excluding center hole
    gx9 = np.linspace(0.06, 0.94, 9)
    gy9 = np.linspace(0.06, 0.94, 9)
    s64_list = []
    for x in gx9:
        for y in gy9:
            if np.hypot(x - 0.5, y - 0.5) > 0.13:
                s64_list.append([x, y])
    s64 = np.array(s64_list[:64], dtype=np.float32)
    
    return {4: s4, 8: s8, 16: s16, 32: s32, 64: s64}

SENSOR_LAYOUTS = get_sensor_locations()

def load_coords_and_stress(npz_path):
    data = np.load(npz_path)
    if 'element_centroids_mm' in data:
        coords_mm = data['element_centroids_mm']
        vm_stress = data['element_von_mises_psi']
    elif 'element_centroids' in data:
        coords = data['element_centroids']
        coords_mm = coords if np.max(coords[:, 0]) > 2.0 else coords * 1000.0
        vm_stress = data['element_von_mises_psi']
    elif 'nodes' in data and 'elements' in data:
        nodes = data['nodes']
        elements = data['elements']
        coords = nodes[elements].mean(axis=1)
        coords_mm = coords if np.max(coords[:, 0]) > 2.0 else coords * 1000.0
        vm_stress = data['element_von_mises_psi']
    else:
        raise ValueError('Unknown schema')
    return coords_mm, vm_stress

def extract_features_single(row_dict):
    field_file = row_dict['field_file']
    if not os.path.exists(field_file):
        return None
        
    try:
        coords_mm, vm_stress = load_coords_and_stress(field_file)
        L = float(row_dict['plate_length_mm'])
        H = float(row_dict['plate_height_mm'])
        
        cx, cy = L / 2.0, H / 2.0
        x = coords_mm[:, 0]
        y = coords_mm[:, 1]
        
        # Quadrants relative to plate center (L/2, H/2)
        m_q1 = (x >= cx) & (y >= cy)
        m_q2 = (x < cx) & (y >= cy)
        m_q3 = (x < cx) & (y < cy)
        m_q4 = (x >= cx) & (y < cy)
        
        # Central hole ring: radius in [100 mm, 125 mm]
        r_dist = np.hypot(x - cx, y - cy)
        m_ring = (r_dist >= 100.0) & (r_dist <= 125.0)
        
        def calc_stats(m):
            if np.any(m):
                s = vm_stress[m]
                return float(np.max(s)), float(np.mean(s)), float(np.percentile(s, 90))
            return 0.0, 0.0, 0.0
            
        max_q1, mean_q1, p90_q1 = calc_stats(m_q1)
        max_q2, mean_q2, p90_q2 = calc_stats(m_q2)
        max_q3, mean_q3, p90_q3 = calc_stats(m_q3)
        max_q4, mean_q4, p90_q4 = calc_stats(m_q4)
        max_ring, mean_ring, p90_ring = calc_stats(m_ring)
        
        overall_max = float(np.max(vm_stress))
        
        # Sensor queries in normalized coordinates [0, 1]
        norm_coords = np.column_stack([x / L, y / H])
        tree = cKDTree(norm_coords)
        
        s64_coords = SENSOR_LAYOUTS[64]
        dists, idxs = tree.query(s64_coords)
        sensor_stress_64 = vm_stress[idxs].astype(np.float32)
        
        out_rec = {
            'sim_id': row_dict.get('enriched_id', row_dict.get('confirmatory_id', '')),
            'geometry_family': row_dict['geometry_family'],
            'plate_length_mm': L,
            'plate_height_mm': H,
            'aspect_ratio': L / H,
            'defect_center_x_mm': float(row_dict['defect_center_x_mm']),
            'defect_center_y_mm': float(row_dict['defect_center_y_mm']),
            'defect_diameter_mm': float(row_dict['defect_diameter_mm']),
            'defect_x_over_L': float(row_dict['defect_x_over_L']),
            'defect_y_over_H': float(row_dict['defect_y_over_H']),
            'defect_d_over_Dc': float(row_dict['defect_d_over_Dc']),
            'split': row_dict.get('split', 'confirmatory'),
            # D1 Regional Features (Kt = stress / 100 psi)
            'D1_Kt_max': overall_max / 100.0,
            'D1_Kt_Q1': max_q1 / 100.0,
            'D1_Kt_Q2': max_q2 / 100.0,
            'D1_Kt_Q3': max_q3 / 100.0,
            'D1_Kt_Q4': max_q4 / 100.0,
            'D1_mean_Q1': mean_q1 / 100.0,
            'D1_mean_Q2': mean_q2 / 100.0,
            'D1_mean_Q3': mean_q3 / 100.0,
            'D1_mean_Q4': mean_q4 / 100.0,
            'D1_p90_Q1': p90_q1 / 100.0,
            'D1_p90_Q2': p90_q2 / 100.0,
            'D1_p90_Q3': p90_q3 / 100.0,
            'D1_p90_Q4': p90_q4 / 100.0,
            'D1_Kt_ring': max_ring / 100.0,
            'D1_mean_ring': mean_ring / 100.0,
            'D1_p90_ring': p90_ring / 100.0,
        }
        
        for i in range(64):
            out_rec[f'D2_sensor_{i:02d}'] = float(sensor_stress_64[i] / 100.0)
            
        return out_rec
    except Exception as e:
        print(f"Error processing {field_file}: {e}")
        return None

def process_batch(rows):
    res = []
    for r in rows:
        feat = extract_features_single(r)
        if feat is not None:
            res.append(feat)
    return res

def run_extraction():
    print("=== EXTRACTING STAGE 3D D1 AND D2 INVERSE FEATURES ===")
    
    # 1. Confirmatory Samples
    conf_path = "data/stage3D_blind_confirmatory/stage3D_confirmatory_metadata.csv"
    if os.path.exists(conf_path):
        print(f"Reading {conf_path}...")
        df_conf = pd.read_csv(conf_path)
        rows_conf = df_conf.to_dict('records')
        print(f"Extracting features for {len(rows_conf)} confirmatory samples...")
        res_conf = []
        for r in rows_conf:
            feat = extract_features_single(r)
            if feat:
                res_conf.append(feat)
        df_conf_feat = pd.DataFrame(res_conf)
        out_conf = "data/stage3D_blind_confirmatory/stage3D_confirmatory_features.parquet"
        df_conf_feat.to_parquet(out_conf, index=False)
        print(f"Saved {len(df_conf_feat)} confirmatory feature records to {out_conf}")
        
    # 2. Enriched Master Dataset
    master_path = "data/enriched_dataset_51625.csv"
    print(f"Reading {master_path}...")
    df_master = pd.read_csv(master_path)
    rows_master = df_master.to_dict('records')
    total = len(rows_master)
    print(f"Extracting features for {total} master dataset samples using multiprocessing...")
    
    chunk_size = 1000
    chunks = [rows_master[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
    n_workers = min(os.cpu_count() or 4, 12)
    all_results = []
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_batch, chunk): idx for idx, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            c_res = future.result()
            all_results.extend(c_res)
            if len(all_results) % 5000 < chunk_size:
                elapsed = time.time() - t0
                print(f"  Processed {len(all_results)}/{total} in {elapsed:.1f} s ({len(all_results)/elapsed:.0f} samples/s)...")
                
    df_feat = pd.DataFrame(all_results)
    out_master = "data/stage3D_features_master.parquet"
    df_feat.to_parquet(out_master, index=False)
    print(f"Saved {len(df_feat)} master feature records to {out_master} in {time.time() - t0:.1f} s!")

if __name__ == '__main__':
    run_extraction()
