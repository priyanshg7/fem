"""
Stage 2V Phase C - Adaptive Candidate Sampling and Selection
Generates >= 20,000 valid candidate configurations per geometry family using
scrambled space-filling sampling, scores candidates via 3 acquisition strategies,
and selects 250 high-value candidates per geometry (1,250 total).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from scipy.stats import qmc

def generate_valid_candidate_pool(
    geometry_family: str,
    target_valid: int = 25000,
    seed: int = 1001
) -> pd.DataFrame:
    """Generate a dense pool of valid unsimulated candidates using Scrambled Sobol sampling."""
    
    # Geometry parameters
    geom_params = {
        'rectangular_1000x600': {'L': 1000.0, 'H': 600.0, 'Dc': 50.0, 'g': 6.0, 'd_min': 3.75, 'd_max': 15.0},
        'rectangular_1000x700': {'L': 1000.0, 'H': 700.0, 'Dc': 58.3333, 'g': 7.0, 'd_min': 4.375, 'd_max': 17.5},
        'rectangular_1000x800': {'L': 1000.0, 'H': 800.0, 'Dc': 66.6667, 'g': 8.0, 'd_min': 5.0, 'd_max': 20.0},
        'rectangular_1000x900': {'L': 1000.0, 'H': 900.0, 'Dc': 75.0, 'g': 9.0, 'd_min': 5.625, 'd_max': 22.5},
        'square_1000x1000':     {'L': 1000.0, 'H': 1000.0, 'Dc': 83.3333, 'g': 10.0, 'd_min': 6.25, 'd_max': 25.0},
    }
    
    p = geom_params[geometry_family]
    L, H, Dc, g = p['L'], p['H'], p['Dc'], p['g']
    d_min, d_max = p['d_min'], p['d_max']
    xc, yc = L / 2.0, H / 2.0
    
    # Read existing simulations to avoid duplicates
    df_uni = pd.read_csv('data/unified_dataset_50375.csv')
    sub_existing = df_uni[df_uni['geometry_family'] == geometry_family]
    existing_coords = sub_existing[['defect_center_x_mm', 'defect_center_y_mm', 'defect_diameter_mm']].values
    existing_tree = cKDTree(existing_coords)
    
    # Scrambled Sobol sampling in 3D: [x_norm, y_norm, d_norm]
    # Draw in chunks until >= target_valid
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    
    valid_candidates = []
    total_sampled = 0
    total_rejected = 0
    
    chunk_size = 32768
    while len(valid_candidates) < target_valid:
        sample = sampler.random(n=chunk_size)
        total_sampled += chunk_size
        
        # Map to physical dimensions
        # x in [g + d/2, L - g - d/2], y in [g + d/2, H - g - d/2], d in [d_min, d_max]
        d_cand = d_min + sample[:, 2] * (d_max - d_min)
        x_cand = (g + d_cand / 2.0) + sample[:, 0] * (L - 2.0 * g - d_cand)
        y_cand = (g + d_cand / 2.0) + sample[:, 1] * (H - 2.0 * g - d_cand)
        
        # Check boundary clearances
        c1 = (x_cand - d_cand / 2.0 >= g) & (x_cand + d_cand / 2.0 <= L - g)
        c2 = (y_cand - d_cand / 2.0 >= g) & (y_cand + d_cand / 2.0 <= H - g)
        
        # Check central hole clearance: dist >= (Dc + d)/2 + g
        dist_c = np.hypot(x_cand - xc, y_cand - yc)
        c3 = dist_c >= ((Dc + d_cand) / 2.0 + g)
        
        valid_mask = c1 & c2 & c3
        n_valid_chunk = np.sum(valid_mask)
        total_rejected += (chunk_size - n_valid_chunk)
        
        if n_valid_chunk > 0:
            v_x = x_cand[valid_mask]
            v_y = y_cand[valid_mask]
            v_d = d_cand[valid_mask]
            
            chunk_pts = np.column_stack([v_x, v_y, v_d])
            # Check for duplicates against existing simulations (dist > 0.5 mm)
            dists, _ = existing_tree.query(chunk_pts, k=1)
            not_dup_mask = dists > 0.5
            
            for pt in chunk_pts[not_dup_mask]:
                valid_candidates.append(pt)
                if len(valid_candidates) >= target_valid:
                    break
                    
    cand_arr = np.array(valid_candidates[:target_valid])
    df_cand = pd.DataFrame({
        'geometry_family': geometry_family,
        'plate_length_mm': L,
        'plate_height_mm': H,
        'central_hole_diameter_mm': Dc,
        'defect_center_x_mm': cand_arr[:, 0],
        'defect_center_y_mm': cand_arr[:, 1],
        'defect_diameter_mm': cand_arr[:, 2],
        'aspect_ratio': L / H,
        'central_hole_d_over_L': Dc / L,
        'defect_x_over_L': cand_arr[:, 0] / L,
        'defect_y_over_H': cand_arr[:, 1] / H,
        'defect_d_over_Dc': cand_arr[:, 2] / Dc,
        'defect_d_over_H': cand_arr[:, 2] / H,
    })
    
    # Distance metrics
    dist_c = np.hypot(df_cand['defect_center_x_mm'] - xc, df_cand['defect_center_y_mm'] - yc)
    df_cand['hole_to_hole_clearance_mm'] = dist_c - (Dc + df_cand['defect_diameter_mm']) / 2.0
    df_cand['boundary_clearance_mm'] = np.minimum(
        np.minimum(df_cand['defect_center_x_mm'] - df_cand['defect_diameter_mm'] / 2.0, L - (df_cand['defect_center_x_mm'] + df_cand['defect_diameter_mm'] / 2.0)),
        np.minimum(df_cand['defect_center_y_mm'] - df_cand['defect_diameter_mm'] / 2.0, H - (df_cand['defect_center_y_mm'] + df_cand['defect_diameter_mm'] / 2.0))
    )
    
    print(f"[{geometry_family}] Pool generated: {len(df_cand)} valid candidates (sampled {total_sampled}, rejected {total_rejected})")
    return df_cand

def select_adaptive_batch(
    df_cand: pd.DataFrame,
    val_preds_df: pd.DataFrame,
    n_total: int = 250,
    n_strategy_a: int = 150,
    n_strategy_b: int = 60,
    n_strategy_c: int = 40
) -> pd.DataFrame:
    """Score and select 250 adaptive candidates combining Strategies A, B, and C."""
    
    g_name = df_cand['geometry_family'].iloc[0]
    
    # Load preliminary models
    et = joblib.load('models/stage2V_preliminary/ExtraTrees.joblib')
    rf = joblib.load('models/stage2V_preliminary/RandomForest.joblib')
    hgb = joblib.load('models/stage2V_preliminary/HistGradientBoosting.joblib')
    mlp = joblib.load('models/stage2V_preliminary/MLP_NeuralNet.joblib')
    scaler = joblib.load('models/stage2V_preliminary/scaler.joblib')
    
    features = [
        'defect_x_over_L',
        'defect_y_over_H',
        'defect_d_over_Dc',
        'aspect_ratio',
        'central_hole_d_over_L'
    ]
    X_cand = df_cand[features].values
    X_cand_scaled = scaler.transform(X_cand)
    
    # Predict with all 4 models
    pred_et = et.predict(X_cand)
    pred_rf = rf.predict(X_cand)
    pred_hgb = hgb.predict(X_cand)
    pred_mlp = mlp.predict(X_cand_scaled)
    
    preds_stack = np.column_stack([pred_et, pred_rf, pred_hgb, pred_mlp])
    mean_preds = np.mean(preds_stack, axis=1)
    std_disagree = np.std(preds_stack, axis=1)
    
    df_cand['model_mean_pred_Kt'] = mean_preds
    df_cand['model_disagreement_sigma'] = std_disagree
    
    # --- Strategy A: Model Disagreement & Uncertainty (150 candidates) ---
    # Rank candidates by disagreement sigma, with spatial thinning
    cand_pool = df_cand.copy()
    cand_pool['strategy_a_score'] = std_disagree
    
    # Greedy selection with minimum distance thinning to avoid spatial clustering
    sorted_a = cand_pool.sort_values(by='strategy_a_score', ascending=False)
    selected_a_idx = []
    coords_selected = []
    
    for idx, row in sorted_a.iterrows():
        pt = np.array([row['defect_x_over_L'], row['defect_y_over_H'], row['defect_d_over_Dc']])
        if len(coords_selected) == 0:
            selected_a_idx.append(idx)
            coords_selected.append(pt)
        else:
            # check distance in normalized space
            dists = np.linalg.norm(np.array(coords_selected) - pt, axis=1)
            if np.min(dists) > 0.025:  # thinning threshold
                selected_a_idx.append(idx)
                coords_selected.append(pt)
        if len(selected_a_idx) >= n_strategy_a:
            break
            
    # --- Strategy B: High-Validation-Error Neighbourhoods (60 candidates) ---
    # Identify high-error points from validation set
    sub_val = val_preds_df[val_preds_df['geometry_family'] == g_name]
    top_val_errors = sub_val.sort_values(by='committee_abs_error', ascending=False).head(150)
    val_pts = top_val_errors[['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc']].values
    val_tree = cKDTree(val_pts)
    
    # Remaining candidates not in selected_a
    remaining_mask = ~cand_pool.index.isin(selected_a_idx)
    df_rem = cand_pool[remaining_mask].copy()
    rem_pts = df_rem[['defect_x_over_L', 'defect_y_over_H', 'defect_d_over_Dc']].values
    
    dists_to_error, _ = val_tree.query(rem_pts, k=1)
    df_rem['dist_to_high_error'] = dists_to_error
    
    # Nearest unsimulated candidates to high-error validation points
    sorted_b = df_rem.sort_values(by='dist_to_high_error', ascending=True)
    selected_b_idx = []
    coords_b = list(coords_selected)
    
    for idx, row in sorted_b.iterrows():
        pt = np.array([row['defect_x_over_L'], row['defect_y_over_H'], row['defect_d_over_Dc']])
        dists = np.linalg.norm(np.array(coords_b) - pt, axis=1)
        if np.min(dists) > 0.020:
            selected_b_idx.append(idx)
            coords_b.append(pt)
        if len(selected_b_idx) >= n_strategy_b:
            break
            
    # --- Strategy C: Coverage & Physical-Risk Diversity (40 candidates) ---
    # Targets: thin ligament, near shoulder, near boundary, extreme D
    remaining_mask_c = ~(cand_pool.index.isin(selected_a_idx) | cand_pool.index.isin(selected_b_idx))
    df_rem_c = cand_pool[remaining_mask_c].copy()
    
    # Create physical risk score
    # High risk when:
    # 1. defect is near central hole shoulder (Y ~ H/2)
    # 2. hole-to-hole clearance is small
    # 3. defect diameter is large (D/Dc -> 0.30)
    # 4. boundary clearance is small
    shoulder_dist = np.abs(df_rem_c['defect_y_over_H'] - 0.50)
    risk_score = (
        (1.0 / (df_rem_c['hole_to_hole_clearance_mm'] + 1.0)) * 20.0 +
        (1.0 / (shoulder_dist + 0.05)) * 5.0 +
        (df_rem_c['defect_d_over_Dc'] / 0.30) * 10.0 +
        (1.0 / (df_rem_c['boundary_clearance_mm'] + 2.0)) * 10.0
    )
    df_rem_c['physical_risk_score'] = risk_score
    sorted_c = df_rem_c.sort_values(by='physical_risk_score', ascending=False)
    
    selected_c_idx = []
    coords_all = list(coords_b)
    for idx, row in sorted_c.iterrows():
        pt = np.array([row['defect_x_over_L'], row['defect_y_over_H'], row['defect_d_over_Dc']])
        dists = np.linalg.norm(np.array(coords_all) - pt, axis=1)
        if np.min(dists) > 0.020:
            selected_c_idx.append(idx)
            coords_all.append(pt)
        if len(selected_c_idx) >= n_strategy_c:
            break
            
    # Combine selected candidates
    df_a = cand_pool.loc[selected_a_idx].copy()
    df_a['acquisition_strategy'] = 'Strategy_A_Model_Disagreement'
    
    df_b = cand_pool.loc[selected_b_idx].copy()
    df_b['acquisition_strategy'] = 'Strategy_B_High_Error_Neighborhood'
    
    df_c = cand_pool.loc[selected_c_idx].copy()
    df_c['acquisition_strategy'] = 'Strategy_C_Physical_Risk_Coverage'
    
    batch = pd.concat([df_a, df_b, df_c], ignore_index=True)
    batch['adaptive_simulation_id'] = [f"adapt_{g_name}_{i:04d}" for i in range(len(batch))]
    batch['acquisition_rank'] = range(1, len(batch) + 1)
    
    print(f"[{g_name}] Selected adaptive batch of {len(batch)} candidates:")
    print(f"  Strategy A (Disagreement): {len(df_a)}")
    print(f"  Strategy B (Error Nbrhd): {len(df_b)}")
    print(f"  Strategy C (Physical Risk): {len(df_c)}")
    
    return batch

def run_adaptive_selection():
    print("Starting Stage 2V Phase C Adaptive Candidate Selection...")
    val_preds_df = pd.read_csv('data/stage2V_preliminary_val_predictions.csv')
    
    geometries = [
        'rectangular_1000x600',
        'rectangular_1000x700',
        'rectangular_1000x800',
        'rectangular_1000x900',
        'square_1000x1000'
    ]
    
    all_batches = []
    os.makedirs('data/stage2V_adaptive_selection', exist_ok=True)
    
    selection_summary = {}
    for g in geometries:
        # 1. Generate 25,000 valid candidates per geometry
        df_cand = generate_valid_candidate_pool(g, target_valid=25000, seed=1001 + len(all_batches)*100)
        
        # 2. Select 250 adaptive candidates combining Strategies A (150), B (60), C (40)
        batch = select_adaptive_batch(
            df_cand, 
            val_preds_df, 
            n_total=250, 
            n_strategy_a=150, 
            n_strategy_b=60, 
            n_strategy_c=40
        )
        
        batch_path = f'data/stage2V_adaptive_selection/candidates_{g}.csv'
        batch.to_csv(batch_path, index=False)
        all_batches.append(batch)
        
        selection_summary[g] = {
            'candidate_pool_size': len(df_cand),
            'selected_total': len(batch),
            'strategy_a_count': int((batch['acquisition_strategy'] == 'Strategy_A_Model_Disagreement').sum()),
            'strategy_b_count': int((batch['acquisition_strategy'] == 'Strategy_B_High_Error_Neighborhood').sum()),
            'strategy_c_count': int((batch['acquisition_strategy'] == 'Strategy_C_Physical_Risk_Coverage').sum()),
            'mean_disagreement': float(batch['model_disagreement_sigma'].mean()),
            'max_disagreement': float(batch['model_disagreement_sigma'].max())
        }
        
    master_adaptive = pd.concat(all_batches, ignore_index=True)
    master_adaptive.to_csv('data/stage2V_adaptive_selection/master_adaptive_candidates_1250.csv', index=False)
    print(f"\nSuccessfully selected master adaptive batch of {len(master_adaptive)} candidates.")
    
    with open('output/stage2V_adaptive_selection_summary.json', 'w') as f:
        json.dump(selection_summary, f, indent=2)
    print("Saved output/stage2V_adaptive_selection_summary.json")

if __name__ == '__main__':
    run_adaptive_selection()

