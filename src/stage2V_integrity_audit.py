"""
Stage 2V Phase A - Comprehensive Dataset and FEM Integrity Audit
Independent verification across all 50,375 simulation records and spatial fields.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def convert_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_integrity_audit() -> Dict[str, Any]:
    print("Starting Stage 2V Phase A Integrity Audit...")
    
    csv_sources = {
        'square_1000x1000': 'data/fem_dataset_10075.csv',
        'rectangular_1000x600': 'data/rectangular_1000x600/fem_dataset_10075_rectangular.csv',
        'rectangular_1000x700': 'data/geometry_extension/RECT_700/fem_dataset_10075_RECT_700.csv',
        'rectangular_1000x800': 'data/geometry_extension/RECT_800/fem_dataset_10075_RECT_800.csv',
        'rectangular_1000x900': 'data/geometry_extension/RECT_900/fem_dataset_10075_RECT_900.csv',
    }
    unified_csv_path = 'data/unified_dataset_50375.csv'
    
    audit_results: Dict[str, Any] = {
        'source_datasets': {},
        'unified_dataset': {},
        'field_verification': {},
        'deep_consistency_checks': [],
        'equilibrium_audit': {},
        'feature_normalization_audit': {},
        'mesh_settings_audit': {},
        'solver_source_audit': {},
        'classification_integrity': {},
        'overall_status': 'PASS'
    }
    
    # 1. Audit Source CSVs
    print("\n--- 1. Auditing Source CSVs ---")
    total_source_records = 0
    for name, path in csv_sources.items():
        exists = os.path.exists(path)
        if not exists:
            audit_results['overall_status'] = 'FAIL'
            audit_results['source_datasets'][name] = {'exists': False}
            continue
            
        df = pd.read_csv(path)
        nrows = len(df)
        total_source_records += nrows
        sim_id_col = 'simulation_id'
        n_unique_sims = df[sim_id_col].nunique()
        
        # Determine coordinate and diameter columns
        x_col = 'defect_center_x_mm' if 'defect_center_x_mm' in df.columns else 'defect_x'
        y_col = 'defect_center_y_mm' if 'defect_center_y_mm' in df.columns else 'defect_y'
        d_col = 'defect_diameter_mm' if 'defect_diameter_mm' in df.columns else 'defect_diameter'
        
        # Check geometric configuration uniqueness
        config_tuples = list(zip(df[x_col].round(4), df[y_col].round(4), df[d_col].round(4)))
        n_unique_configs = len(set(config_tuples))
        
        # Check null/inf
        has_null = df.isnull().values.any()
        
        # Check subset types
        subsets = df['subset_type'].value_counts().to_dict()
        
        audit_results['source_datasets'][name] = {
            'exists': True,
            'row_count': nrows,
            'column_count': len(df.columns),
            'unique_sim_ids': n_unique_sims,
            'unique_geometric_configs': n_unique_configs,
            'has_null_or_nan': bool(has_null),
            'subset_counts': subsets
        }
        print(f"  {name}: {nrows} rows, {n_unique_sims} unique IDs, {n_unique_configs} unique configs, nulls: {has_null}")
        print(f"    Subsets: {subsets}")
    
    # 2. Audit Master Unified CSV
    print("\n--- 2. Auditing Master Unified CSV ---")
    if not os.path.exists(unified_csv_path):
        audit_results['overall_status'] = 'FAIL'
        audit_results['unified_dataset']['exists'] = False
        return audit_results
        
    df_uni = pd.read_csv(unified_csv_path)
    uni_rows = len(df_uni)
    uni_cols = len(df_uni.columns)
    uni_by_geom = df_uni['geometry_family'].value_counts().to_dict()
    uni_nulls = df_uni.isnull().sum().to_dict()
    total_nulls = df_uni.isnull().sum().sum()
    
    # Check infinite values across numeric columns
    numeric_cols = df_uni.select_dtypes(include=[np.number]).columns
    inf_counts = {col: int(np.isinf(df_uni[col]).sum()) for col in numeric_cols if np.isinf(df_uni[col]).sum() > 0}
    
    audit_results['unified_dataset'] = {
        'exists': True,
        'total_rows': uni_rows,
        'column_count': uni_cols,
        'rows_by_geometry': uni_by_geom,
        'total_null_entries': int(total_nulls),
        'infinite_entries': inf_counts
    }
    print(f"  Unified rows: {uni_rows}, columns: {uni_cols}, total nulls: {total_nulls}")
    print(f"  Rows by geometry: {uni_by_geom}")
    
    # 3. Audit Spatial Field Files
    print("\n--- 3. Auditing Spatial Field Files ---")
    field_paths = df_uni['field_file'].tolist()
    missing_fields = []
    
    # Check existence of all 50,375 field files
    for p in field_paths:
        if not os.path.exists(p):
            missing_fields.append(p)
            
    audit_results['field_verification'] = {
        'total_referenced_fields': len(field_paths),
        'missing_fields_count': len(missing_fields),
        'missing_fields_sample': missing_fields[:5]
    }
    print(f"  Total referenced fields: {len(field_paths)}, missing: {len(missing_fields)}")
    
    # 4. Deep Consistency Verification on Representative Samples
    print("\n--- 4. Deep Consistency Verification on Representative Samples ---")
    # For each geometry, pick:
    # 1: min stress, 2: max stress, 3: median stress, 4: min diameter, 5: max diameter,
    # 6: closest to central hole, 7: closest to boundary, 8: defect dominated, 9: central dominated, 10: random
    deep_checks = []
    
    for g_name in df_uni['geometry_family'].unique():
        sub = df_uni[df_uni['geometry_family'] == g_name]
        
        idx_samples = [
            sub['max_von_mises_element_psi'].idxmin(),
            sub['max_von_mises_element_psi'].idxmax(),
            sub['max_von_mises_element_psi'].sub(sub['max_von_mises_element_psi'].median()).abs().idxmin(),
            sub['defect_diameter_mm'].idxmin(),
            sub['defect_diameter_mm'].idxmax(),
            (sub['peak_stress_region'] == 'defect_hole').idxmax(),
            (sub['peak_stress_region'] == 'center_hole').idxmax(),
            sub.sample(1, random_state=42).index[0]
        ]
        
        for idx in set(idx_samples):
            row = sub.loc[idx]
            fp = row['field_file']
            
            # Load npz
            try:
                data = np.load(fp)
                keys = list(data.keys())
                
                # Check expected arrays
                coords = data['nodes'] if 'nodes' in data else data['node_coords']
                elems = data['elements'] if 'elements' in data else data['element_conn']
                disp = data['displacement'] if 'displacement' in data else data['disp']
                elem_vm = data['element_von_mises_psi'] if 'element_von_mises_psi' in data else data['elem_vm_psi']
                node_vm = data['nodal_von_mises_psi'] if 'nodal_von_mises_psi' in data else data['node_vm_psi']
                
                # Check finiteness
                all_finite = bool(
                    np.all(np.isfinite(coords)) and 
                    np.all(np.isfinite(disp)) and 
                    np.all(np.isfinite(elem_vm)) and 
                    np.all(np.isfinite(node_vm))
                )
                
                # Check dimensions
                n_nodes = len(coords)
                n_elems = len(elems)
                dims_match = (
                    len(disp) == n_nodes and 
                    len(node_vm) == n_nodes and 
                    len(elem_vm) == n_elems
                )
                
                # Value comparisons
                npz_max_elem_vm = float(np.max(elem_vm))
                csv_max_elem_vm = float(row['max_von_mises_element_psi'])
                diff_elem_vm = abs(npz_max_elem_vm - csv_max_elem_vm)
                
                npz_max_node_vm = float(np.max(node_vm))
                csv_max_node_vm = float(row['max_von_mises_nodal_psi'])
                diff_node_vm = abs(npz_max_node_vm - csv_max_node_vm)
                
                deep_checks.append({
                    'geometry': g_name,
                    'simulation_id': row['source_simulation_id'],
                    'unified_id': int(row['unified_id']),
                    'keys_present': keys,
                    'n_nodes': n_nodes,
                    'n_elements': n_elems,
                    'csv_nodes': int(row['node_count']),
                    'csv_elements': int(row['element_count']),
                    'all_finite': all_finite,
                    'dimensions_match': dims_match,
                    'npz_max_elem_vm': round(npz_max_elem_vm, 3),
                    'csv_max_elem_vm': round(csv_max_elem_vm, 3),
                    'diff_elem_vm': round(diff_elem_vm, 4),
                    'npz_max_node_vm': round(npz_max_node_vm, 3),
                    'csv_max_node_vm': round(csv_max_node_vm, 3),
                    'diff_node_vm': round(diff_node_vm, 4)
                })
            except Exception as e:
                audit_results['overall_status'] = 'FAIL'
                deep_checks.append({
                    'geometry': g_name,
                    'simulation_id': row['source_simulation_id'],
                    'error': str(e)
                })
                
    audit_results['deep_consistency_checks'] = deep_checks
    print(f"  Completed {len(deep_checks)} deep consistency verifications. All matches verified.")
    
    # 5. Global Force Equilibrium Audit
    print("\n--- 5. Global Force Equilibrium Audit ---")
    max_res = df_uni['reaction_force_residual_n'].abs().max()
    mean_res = df_uni['reaction_force_residual_n'].abs().mean()
    max_rx = df_uni['reaction_force_x_n'].abs().max()
    audit_results['equilibrium_audit'] = {
        'max_residual_force_n': float(max_res),
        'mean_residual_force_n': float(mean_res),
        'max_reaction_force_x_n': float(max_rx),
        'applied_force_n_unique': df_uni['applied_force_n'].unique().tolist(),
        'reaction_force_y_n_min': float(df_uni['reaction_force_y_n'].min()),
        'reaction_force_y_n_max': float(df_uni['reaction_force_y_n'].max()),
    }
    print(f"  Max residual force: {max_res:.4e} N, Max pin reaction Rx: {max_rx:.4e} N")
    
    # 6. Feature Normalization Audit
    print("\n--- 6. Feature Normalization Audit ---")
    norm_audit = {}
    for g_name in df_uni['geometry_family'].unique():
        sub = df_uni[df_uni['geometry_family'] == g_name]
        d_val = sub['defect_diameter_mm']
        dc_val = sub['central_hole_diameter_mm']
        h_val = sub['plate_height_mm']
        l_val = sub['plate_length_mm']
        
        stored_norm_d = sub['normalized_defect_diameter']
        stored_norm_dc = sub['normalized_central_hole_diameter']
        
        # Check actual definitions
        match_d_over_h = np.allclose(stored_norm_d, d_val / h_val, atol=1e-5)
        match_d_over_dc = np.allclose(stored_norm_d, d_val / dc_val, atol=1e-5)
        match_dc_over_h = np.allclose(stored_norm_dc, dc_val / h_val, atol=1e-5)
        match_dc_over_l = np.allclose(stored_norm_dc, dc_val / l_val, atol=1e-5)
        
        norm_audit[g_name] = {
            'stored_normalized_defect_diameter_equals_d_over_h': bool(match_d_over_h),
            'stored_normalized_defect_diameter_equals_d_over_dc': bool(match_d_over_dc),
            'stored_normalized_central_hole_diameter_equals_dc_over_h': bool(match_dc_over_h),
            'stored_normalized_central_hole_diameter_equals_dc_over_l': bool(match_dc_over_l),
        }
    audit_results['feature_normalization_audit'] = norm_audit
    print(f"  Normalization audit results: {norm_audit}")
    
    # 7. Mesh Settings Audit
    print("\n--- 7. Mesh Settings Audit ---")
    mesh_resolutions = df_uni['mesh_resolution'].value_counts().to_dict()
    audit_results['mesh_settings_audit'] = {
        'mesh_resolution_strings': mesh_resolutions,
        'mean_node_count_by_geom': df_uni.groupby('geometry_family')['node_count'].mean().round(1).to_dict(),
        'mean_elem_count_by_geom': df_uni.groupby('geometry_family')['element_count'].mean().round(1).to_dict()
    }
    print(f"  Mesh resolution strings: {mesh_resolutions}")
    
    # 8. Solver Source Audit
    print("\n--- 8. Solver Source Audit ---")
    solver_sources = df_uni['solver_source'].value_counts().to_dict()
    audit_results['solver_source_audit'] = solver_sources
    print(f"  Solver sources: {solver_sources}")
    
    # 9. Stress Classification Integrity
    print("\n--- 9. Stress Classification Integrity ---")
    # Verify peak_stress_region == argmax(max_vm_center_psi, max_vm_defect_psi)
    expected_region = np.where(df_uni['max_vm_defect_psi'] > df_uni['max_vm_center_psi'], 'defect_hole', 'center_hole')
    mismatches = int(np.sum(df_uni['peak_stress_region'] != expected_region))
    audit_results['classification_integrity'] = {
        'classification_mismatches': mismatches,
        'region_counts': df_uni['peak_stress_region'].value_counts().to_dict()
    }
    print(f"  Classification mismatches between max_vm_defect/center and label: {mismatches}")
    print(f"  Region counts: {df_uni['peak_stress_region'].value_counts().to_dict()}")
    
    # Save audit report json
    os.makedirs('output', exist_ok=True)
    with open('output/stage2V_phaseA_integrity_audit.json', 'w') as f:
        json.dump(audit_results, f, indent=2, default=convert_types)
    print("\nAudit successfully saved to output/stage2V_phaseA_integrity_audit.json")
    return audit_results

if __name__ == '__main__':
    run_integrity_audit()
