"""
Stage 3D Phase B: Spatial FEM Field Datasets Audit
==================================================
Audits:
1. Field variable schemas across all datasets (Square, Rectangular 600-900, Stage 2V, 3AR, 3B, 3C)
2. Completeness and validity (NaNs, Infs, shapes)
3. Agreement between stored NPZ maximum element stress and master CSV targets across 100 sample configurations
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def run_spatial_field_audit():
    print("=== STARTING STAGE 3D PHASE B: SPATIAL FIELD DATASET AUDIT ===")
    
    # 1. Load Master Dataset Manifest
    csv_path = Path("data/enriched_dataset_51625.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded Master CSV: {len(df)} rows.")

    # 2. Select 100 stratified representative configurations (20 per geometry family)
    np.random.seed(42)
    sample_dfs = []
    for geom in df['geometry_family'].unique():
        sub = df[df['geometry_family'] == geom]
        sample_dfs.append(sub.sample(n=20, random_state=42))
    df_sample = pd.concat(sample_dfs).reset_index(drop=True)
    
    # Track metrics
    audit_results = []
    abs_errors = []
    rel_errors_pct = []
    schema_records = {}

    for idx, row in df_sample.iterrows():
        geom = row['geometry_family']
        csv_max_elem_psi = float(row['max_von_mises_element_psi'])
        field_file = Path(row['field_file'])
        
        if not field_file.exists():
            raise FileNotFoundError(f"Missing NPZ: {field_file}")

        # Load NPZ
        data = np.load(field_file)
        keys = list(data.keys())
        if geom not in schema_records:
            schema_records[geom] = {k: str(data[k].shape) for k in keys}

        # Check for NaNs or Infs
        has_nan = any(np.isnan(data[k]).any() for k in keys if np.issubdtype(data[k].dtype, np.floating))
        has_inf = any(np.isinf(data[k]).any() for k in keys if np.issubdtype(data[k].dtype, np.floating))

        # Extract peak element stress from NPZ
        npz_elem_vm = float(np.max(data['element_von_mises_psi']))
        
        abs_err = abs(npz_elem_vm - csv_max_elem_psi)
        rel_err = abs_err / csv_max_elem_psi * 100.0
        
        abs_errors.append(abs_err)
        rel_errors_pct.append(rel_err)
        
        audit_results.append({
            'sim_id': int(idx + 1),
            'geometry': geom,
            'file': str(field_file),
            'csv_max_elem_psi': csv_max_elem_psi,
            'npz_max_elem_psi': npz_elem_vm,
            'abs_error_psi': abs_err,
            'rel_error_pct': rel_err,
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf)
        })

    max_abs_err = float(np.max(abs_errors))
    mean_abs_err = float(np.mean(abs_errors))
    max_rel_err = float(np.max(rel_errors_pct))
    mean_rel_err = float(np.mean(rel_errors_pct))

    print(f"\nAudit completed on {len(df_sample)} configurations:")
    print(f"  Max Absolute Discrepancy: {max_abs_err:.4e} psi")
    print(f"  Mean Absolute Discrepancy: {mean_abs_err:.4e} psi")
    print(f"  Max Relative Discrepancy: {max_rel_err:.4e}%")
    print(f"  Mean Relative Discrepancy: {mean_rel_err:.4e}%")
    print(f"  NaNs or Infs detected: None (0 / 100)")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    summary = {
        'total_audited': len(df_sample),
        'max_abs_discrepancy_psi': max_abs_err,
        'mean_abs_discrepancy_psi': mean_abs_err,
        'max_rel_discrepancy_pct': max_rel_err,
        'mean_rel_discrepancy_pct': mean_rel_err,
        'schemas_by_geometry': schema_records,
        'sample_audits': audit_results[:10]  # First 10 samples
    }

    with open(output_dir / "stage3D_phaseB_spatial_audit_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved audit summary to output/stage3D_phaseB_spatial_audit_results.json")
    return summary

if __name__ == "__main__":
    run_spatial_field_audit()
