"""
Stage 3B: Phase A Audit Script
Audits dataset counts, parameter distributions (specifically defect diameter and d/Dc),
split manifests, and prepares baseline comparison data for the reference paper.
"""

import os
import json
import numpy as np
import pandas as pd

def run_phaseA_audit():
    print("=== STAGE 3B PHASE A AUDIT ===")
    
    # 1. Load Enriched Dataset (51,625)
    enriched_path = "data/enriched_dataset_51625.csv"
    assert os.path.exists(enriched_path), f"Missing {enriched_path}"
    df_enriched = pd.read_csv(enriched_path)
    print(f"Loaded enriched dataset: {df_enriched.shape}")
    
    # 2. Load Unified Dataset (50,375)
    unified_path = "data/unified_dataset_50375.csv"
    assert os.path.exists(unified_path), f"Missing {unified_path}"
    df_unified = pd.read_csv(unified_path)
    print(f"Loaded unified dataset: {df_unified.shape}")
    
    # 3. Load Split Manifest
    manifest_path = "data/stage3A_ann_split_manifest.csv"
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    df_manifest = pd.read_csv(manifest_path)
    print(f"Loaded split manifest: {df_manifest.shape}")
    print(f"Manifest split distribution:\n{df_manifest['split'].value_counts()}")
    
    # 4. Check defect diameter and d/Dc distribution across geometries in enriched dataset
    print(f"\nDefect diameter column in enriched dataset: defect_diameter_mm")
    print(f"Defect d/Dc column in enriched dataset: defect_d_over_Dc")
    
    # Diameter statistics overall
    stats_overall = df_enriched['defect_d_over_Dc'].describe()
    print("\nOverall d/Dc statistics (51,625):")
    print(stats_overall)
    
    # Per-geometry stats
    geom_col = 'geometry_family'
    print(f"\nPer-geometry statistics for d/Dc (grouping by {geom_col}):")
    geom_groups = df_enriched.groupby(geom_col)[['defect_diameter_mm', 'central_hole_diameter_mm', 'defect_d_over_Dc']].agg(
        ['count', 'min', 'max', 'mean', 'std', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 50), lambda x: np.percentile(x, 75)]
    )
    print(geom_groups)
    
    # Per-split stats
    print("\nPer-split statistics for defect_d_over_Dc:")
    split_groups = df_enriched.groupby('split')[['defect_diameter_mm', 'defect_d_over_Dc', 'Kt_element', 'max_von_mises_element_psi']].agg(
        ['count', 'min', 'max', 'mean', 'std', lambda x: np.percentile(x, 50), lambda x: np.percentile(x, 90), lambda x: np.percentile(x, 99)]
    )
    print(split_groups)
    
    # Build dictionary for JSON export
    audit_data = {
        "total_enriched_records": len(df_enriched),
        "total_unified_records": len(df_unified),
        "split_counts": df_manifest['split'].value_counts().to_dict(),
        "defect_d_over_Dc_overall": {
            "min": float(df_enriched['defect_d_over_Dc'].min()),
            "max": float(df_enriched['defect_d_over_Dc'].max()),
            "mean": float(df_enriched['defect_d_over_Dc'].mean()),
            "std": float(df_enriched['defect_d_over_Dc'].std()),
            "p25": float(np.percentile(df_enriched['defect_d_over_Dc'], 25)),
            "p50": float(np.percentile(df_enriched['defect_d_over_Dc'], 50)),
            "p75": float(np.percentile(df_enriched['defect_d_over_Dc'], 75)),
        },
        "per_geometry_summary": {},
        "per_split_summary": {},
        "paper_reported_benchmarks": {
            "study_type": "Single Square Plate (L=200, H=200, Dc=40 mm)",
            "forward_ann": {"R": 0.8242},
            "dnn_stacking": {"R": 0.8815},
            "pca_stacking": {"R": 0.9102},
            "meta_stacking": {"R": 0.9344}
        }
    }
    
    for geom in df_enriched['geometry_family'].unique():
        sub = df_enriched[df_enriched['geometry_family'] == geom]
        audit_data["per_geometry_summary"][geom] = {
            "count": len(sub),
            "plate_length_mm": float(sub['plate_length_mm'].iloc[0]),
            "plate_height_mm": float(sub['plate_height_mm'].iloc[0]),
            "aspect_ratio": float(sub['aspect_ratio'].iloc[0]),
            "central_hole_diameter_mm": float(sub['central_hole_diameter_mm'].iloc[0]),
            "defect_diameter_min": float(sub['defect_diameter_mm'].min()),
            "defect_diameter_max": float(sub['defect_diameter_mm'].max()),
            "d_over_Dc_min": float(sub['defect_d_over_Dc'].min()),
            "d_over_Dc_max": float(sub['defect_d_over_Dc'].max()),
            "d_over_Dc_mean": float(sub['defect_d_over_Dc'].mean()),
            "Kt_element_min": float(sub['Kt_element'].min()),
            "Kt_element_max": float(sub['Kt_element'].max()),
            "Kt_element_mean": float(sub['Kt_element'].mean()),
        }
        
    for sp in df_enriched['split'].unique():
        sub = df_enriched[df_enriched['split'] == sp]
        audit_data["per_split_summary"][sp] = {
            "count": len(sub),
            "d_over_Dc_min": float(sub['defect_d_over_Dc'].min()),
            "d_over_Dc_max": float(sub['defect_d_over_Dc'].max()),
            "d_over_Dc_mean": float(sub['defect_d_over_Dc'].mean()),
            "Kt_element_min": float(sub['Kt_element'].min()),
            "Kt_element_max": float(sub['Kt_element'].max()),
            "Kt_element_mean": float(sub['Kt_element'].mean()),
            "Kt_element_p99": float(np.percentile(sub['Kt_element'], 99)),
        }
    
    os.makedirs("output", exist_ok=True)
    with open("output/stage3B_phaseA_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print("\nSaved output/stage3B_phaseA_audit.json successfully.")

if __name__ == "__main__":
    run_phaseA_audit()
