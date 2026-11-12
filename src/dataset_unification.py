"""
Dataset Unification Module for Multi-Geometry FEM Campaign.

Merges the original 10,075 Square Plate dataset (1000 mm x 1000 mm)
and the new 10,075 Rectangular Plate dataset (1000 mm x 600 mm)
into a unified, standardized 20,150-record dataset.

Features:
- Standardized column naming, SI and engineering units across both geometry families.
- Normalized geometric and response variables for surrogate ML modeling:
  * normalized_defect_x = X / Plate_Length
  * normalized_defect_y = Y / Plate_Height
  * normalized_defect_diameter = D / Plate_Height
  * normalized_central_hole_diameter = D_c / Plate_Height (= 1/12 for both)
  * normalized_max_stress_element = max_sigma_vM / P (Kt)
  * normalized_displacement = max_disp / P
  * normalized_strain_energy = strain_energy / P^2
- Zero file duplication: links directly to original compressed field archives.
- Produces complete Data Dictionary (JSON).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


DATA_DICTIONARY = {
    "unified_id": "Unique global identifier across entire multi-geometry dataset (e.g. sq_00000, rect_00000)",
    "source_simulation_id": "Original integer simulation ID in the respective source campaign (0 to 10074)",
    "geometry_family": "Classification of plate geometry ('square_1000x1000' or 'rectangular_1000x600')",
    "plate_length_mm": "Plate dimension along X (tensile traction width) in millimeters",
    "plate_height_mm": "Plate dimension along Y (loading axis height) in millimeters",
    "aspect_ratio": "Dimensionless ratio of plate length (width) to plate height (L / H)",
    "central_hole_center_x_mm": "X-coordinate of fixed central circular hole in millimeters",
    "central_hole_center_y_mm": "Y-coordinate of fixed central circular hole in millimeters",
    "central_hole_diameter_mm": "Diameter of fixed central circular hole in millimeters",
    "defect_center_x_mm": "X-coordinate of variable defect circular hole in millimeters",
    "defect_center_y_mm": "Y-coordinate of variable defect circular hole in millimeters",
    "defect_diameter_mm": "Diameter of variable defect circular hole in millimeters",
    "normalized_defect_x": "Dimensionless defect X-position relative to plate length (X / L)",
    "normalized_defect_y": "Dimensionless defect Y-position relative to plate height (Y / H)",
    "normalized_defect_diameter": "Dimensionless defect diameter relative to plate height (D / H)",
    "normalized_central_hole_diameter": "Dimensionless central hole diameter relative to plate height (D_c / H)",
    "subset_type": "Sampling allocation category ('fixed_d_comparison' or 'variable_3param')",
    "youngs_modulus_psi": "Young's modulus of isotropic linearly elastic material in psi",
    "youngs_modulus_pa": "Young's modulus of isotropic linearly elastic material in Pascals",
    "poisson_ratio": "Poisson's ratio of material (dimensionless)",
    "plate_thickness_m": "Plate thickness assumed for 2D plane stress formulation in meters (1.0 m)",
    "applied_traction_psi": "Applied uniform uniaxial tensile traction on top boundary in psi (100.0 psi)",
    "applied_traction_pa": "Applied uniform uniaxial tensile traction on top boundary in Pascals (689,475.7 Pa)",
    "max_von_mises_element_psi": "Maximum element-centroid Von Mises stress in psi (Primary ML Target)",
    "max_von_mises_element_mpa": "Maximum element-centroid Von Mises stress in MPa",
    "max_von_mises_nodal_psi": "Maximum recovered nodal Von Mises stress via patch averaging in psi",
    "max_von_mises_nodal_mpa": "Maximum recovered nodal Von Mises stress in MPa",
    "max_principal_stress_psi": "Maximum principal tensile stress (sigma_1) in psi",
    "max_displacement_m": "Maximum resultant displacement magnitude in meters",
    "strain_energy_j": "Total internal elastic strain energy stored in plate in Joules",
    "normalized_max_stress_element": "Dimensionless stress concentration factor (max_von_mises_element_psi / P)",
    "normalized_max_stress_nodal": "Dimensionless nodal stress concentration factor (max_von_mises_nodal_psi / P)",
    "normalized_displacement_m_per_pa": "Linear scale-invariant displacement compliance in meters per Pascal",
    "normalized_strain_energy_j_per_pa2": "Linear scale-invariant strain energy compliance in Joules per Pascal squared",
    "applied_force_n": "Total integrated applied vertical tensile force on top boundary in Newtons",
    "reaction_force_y_n": "Total integrated vertical reaction force on bottom roller boundary in Newtons",
    "reaction_force_x_n": "Total horizontal reaction force at bottom midpoint pin constraint in Newtons",
    "reaction_force_residual_n": "Vertical force equilibrium error |F_applied + R_bottom| in Newtons",
    "max_vm_center_psi": "Peak element-centroid Von Mises stress in region adjacent to central hole in psi",
    "max_vm_defect_psi": "Peak element-centroid Von Mises stress in region adjacent to defect hole in psi",
    "peak_stress_region": "Boundary region exhibiting global maximum stress ('center_hole' or 'defect_hole')",
    "node_count": "Total number of nodes in the generated triangular mesh",
    "element_count": "Total number of 3-node linear triangular elements (CST) in mesh",
    "mesh_generation_time_s": "Time taken by Gmsh CAD construction and frontal-Delaunay 2D meshing in seconds",
    "solve_time_s": "Time taken for global stiffness assembly and direct sparse linear solve in seconds",
    "simulation_time_s": "Total execution time for geometry, meshing, solution, and stress recovery in seconds",
    "solver_source": "Finite element software solver identity ('scipy_sparse_cst_plane_stress')",
    "mesh_resolution": "Adaptive mesh refinement specification",
    "random_seed": "Random seed used for space-filling Sobol sampling",
    "source_dataset": "Relative filepath to original campaign summary CSV",
    "field_file": "Relative filepath to compressed .npz archive storing spatial stress and displacement fields"
}


def build_unified_dataset(
    square_csv_path: str = "data/fem_dataset_10075.csv",
    rect_csv_path: str = "data/rectangular_1000x600/fem_dataset_10075_rectangular.csv",
    output_unified_csv: str = "data/unified_dataset_20150.csv",
    output_dict_json: str = "data/unified_dataset_data_dictionary.json"
) -> pd.DataFrame:
    """
    Load, standardize, concatenate and save the unified multi-geometry dataset.
    """
    sq_path = Path(square_csv_path)
    rect_path = Path(rect_csv_path)
    
    if not sq_path.exists():
        raise FileNotFoundError(f"Square dataset not found at {sq_path}")
    if not rect_path.exists():
        raise FileNotFoundError(f"Rectangular dataset not found at {rect_path}")

    print(f"Loading Square Dataset from {sq_path}...")
    df_sq = pd.read_csv(sq_path)
    print(f"Loaded {len(df_sq)} square records.")

    print(f"Loading Rectangular Dataset from {rect_path}...")
    df_rect = pd.read_csv(rect_path)
    print(f"Loaded {len(df_rect)} rectangular records.")

    # 1. Standardize Square Records
    p_pa = 689475.7293168361
    p_psi = 100.0

    df_sq_std = pd.DataFrame()
    df_sq_std["unified_id"] = [f"sq_{i:05d}" for i in df_sq["simulation_id"]]
    df_sq_std["source_simulation_id"] = df_sq["simulation_id"].astype(int)
    df_sq_std["geometry_family"] = "square_1000x1000"
    df_sq_std["plate_length_mm"] = 1000.0
    df_sq_std["plate_height_mm"] = 1000.0
    df_sq_std["aspect_ratio"] = 1.0
    df_sq_std["central_hole_center_x_mm"] = 500.0
    df_sq_std["central_hole_center_y_mm"] = 500.0
    df_sq_std["central_hole_diameter_mm"] = round(1000.0 / 12.0, 4)
    
    df_sq_std["defect_center_x_mm"] = round(df_sq["defect_x"] * 1000.0, 4)
    df_sq_std["defect_center_y_mm"] = round(df_sq["defect_y"] * 1000.0, 4)
    df_sq_std["defect_diameter_mm"] = round(df_sq["defect_diameter"] * 1000.0, 4)
    
    df_sq_std["normalized_defect_x"] = round(df_sq["defect_x"] / 1.0, 6)
    df_sq_std["normalized_defect_y"] = round(df_sq["defect_y"] / 1.0, 6)
    df_sq_std["normalized_defect_diameter"] = round(df_sq["defect_diameter"] / 1.0, 6)
    df_sq_std["normalized_central_hole_diameter"] = round(1.0 / 12.0, 6)
    
    # Subset type mapping
    df_sq_std["subset_type"] = df_sq["subset_type"].replace({"fixed_d_1_80": "fixed_d_comparison"})
    
    df_sq_std["youngs_modulus_psi"] = 1e7
    df_sq_std["youngs_modulus_pa"] = 6.894757293168361e10
    df_sq_std["poisson_ratio"] = 0.30
    df_sq_std["plate_thickness_m"] = 1.0
    df_sq_std["applied_traction_psi"] = p_psi
    df_sq_std["applied_traction_pa"] = p_pa
    
    df_sq_std["max_von_mises_element_psi"] = df_sq["max_von_mises_element_psi"]
    df_sq_std["max_von_mises_element_mpa"] = df_sq["max_von_mises_element_mpa"]
    df_sq_std["max_von_mises_nodal_psi"] = df_sq["max_von_mises_nodal_psi"]
    df_sq_std["max_von_mises_nodal_mpa"] = df_sq["max_von_mises_nodal_mpa"]
    df_sq_std["max_principal_stress_psi"] = df_sq["max_principal_stress_psi"]
    df_sq_std["max_displacement_m"] = df_sq["max_displacement"]
    df_sq_std["strain_energy_j"] = df_sq["strain_energy"]
    
    df_sq_std["normalized_max_stress_element"] = round(df_sq["max_von_mises_element_psi"] / p_psi, 6)
    df_sq_std["normalized_max_stress_nodal"] = round(df_sq["max_von_mises_nodal_psi"] / p_psi, 6)
    df_sq_std["normalized_displacement_m_per_pa"] = df_sq["max_displacement"] / p_pa
    df_sq_std["normalized_strain_energy_j_per_pa2"] = df_sq["strain_energy"] / (p_pa ** 2)
    
    df_sq_std["applied_force_n"] = df_sq["applied_force_n"]
    df_sq_std["reaction_force_y_n"] = df_sq["reaction_force_y_n"]
    df_sq_std["reaction_force_x_n"] = df_sq["reaction_force_x_n"]
    df_sq_std["reaction_force_residual_n"] = df_sq["reaction_force_residual"]
    
    df_sq_std["max_vm_center_psi"] = df_sq["max_vm_center_psi"]
    df_sq_std["max_vm_defect_psi"] = df_sq["max_vm_defect_psi"]
    df_sq_std["peak_stress_region"] = df_sq["peak_stress_region"]
    
    df_sq_std["node_count"] = df_sq["node_count"].astype(int)
    df_sq_std["element_count"] = df_sq["element_count"].astype(int)
    df_sq_std["mesh_generation_time_s"] = df_sq["mesh_generation_time"]
    df_sq_std["solve_time_s"] = df_sq["solve_time"]
    df_sq_std["simulation_time_s"] = df_sq["simulation_time"]
    df_sq_std["solver_source"] = df_sq["solver_source"]
    df_sq_std["mesh_resolution"] = df_sq["mesh_resolution"]
    df_sq_std["random_seed"] = df_sq["random_seed"].astype(int)
    df_sq_std["source_dataset"] = "data/fem_dataset_10075.csv"
    df_sq_std["field_file"] = "data/" + df_sq["field_file"]

    # 2. Standardize Rectangular Records
    df_rect_std = pd.DataFrame()
    df_rect_std["unified_id"] = [f"rect_{i:05d}" for i in df_rect["simulation_id"]]
    df_rect_std["source_simulation_id"] = df_rect["simulation_id"].astype(int)
    df_rect_std["geometry_family"] = df_rect["geometry_family"]
    df_rect_std["plate_length_mm"] = df_rect["plate_length_mm"]
    df_rect_std["plate_height_mm"] = df_rect["plate_height_mm"]
    df_rect_std["aspect_ratio"] = df_rect["aspect_ratio"]
    df_rect_std["central_hole_center_x_mm"] = df_rect["central_hole_center_x_mm"]
    df_rect_std["central_hole_center_y_mm"] = df_rect["central_hole_center_y_mm"]
    df_rect_std["central_hole_diameter_mm"] = df_rect["central_hole_diameter_mm"]
    
    df_rect_std["defect_center_x_mm"] = df_rect["defect_center_x_mm"]
    df_rect_std["defect_center_y_mm"] = df_rect["defect_center_y_mm"]
    df_rect_std["defect_diameter_mm"] = df_rect["defect_diameter_mm"]
    
    df_rect_std["normalized_defect_x"] = df_rect["normalized_defect_x"]
    df_rect_std["normalized_defect_y"] = df_rect["normalized_defect_y"]
    df_rect_std["normalized_defect_diameter"] = df_rect["normalized_defect_diameter"]
    df_rect_std["normalized_central_hole_diameter"] = df_rect["normalized_central_hole_diameter"]
    
    df_rect_std["subset_type"] = df_rect["subset_type"].replace({"fixed_d_7_5mm": "fixed_d_comparison"})
    
    df_rect_std["youngs_modulus_psi"] = df_rect["youngs_modulus_psi"]
    df_rect_std["youngs_modulus_pa"] = df_rect["youngs_modulus_pa"]
    df_rect_std["poisson_ratio"] = df_rect["poisson_ratio"]
    df_rect_std["plate_thickness_m"] = df_rect["plate_thickness_m"]
    df_rect_std["applied_traction_psi"] = df_rect["applied_traction_psi"]
    df_rect_std["applied_traction_pa"] = df_rect["applied_traction_pa"]
    
    df_rect_std["max_von_mises_element_psi"] = df_rect["max_von_mises_element_psi"]
    df_rect_std["max_von_mises_element_mpa"] = df_rect["max_von_mises_element_mpa"]
    df_rect_std["max_von_mises_nodal_psi"] = df_rect["max_von_mises_nodal_psi"]
    df_rect_std["max_von_mises_nodal_mpa"] = df_rect["max_von_mises_nodal_mpa"]
    df_rect_std["max_principal_stress_psi"] = df_rect["max_principal_stress_psi"]
    df_rect_std["max_displacement_m"] = df_rect["max_displacement_m"]
    df_rect_std["strain_energy_j"] = df_rect["strain_energy_j"]
    
    df_rect_std["normalized_max_stress_element"] = df_rect["normalized_max_stress_element"]
    df_rect_std["normalized_max_stress_nodal"] = df_rect["normalized_max_stress_nodal"]
    df_rect_std["normalized_displacement_m_per_pa"] = df_rect["normalized_displacement_m_per_pa"]
    df_rect_std["normalized_strain_energy_j_per_pa2"] = df_rect["normalized_strain_energy_j_per_pa2"]
    
    df_rect_std["applied_force_n"] = df_rect["applied_force_n"]
    df_rect_std["reaction_force_y_n"] = df_rect["reaction_force_y_n"]
    df_rect_std["reaction_force_x_n"] = df_rect["reaction_force_x_n"]
    df_rect_std["reaction_force_residual_n"] = df_rect["reaction_force_residual_n"]
    
    df_rect_std["max_vm_center_psi"] = df_rect["max_vm_center_psi"]
    df_rect_std["max_vm_defect_psi"] = df_rect["max_vm_defect_psi"]
    df_rect_std["peak_stress_region"] = df_rect["peak_stress_region"]
    
    df_rect_std["node_count"] = df_rect["node_count"].astype(int)
    df_rect_std["element_count"] = df_rect["element_count"].astype(int)
    df_rect_std["mesh_generation_time_s"] = df_rect["mesh_generation_time_s"]
    df_rect_std["solve_time_s"] = df_rect["solve_time_s"]
    df_rect_std["simulation_time_s"] = df_rect["simulation_time_s"]
    df_rect_std["solver_source"] = df_rect["solver_source"]
    df_rect_std["mesh_resolution"] = df_rect["mesh_resolution"]
    df_rect_std["random_seed"] = df_rect["random_seed"].astype(int)
    df_rect_std["source_dataset"] = "data/rectangular_1000x600/fem_dataset_10075_rectangular.csv"
    df_rect_std["field_file"] = df_rect["field_file"]

    # 3. Concatenate into Unified Dataset
    df_unified = pd.concat([df_sq_std, df_rect_std], ignore_index=True)
    print(f"Concatenated Unified Dataset: {len(df_unified)} total rows, {len(df_unified.columns)} columns.")

    # Save CSV
    out_csv = Path(output_unified_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_unified.to_csv(out_csv, index=False)
    print(f"Saved unified dataset to {out_csv}")

    # Save Data Dictionary
    out_json = Path(output_dict_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
    print(f"Saved data dictionary to {out_json}")

    return df_unified


if __name__ == "__main__":
    build_unified_dataset()

