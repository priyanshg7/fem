"""
Automated Parametric FEM Simulation and Dataset Generator for Rectangular Plate (Stage 2R).

Physical Setup:
- Plate Dimensions: Length = 1000 mm (1.0 m) along X, Height = 600 mm (0.6 m) along Y
- Fixed Central Hole: Center at (500 mm, 300 mm), Diameter = 50 mm (Radius = 25 mm, D_c/H = 1/12)
- Variable Defect Hole: Center at (X, Y), Diameter D in [3.75 mm, 15.00 mm] (0.00375 m to 0.01500 m)
- Fixed-Diameter Comparison Subset: D = 7.50 mm (0.00750 m, D/H = 1/80)
- Boundary & Hole Clearances: minimum 6 mm (0.006 m) material clearance
- Boundary Conditions: Bottom roller (u_y = 0 along y=0), bottom midpoint pinned (u_x = 0 at (500 mm, 0 mm)),
  top uniform tensile traction P = 100 psi (689,475.7 Pa).
- Adaptive Mesh Rule: h_defect = D / 12.0, h_center = 3 mm (0.003 m), h_coarse = 20 mm (0.020 m)

Features:
- Deterministic 3D Sobol sampling with seeds independent of square plate study (420 & 10420).
- Scale-invariant normalized response metrics: max_stress/P, displacement/P, strain_energy/P^2.
- Multi-process parallel execution with automatic core scaling.
- Incremental checkpointing and robust crash resumption.
- Compressed per-simulation .npz spatial field storage.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.geometry import PlateGeometryConfig, validate_defect_position
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, FEMResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetGeneratorRectangular")


# Rectangular Plate Geometry Constants (in SI meters)
PLATE_WIDTH_M = 1.0       # 1000 mm along X
PLATE_LENGTH_M = 0.6      # 600 mm along Y (height)
ASPECT_RATIO = PLATE_WIDTH_M / PLATE_LENGTH_M  # 1.666667

CENTER_HOLE_X_M = 0.50    # 500 mm
CENTER_HOLE_Y_M = 0.30    # 300 mm
CENTER_HOLE_DIAMETER_M = 0.050  # 50 mm (D_c / H = 50 / 600 = 1/12)
CENTER_HOLE_RADIUS_M = 0.025   # 25 mm

# Defect Parameters (in SI meters)
D_FIXED_M = 0.00750       # 7.50 mm (D / H = 7.5 / 600 = 1/80)
D_MIN_M = 0.00375         # 3.75 mm (D / H = 3.75 / 600 = 1/160)
D_MAX_M = 0.01500         # 15.00 mm (D / H = 15.0 / 600 = 1/40)

CLEARANCE_MARGIN_M = 0.006  # 6 mm

# Mesh Parameters (in SI meters)
H_CENTER_M = 0.0030       # 3 mm
H_COARSE_M = 0.0200       # 20 mm

# Conversion constants
PSI_TO_MPA = 0.006894757293168361
PA_TO_PSI = 1.4503773773020923e-4


def generate_rectangular_sobol_samples(
    n_fixed: int = 2075,
    n_var: int = 8000,
    seed_fixed: int = 420,
    seed_var: int = 10420,
    margin: float = CLEARANCE_MARGIN_M
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Generate deterministic (X, Y, D) parameter sets for the rectangular plate:
    1. Fixed-diameter comparison subset (D = 7.50 mm): n_fixed samples (2D Sobol)
    2. Continuous 3-parameter subset (D in [3.75, 15.00] mm): n_var samples (3D Sobol)
    """
    plate_w = PLATE_WIDTH_M
    plate_h = PLATE_LENGTH_M
    xc = CENTER_HOLE_X_M
    yc = CENTER_HOLE_Y_M
    rc = CENTER_HOLE_RADIUS_M
    g = margin

    # 1. Fixed D = 7.50 mm
    rd_f = D_FIXED_M / 2.0
    x_min_f, x_max_f = rd_f + g, plate_w - rd_f - g
    y_min_f, y_max_f = rd_f + g, plate_h - rd_f - g
    r_excl_f = rc + rd_f + g

    m_f = int(np.ceil(np.log2(n_fixed * 1.3)))
    m_f = max(m_f, 13)
    sampler_f = qmc.Sobol(d=2, scramble=True, seed=seed_fixed)
    raw_f = sampler_f.random_base2(m=m_f)

    pts_f = np.empty_like(raw_f)
    pts_f[:, 0] = x_min_f + raw_f[:, 0] * (x_max_f - x_min_f)
    pts_f[:, 1] = y_min_f + raw_f[:, 1] * (y_max_f - y_min_f)
    dist_f = np.hypot(pts_f[:, 0] - xc, pts_f[:, 1] - yc)
    valid_mask_f = dist_f >= r_excl_f
    valid_f = pts_f[valid_mask_f][:n_fixed]

    if len(valid_f) < n_fixed:
        raise RuntimeError(f"Insufficient valid fixed-D rectangular samples ({len(valid_f)} < {n_fixed})")

    fixed_data = np.hstack([valid_f, np.full((n_fixed, 1), D_FIXED_M)])
    fixed_types = ["fixed_d_7_5mm"] * n_fixed
    rejections_fixed = len(raw_f) - np.sum(valid_mask_f)

    # 2. Variable 3D subset
    m_v = int(np.ceil(np.log2(n_var * 1.4)))
    m_v = max(m_v, 15)
    sampler_v = qmc.Sobol(d=3, scramble=True, seed=seed_var)
    raw_v = sampler_v.random_base2(m=m_v)

    d_candidates = D_MIN_M + raw_v[:, 2] * (D_MAX_M - D_MIN_M)
    rd_candidates = d_candidates / 2.0

    x_min_v = rd_candidates + g
    x_max_v = plate_w - rd_candidates - g
    y_min_v = rd_candidates + g
    y_max_v = plate_h - rd_candidates - g

    x_cand = x_min_v + raw_v[:, 0] * (x_max_v - x_min_v)
    y_cand = y_min_v + raw_v[:, 1] * (y_max_v - y_min_v)

    dist_v = np.hypot(x_cand - xc, y_cand - yc)
    excl_v = rc + rd_candidates + g
    valid_mask_v = dist_v >= excl_v

    valid_v_x = x_cand[valid_mask_v][:n_var]
    valid_v_y = y_cand[valid_mask_v][:n_var]
    valid_v_d = d_candidates[valid_mask_v][:n_var]

    if len(valid_v_x) < n_var:
        raise RuntimeError(f"Insufficient valid variable-3D rectangular samples ({len(valid_v_x)} < {n_var})")

    var_data = np.column_stack([valid_v_x, valid_v_y, valid_v_d])
    var_types = ["variable_3param"] * n_var
    rejections_var = len(raw_v) - np.sum(valid_mask_v)

    total_data = np.vstack([fixed_data, var_data])
    total_types = fixed_types + var_types

    rejections_dict = {
        "fixed_subset_rejections": int(rejections_fixed),
        "variable_subset_rejections": int(rejections_var),
        "total_candidates_drawn": int(len(raw_f) + len(raw_v)),
        "total_accepted": int(len(total_data)),
        "seed_fixed": seed_fixed,
        "seed_var": seed_var
    }

    return total_data, total_types, rejections_dict


def run_rectangular_simulation_worker(task_args: Tuple) -> Dict[str, Any]:
    """
    Worker function executed in parallel across CPU cores.
    
    Arguments:
        (sim_id, x, y, d, subset_type, save_fields, fields_dir, seed)
    """
    (sim_id, x, y, d, subset_type, save_fields, fields_dir, seed) = task_args
    t_start = time.perf_counter()

    try:
        # Adaptive mesh rule: h_defect = D / 12.0
        h_defect = float(d / 12.0)
        h_center = H_CENTER_M
        h_coarse = H_COARSE_M

        geom_config = PlateGeometryConfig(
            plate_width=PLATE_WIDTH_M,
            plate_length=PLATE_LENGTH_M,
            center_hole_x=CENTER_HOLE_X_M,
            center_hole_y=CENTER_HOLE_Y_M,
            center_hole_diameter=CENTER_HOLE_DIAMETER_M,
            has_defect=True,
            defect_hole_diameter=float(d),
            defect_hole_x=float(x),
            defect_hole_y=float(y),
            h_coarse=h_coarse,
            h_center_hole=h_center,
            h_defect_hole=h_defect,
            clearance_margin=CLEARANCE_MARGIN_M
        )

        if not validate_defect_position(geom_config, margin=CLEARANCE_MARGIN_M):
            return {
                "simulation_id": sim_id,
                "status": "FAILED",
                "error": f"Clearance violation at ({x:.6f}, {y:.6f}, d={d:.6f})"
            }

        mat_config = MaterialConfig()

        result: FEMResult = solve_plane_stress_fem(geom_config, mat_config, verbose=False)
        t_duration = time.perf_counter() - t_start

        # Validity checks
        if result.num_nodes <= 0 or result.num_elements <= 0:
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Empty mesh"}
        if np.isnan(result.max_element_von_mises_psi) or np.isinf(result.max_element_von_mises_psi):
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-finite element stress"}
        if np.isnan(result.max_von_mises_psi) or np.isinf(result.max_von_mises_psi):
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-finite nodal stress"}
        if result.strain_energy <= 0.0:
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-positive strain energy"}

        rel_eq_err = abs(result.equilibrium_error_n) / abs(result.applied_force_total_n)
        if rel_eq_err > 1e-4:
            return {
                "simulation_id": sim_id,
                "status": "FAILED",
                "error": f"Equilibrium error excessive: {rel_eq_err:.2e}"
            }

        # Regional Stress Identification
        el_coords = result.nodes[result.elements]
        centroids = np.mean(el_coords, axis=1)

        rc = CENTER_HOLE_RADIUS_M
        rd = d / 2.0
        dc = np.hypot(centroids[:, 0] - CENTER_HOLE_X_M, centroids[:, 1] - CENTER_HOLE_Y_M)
        dd = np.hypot(centroids[:, 0] - x, centroids[:, 1] - y)

        mask_center = (dc <= 1.8 * rc) & (dc < dd)
        mask_defect = (dd <= 2.2 * rd) & (dd < dc)

        vm_elem_psi = result.element_von_mises * PA_TO_PSI
        max_vm_center_psi = float(np.max(vm_elem_psi[mask_center])) if np.any(mask_center) else float(result.max_element_von_mises_psi)
        max_vm_defect_psi = float(np.max(vm_elem_psi[mask_defect])) if np.any(mask_defect) else 0.0

        global_max_elem_idx = int(np.argmax(vm_elem_psi))
        if mask_defect[global_max_elem_idx]:
            peak_stress_region = "defect_hole"
        elif mask_center[global_max_elem_idx]:
            peak_stress_region = "center_hole"
        else:
            peak_stress_region = "far_field"

        # Scale-invariant linear response metrics
        p_applied_psi = mat_config.applied_traction_psi      # 100.0 psi
        p_applied_pa = mat_config.applied_traction_pa        # 689,475.7 Pa
        
        normalized_max_stress_elem = result.max_element_von_mises_psi / p_applied_psi
        normalized_max_stress_nodal = result.max_von_mises_psi / p_applied_psi
        normalized_displacement = result.max_displacement / p_applied_pa
        normalized_strain_energy = result.strain_energy / (p_applied_pa ** 2)

        # Save compressed full-field data
        field_rel_path = f"data/rectangular_1000x600/fields/sim_{sim_id:05d}.npz"
        if save_fields and fields_dir is not None:
            field_abs_path = Path(fields_dir) / f"sim_{sim_id:05d}.npz"
            np.savez_compressed(
                field_abs_path,
                sim_id=np.int32(sim_id),
                geometry_family="rectangular_1000x600",
                plate_length_mm=np.float32(PLATE_WIDTH_M * 1000.0),
                plate_height_mm=np.float32(PLATE_LENGTH_M * 1000.0),
                defect_coords_mm=np.array([x * 1000.0, y * 1000.0], dtype=np.float32),
                defect_diameter_mm=np.float32(d * 1000.0),
                subset_type=subset_type,
                element_centroids_mm=(centroids * 1000.0).astype(np.float32),
                element_von_mises_psi=vm_elem_psi.astype(np.float32),
                nodal_coords_mm=(result.nodes * 1000.0).astype(np.float32),
                nodal_von_mises_psi=(result.nodal_von_mises * PA_TO_PSI).astype(np.float32),
                displacements_m=result.displacements.astype(np.float32)
            )

        # Summary record with explicit units and normalized quantities
        record = {
            "simulation_id": int(sim_id),
            "status": "SUCCESS",
            "geometry_family": "rectangular_1000x600",
            "plate_length_mm": 1000.0,
            "plate_height_mm": 600.0,
            "aspect_ratio": round(ASPECT_RATIO, 6),
            "central_hole_center_x_mm": 500.0,
            "central_hole_center_y_mm": 300.0,
            "central_hole_diameter_mm": 50.0,
            "defect_center_x_mm": round(float(x * 1000.0), 4),
            "defect_center_y_mm": round(float(y * 1000.0), 4),
            "defect_diameter_mm": round(float(d * 1000.0), 4),
            "normalized_defect_x": round(float(x / PLATE_WIDTH_M), 6),
            "normalized_defect_y": round(float(y / PLATE_LENGTH_M), 6),
            "normalized_defect_diameter": round(float(d / PLATE_LENGTH_M), 6),
            "normalized_central_hole_diameter": round(float(CENTER_HOLE_DIAMETER_M / PLATE_LENGTH_M), 6),
            "subset_type": subset_type,
            "youngs_modulus_psi": float(mat_config.youngs_modulus_psi),
            "youngs_modulus_pa": float(mat_config.youngs_modulus_pa),
            "poisson_ratio": float(mat_config.poisson_ratio),
            "plate_thickness_m": float(mat_config.thickness),
            "applied_traction_psi": float(p_applied_psi),
            "applied_traction_pa": float(p_applied_pa),
            "max_von_mises_element_psi": round(float(result.max_element_von_mises_psi), 4),
            "max_von_mises_element_mpa": round(float(result.max_element_von_mises_psi * PSI_TO_MPA), 4),
            "max_von_mises_nodal_psi": round(float(result.max_von_mises_psi), 4),
            "max_von_mises_nodal_mpa": round(float(result.max_von_mises_psi * PSI_TO_MPA), 4),
            "max_principal_stress_psi": round(float(result.max_principal_stress_psi), 4),
            "max_displacement_m": float(result.max_displacement),
            "strain_energy_j": float(result.strain_energy),
            "normalized_max_stress_element": round(float(normalized_max_stress_elem), 6),
            "normalized_max_stress_nodal": round(float(normalized_max_stress_nodal), 6),
            "normalized_displacement_m_per_pa": float(normalized_displacement),
            "normalized_strain_energy_j_per_pa2": float(normalized_strain_energy),
            "applied_force_n": round(float(result.applied_force_total_n), 2),
            "reaction_force_y_n": round(float(result.reaction_force_bottom_n), 2),
            "reaction_force_x_n": round(float(result.horizontal_reaction_n), 6),
            "reaction_force_residual_n": float(result.equilibrium_error_n),
            "max_vm_center_psi": round(max_vm_center_psi, 4),
            "max_vm_defect_psi": round(max_vm_defect_psi, 4),
            "peak_stress_region": peak_stress_region,
            "node_count": int(result.num_nodes),
            "element_count": int(result.num_elements),
            "mesh_generation_time_s": round(float(result.mesh_generation_time), 4),
            "solve_time_s": round(float(result.solve_time), 4),
            "simulation_time_s": round(float(t_duration), 4),
            "solver_source": "scipy_sparse_cst_plane_stress",
            "mesh_resolution": "adaptive_hd_D_over_12_hc3mm_hg20mm",
            "random_seed": int(seed),
            "field_file": field_rel_path
        }
        return record

    except Exception as e:
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": str(e)
        }


class RectangularDatasetPipeline:
    """
    Orchestrates the 10,075 rectangular simulation dataset generation.
    """
    def __init__(
        self,
        output_dir: str = "data/rectangular_1000x600",
        fields_dir: str = "data/rectangular_1000x600/fields",
        n_fixed: int = 2075,
        n_var: int = 8000,
        seed_fixed: int = 420,
        seed_var: int = 10420,
        num_workers: Optional[int] = None
    ):
        self.output_dir = Path(output_dir)
        self.fields_dir = Path(fields_dir)
        self.n_fixed = n_fixed
        self.n_var = n_var
        self.n_samples = n_fixed + n_var
        self.seed_fixed = seed_fixed
        self.seed_var = seed_var
        self.num_workers = num_workers or min(os.cpu_count() or 4, 12)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fields_dir.mkdir(parents=True, exist_ok=True)

        self.summary_csv = self.output_dir / "fem_dataset_10075_rectangular.csv"
        self.failed_csv = self.output_dir / "failed_simulations_rectangular.csv"
        self.parameters_csv = self.output_dir / "sampled_parameters_rectangular.csv"
        self.metadata_json = self.output_dir / "dataset_metadata.json"

        self.parameters, self.subset_types, self.rejections = self._init_parameters()

    def _init_parameters(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        if self.parameters_csv.exists():
            df_p = pd.read_csv(self.parameters_csv)
            if len(df_p) >= self.n_samples:
                logger.info(f"Loaded {len(df_p)} parameter sets from {self.parameters_csv}")
                params = df_p[["defect_x_m", "defect_y_m", "defect_diameter_m"]].to_numpy()[:self.n_samples]
                subtypes = df_p["subset_type"].tolist()[:self.n_samples]
                rejections = {"note": "loaded from cached parameter csv"}
                return params, subtypes, rejections

        logger.info(
            f"Generating 10,075 rectangular parameters: {self.n_fixed} fixed-D (seed {self.seed_fixed}) "
            f"+ {self.n_var} variable-3D (seed {self.seed_var})..."
        )
        params, subtypes, rejections = generate_rectangular_sobol_samples(
            n_fixed=self.n_fixed,
            n_var=self.n_var,
            seed_fixed=self.seed_fixed,
            seed_var=self.seed_var
        )

        df_p = pd.DataFrame({
            "simulation_id": np.arange(self.n_samples),
            "defect_x_m": params[:, 0],
            "defect_y_m": params[:, 1],
            "defect_diameter_m": params[:, 2],
            "defect_x_mm": params[:, 0] * 1000.0,
            "defect_y_mm": params[:, 1] * 1000.0,
            "defect_diameter_mm": params[:, 2] * 1000.0,
            "normalized_defect_x": params[:, 0] / PLATE_WIDTH_M,
            "normalized_defect_y": params[:, 1] / PLATE_LENGTH_M,
            "normalized_defect_diameter": params[:, 2] / PLATE_LENGTH_M,
            "subset_type": subtypes
        })
        df_p.to_csv(self.parameters_csv, index=False)
        logger.info(f"Saved 10,075 parameters to {self.parameters_csv}")
        return params, subtypes, rejections

    def get_completed_ids(self) -> set:
        if not self.summary_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.summary_csv, usecols=["simulation_id"])
            return set(df["simulation_id"].tolist())
        except Exception as e:
            logger.warning(f"Error reading existing summary: {e}")
            return set()

    def run(
        self,
        batch_size: Optional[int] = None,
        save_fields: bool = True,
        save_interval: int = 50
    ):
        completed_ids = self.get_completed_ids()
        logger.info(f"Checkpoint check: {len(completed_ids)} / {self.n_samples} already completed.")

        total_to_run = self.n_samples if batch_size is None else min(batch_size, self.n_samples)
        tasks = []
        for sim_id in range(total_to_run):
            if sim_id not in completed_ids:
                x, y, d = self.parameters[sim_id]
                subtype = self.subset_types[sim_id]
                tasks.append((
                    sim_id, x, y, d, subtype,
                    save_fields, str(self.fields_dir), self.seed_fixed
                ))

        if not tasks:
            logger.info("All requested simulations have already been successfully completed!")
            return

        logger.info(
            f"Launching {len(tasks)} simulations using {self.num_workers} worker processes "
            f"(Rectangular 1000x600 mm: h_defect = D / 12.0, h_center = 3mm, h_coarse = 20mm)..."
        )

        success_records = []
        failed_records = []
        t0 = time.time()
        completed_count = len(completed_ids)

        write_header_summary = not self.summary_csv.exists()
        write_header_failed = not self.failed_csv.exists()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_id = {executor.submit(run_rectangular_simulation_worker, t): t[0] for t in tasks}

            for i, future in enumerate(as_completed(future_to_id), 1):
                res = future.result()
                sim_id = res["simulation_id"]

                if res["status"] == "SUCCESS":
                    del res["status"]
                    success_records.append(res)
                else:
                    failed_records.append(res)
                    logger.error(f"Simulation {sim_id} failed: {res.get('error')}")

                if len(success_records) >= save_interval:
                    df_flush = pd.DataFrame(success_records)
                    df_flush.to_csv(self.summary_csv, mode="a", header=write_header_summary, index=False)
                    write_header_summary = False
                    completed_count += len(success_records)
                    success_records = []

                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(tasks) - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {completed_count}/{self.n_samples} ({i}/{len(tasks)} this run) | "
                        f"Rate: {rate:.2f} sims/s | ETA: {eta/60:.1f} min"
                    )

                if len(failed_records) >= 10:
                    df_failed = pd.DataFrame(failed_records)
                    df_failed.to_csv(self.failed_csv, mode="a", header=write_header_failed, index=False)
                    write_header_failed = False
                    failed_records = []

        if success_records:
            df_flush = pd.DataFrame(success_records)
            df_flush.to_csv(self.summary_csv, mode="a", header=write_header_summary, index=False)
            completed_count += len(success_records)

        if failed_records:
            df_failed = pd.DataFrame(failed_records)
            df_failed.to_csv(self.failed_csv, mode="a", header=write_header_failed, index=False)

        total_elapsed = time.time() - t0
        logger.info(
            f"Run completed. Processed {len(tasks)} tasks in {total_elapsed:.1f}s "
            f"({len(tasks)/total_elapsed:.2f} sims/s). Total successful: {completed_count}."
        )

        meta = {
            "geometry_family": "rectangular_1000x600",
            "plate_length_mm": 1000.0,
            "plate_height_mm": 600.0,
            "aspect_ratio": ASPECT_RATIO,
            "central_hole_center_mm": [500.0, 300.0],
            "central_hole_diameter_mm": 50.0,
            "clearance_margin_mm": 6.0,
            "total_simulations": completed_count,
            "target_simulations": self.n_samples,
            "fixed_diameter_subset_count": self.n_fixed,
            "variable_3param_subset_count": self.n_var,
            "fixed_diameter_value_mm": 7.50,
            "variable_diameter_range_mm": [3.75, 15.00],
            "adaptive_mesh_rule": "h_defect = D / 12.0, h_center = 3mm, h_coarse = 20mm",
            "solver": "scipy_sparse_cst_plane_stress",
            "seed_fixed": self.seed_fixed,
            "seed_var": self.seed_var,
            "rejections": self.rejections,
            "summary_csv": str(self.summary_csv),
            "fields_dir": str(self.fields_dir)
        }
        with open(self.metadata_json, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rectangular Plate Automated FEM Dataset Pipeline")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit number of simulations (e.g. 100 for smoke test)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no-fields", action="store_true", help="Disable saving per-simulation .npz field data")
    args = parser.parse_args()

    pipeline = RectangularDatasetPipeline(
        n_fixed=2075,
        n_var=8000,
        num_workers=args.workers
    )
    pipeline.run(
        batch_size=args.batch_size,
        save_fields=not args.no_fields
    )

