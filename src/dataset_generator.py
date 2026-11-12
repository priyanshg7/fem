"""
Automated Parametric FEM Simulation and Dataset Generator for 3-Parameter Plate Study.

Generates 10,075 reproducible plate configurations with three independent variable parameters:
- X: Defect center X-coordinate
- Y: Defect center Y-coordinate
- D: Defect hole diameter (ranging from 1/160 m to 1/40 m)

Features:
- Deterministic 3D space-filling sampling using Scrambled Sobol sequences.
- Dedicated fixed-diameter comparison subset at D = 1/80 m (original paper reference).
- Continuous variable 3-parameter subset across D in [1/160, 1/40] m.
- Strict clearance verification: boundary clearance >= 0.01 m, hole-to-hole clearance >= 0.01 m.
- Adaptive mesh-sizing rule: h_defect = D / 12.0, h_center = 0.0030, h_coarse = 0.020.
- Regional stress identification: separates peak stresses around central hole vs defect hole.
- Multi-process parallel execution with automatic CPU core scaling.
- Incremental checkpointing, fault isolation, and robust crash resumption.
- Per-simulation compressed field storage (.npz) for spatial stress & displacement.
- Master summary table (CSV) for machine learning training.
"""

import os
import sys
import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.geometry import PlateGeometryConfig, validate_defect_position
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, FEMResult

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetGenerator3D")


# Fixed Physical Constants (Singh & Pamnani 2025)
PLATE_DIM = 1.0
CENTER_HOLE_DIAMETER = 1.0 / 12.0
CENTER_HOLE_RADIUS = CENTER_HOLE_DIAMETER / 2.0  # 1/24 ~ 0.0416667
BOUNDARY_CLEARANCE = 0.01
HOLE_CLEARANCE = 0.01

# Diameter ranges
D_FIXED = 1.0 / 80.0       # 0.0125 m (Original paper reference)
D_MIN = 1.0 / 160.0        # 0.00625 m (Smallest defect)
D_MAX = 1.0 / 40.0         # 0.02500 m (Largest defect)

# Production Mesh Parameters
H_CENTER_PROD = 0.0030
H_COARSE_PROD = 0.020

# Conversions
PSI_TO_MPA = 0.006894757293168361


def generate_3param_sobol_samples(
    n_fixed: int = 2075,
    n_var: int = 8000,
    seed_fixed: int = 42,
    seed_var: int = 1042,
    margin: float = BOUNDARY_CLEARANCE
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Generate deterministic, space-filling (X, Y, D) parameter sets:
    1. Fixed-diameter comparison subset (D = 1/80 m): n_fixed samples (2D Sobol)
    2. Continuous 3-parameter subset (D in [1/160, 1/40] m): n_var samples (3D Sobol)
    
    Total samples = n_fixed + n_var (default: 2075 + 8000 = 10,075)
    
    Returns:
        params: (N, 3) array of [X, Y, D]
        subset_types: list of strings ("fixed_d_1_80" or "variable_3param")
        rejections: dict of rejected sample counts
    """
    rc = CENTER_HOLE_RADIUS

    # 1. Fixed D = 1/80 subset
    rd_fixed = D_FIXED / 2.0
    x_min_f, x_max_f = rd_fixed + margin, PLATE_DIM - rd_fixed - margin
    y_min_f, y_max_f = rd_fixed + margin, PLATE_DIM - rd_fixed - margin
    r_excl_f = rc + rd_fixed + margin

    m_f = int(np.ceil(np.log2(n_fixed * 1.3)))
    m_f = max(m_f, 13)  # 2^13 = 8192
    sampler_f = qmc.Sobol(d=2, scramble=True, seed=seed_fixed)
    raw_f = sampler_f.random_base2(m=m_f)

    pts_f = np.empty_like(raw_f)
    pts_f[:, 0] = x_min_f + raw_f[:, 0] * (x_max_f - x_min_f)
    pts_f[:, 1] = y_min_f + raw_f[:, 1] * (y_max_f - y_min_f)
    dist_f = np.hypot(pts_f[:, 0] - 0.5, pts_f[:, 1] - 0.5)
    valid_mask_f = dist_f >= r_excl_f
    valid_f = pts_f[valid_mask_f][:n_fixed]

    if len(valid_f) < n_fixed:
        raise RuntimeError(f"Insufficient valid fixed-D samples ({len(valid_f)} < {n_fixed})")

    fixed_d_col = np.full((n_fixed, 1), D_FIXED)
    fixed_params = np.hstack([valid_f, fixed_d_col])
    fixed_types = ["fixed_d_1_80"] * n_fixed
    rejections_fixed = len(raw_f) - np.sum(valid_mask_f)

    # 2. Variable 3D subset
    m_v = int(np.ceil(np.log2(n_var * 1.4)))
    m_v = max(m_v, 15)  # 2^15 = 32768
    sampler_v = qmc.Sobol(d=3, scramble=True, seed=seed_var)
    raw_v = sampler_v.random_base2(m=m_v)

    d_candidates = D_MIN + raw_v[:, 2] * (D_MAX - D_MIN)
    rd_candidates = d_candidates / 2.0

    x_min_v = rd_candidates + margin
    x_max_v = PLATE_DIM - rd_candidates - margin
    y_min_v = rd_candidates + margin
    y_max_v = PLATE_DIM - rd_candidates - margin

    x_cand = x_min_v + raw_v[:, 0] * (x_max_v - x_min_v)
    y_cand = y_min_v + raw_v[:, 1] * (y_max_v - y_min_v)

    dist_v = np.hypot(x_cand - 0.5, y_cand - 0.5)
    excl_v = rc + rd_candidates + margin
    valid_mask_v = dist_v >= excl_v

    valid_v_x = x_cand[valid_mask_v][:n_var]
    valid_v_y = y_cand[valid_mask_v][:n_var]
    valid_v_d = d_candidates[valid_mask_v][:n_var]

    if len(valid_v_x) < n_var:
        raise RuntimeError(f"Insufficient valid variable-3D samples ({len(valid_v_x)} < {n_var})")

    var_params = np.column_stack([valid_v_x, valid_v_y, valid_v_d])
    var_types = ["variable_3param"] * n_var
    rejections_var = len(raw_v) - np.sum(valid_mask_v)

    total_params = np.vstack([fixed_params, var_params])
    total_types = fixed_types + var_types

    rejections_dict = {
        "fixed_subset_rejections": int(rejections_fixed),
        "variable_subset_rejections": int(rejections_var),
        "total_candidates_drawn": int(len(raw_f) + len(raw_v)),
        "total_accepted": int(len(total_params))
    }

    return total_params, total_types, rejections_dict


def run_3param_simulation_worker(task_args: Tuple) -> Dict[str, Any]:
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
        h_center = H_CENTER_PROD
        h_coarse = H_COARSE_PROD

        # 1. Configure geometry
        geom_config = PlateGeometryConfig(
            plate_length=PLATE_DIM,
            plate_width=PLATE_DIM,
            center_hole_diameter=CENTER_HOLE_DIAMETER,
            center_hole_x=0.5,
            center_hole_y=0.5,
            has_defect=True,
            defect_hole_diameter=float(d),
            defect_hole_x=float(x),
            defect_hole_y=float(y),
            h_coarse=h_coarse,
            h_center_hole=h_center,
            h_defect_hole=h_defect
        )

        # Clearance validation
        if not validate_defect_position(geom_config, margin=BOUNDARY_CLEARANCE):
            return {
                "simulation_id": sim_id,
                "status": "FAILED",
                "error": f"Clearance violation at ({x:.6f}, {y:.6f}, d={d:.6f})"
            }

        mat_config = MaterialConfig()

        # 2. Solve FEM
        result: FEMResult = solve_plane_stress_fem(geom_config, mat_config, verbose=False)
        t_duration = time.perf_counter() - t_start

        # 3. Physics & Numerical Sanity Checks
        if result.num_nodes <= 0 or result.num_elements <= 0:
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Empty mesh"}
        if np.isnan(result.max_element_von_mises_psi) or np.isinf(result.max_element_von_mises_psi):
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-finite element stress"}
        if np.isnan(result.max_von_mises_psi) or np.isinf(result.max_von_mises_psi):
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-finite nodal stress"}
        if result.strain_energy <= 0.0:
            return {"simulation_id": sim_id, "status": "FAILED", "error": "Non-positive strain energy"}

        # Force equilibrium check
        rel_eq_err = abs(result.equilibrium_error_n) / abs(result.applied_force_total_n)
        if rel_eq_err > 1e-4:
            return {
                "simulation_id": sim_id,
                "status": "FAILED",
                "error": f"Equilibrium error excessive: {rel_eq_err:.2e}"
            }

        # 4. Regional Stress Identification
        el_coords = result.nodes[result.elements]  # (M, 3, 2)
        centroids = np.mean(el_coords, axis=1)    # (M, 2)
        
        rc = CENTER_HOLE_RADIUS
        rd = d / 2.0
        dc = np.hypot(centroids[:, 0] - 0.5, centroids[:, 1] - 0.5)
        dd = np.hypot(centroids[:, 0] - x, centroids[:, 1] - y)

        mask_center = (dc <= 1.8 * rc) & (dc < dd)
        mask_defect = (dd <= 2.2 * rd) & (dd < dc)

        vm_elem_psi = result.element_von_mises * 1.4503773773e-4
        max_vm_center_psi = float(np.max(vm_elem_psi[mask_center])) if np.any(mask_center) else float(result.max_element_von_mises_psi)
        max_vm_defect_psi = float(np.max(vm_elem_psi[mask_defect])) if np.any(mask_defect) else 0.0

        # Identify peak stress region
        global_max_elem_idx = int(np.argmax(vm_elem_psi))
        if mask_defect[global_max_elem_idx]:
            peak_stress_region = "defect_hole"
        elif mask_center[global_max_elem_idx]:
            peak_stress_region = "center_hole"
        else:
            peak_stress_region = "far_field"

        # 5. Save compressed full-field data
        field_rel_path = f"fields/sim_{sim_id:05d}.npz"
        if save_fields and fields_dir is not None:
            field_abs_path = Path(fields_dir) / f"sim_{sim_id:05d}.npz"
            np.savez_compressed(
                field_abs_path,
                sim_id=np.int32(sim_id),
                defect_coords=np.array([x, y], dtype=np.float32),
                defect_diameter=np.float32(d),
                subset_type=subset_type,
                element_centroids=centroids.astype(np.float32),
                element_von_mises_psi=vm_elem_psi.astype(np.float32),
                nodal_coords=result.nodes.astype(np.float32),
                nodal_von_mises_psi=(result.nodal_von_mises * 1.4503773773e-4).astype(np.float32),
                displacements=result.displacements.astype(np.float32)
            )

        # 6. Build summary record
        record = {
            "simulation_id": int(sim_id),
            "status": "SUCCESS",
            "defect_x": round(float(x), 6),
            "defect_y": round(float(y), 6),
            "defect_diameter": round(float(d), 6),
            "subset_type": subset_type,
            "max_von_mises_element_psi": round(float(result.max_element_von_mises_psi), 4),
            "max_von_mises_element_mpa": round(float(result.max_element_von_mises_psi * PSI_TO_MPA), 4),
            "max_von_mises_nodal_psi": round(float(result.max_von_mises_psi), 4),
            "max_von_mises_nodal_mpa": round(float(result.max_von_mises_psi * PSI_TO_MPA), 4),
            "max_principal_stress_psi": round(float(result.max_principal_stress_psi), 4),
            "max_displacement": float(result.max_displacement),
            "strain_energy": float(result.strain_energy),
            "applied_force_n": round(float(result.applied_force_total_n), 2),
            "reaction_force_y_n": round(float(result.reaction_force_bottom_n), 2),
            "reaction_force_x_n": round(float(result.horizontal_reaction_n), 6),
            "reaction_force_residual": float(result.equilibrium_error_n),
            "max_vm_center_psi": round(max_vm_center_psi, 4),
            "max_vm_defect_psi": round(max_vm_defect_psi, 4),
            "peak_stress_region": peak_stress_region,
            "node_count": int(result.num_nodes),
            "element_count": int(result.num_elements),
            "mesh_generation_time": round(float(result.mesh_generation_time), 4),
            "solve_time": round(float(result.solve_time), 4),
            "simulation_time": round(float(t_duration), 4),
            "solver_source": "scipy_sparse_cst_plane_stress",
            "mesh_resolution": f"adaptive_hd_D_over_12_hc0.003_hg0.02",
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


class DatasetPipeline3D:
    """
    Orchestrates the 3-parameter (X, Y, D) dataset generation.
    """
    def __init__(
        self,
        output_dir: str = "data",
        fields_dir: str = "data/fields",
        n_fixed: int = 2075,
        n_var: int = 8000,
        seed_fixed: int = 42,
        seed_var: int = 1042,
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

        self.summary_csv = self.output_dir / "fem_dataset_10075.csv"
        self.failed_csv = self.output_dir / "failed_simulations_3param.csv"
        self.parameters_csv = self.output_dir / "sampled_parameters_3param.csv"
        self.metadata_json = self.output_dir / "dataset_metadata.json"

        self.parameters, self.subset_types, self.rejections = self._init_parameters()

    def _init_parameters(self) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        if self.parameters_csv.exists():
            df_p = pd.read_csv(self.parameters_csv)
            if len(df_p) >= self.n_samples:
                logger.info(f"Loaded {len(df_p)} parameter sets from {self.parameters_csv}")
                params = df_p[["defect_x", "defect_y", "defect_diameter"]].to_numpy()[:self.n_samples]
                subtypes = df_p["subset_type"].tolist()[:self.n_samples]
                rejections = {"note": "loaded from cached parameter csv"}
                return params, subtypes, rejections

        logger.info(
            f"Generating 10,075 parameter sets: {self.n_fixed} fixed-D (seed {self.seed_fixed}) "
            f"+ {self.n_var} variable-3D (seed {self.seed_var})..."
        )
        params, subtypes, rejections = generate_3param_sobol_samples(
            n_fixed=self.n_fixed,
            n_var=self.n_var,
            seed_fixed=self.seed_fixed,
            seed_var=self.seed_var
        )

        df_p = pd.DataFrame({
            "simulation_id": np.arange(self.n_samples),
            "defect_x": params[:, 0],
            "defect_y": params[:, 1],
            "defect_diameter": params[:, 2],
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
            f"(Adaptive rule: h_defect = D / 12.0, h_center = {H_CENTER_PROD}, h_coarse = {H_COARSE_PROD})..."
        )

        success_records = []
        failed_records = []
        t0 = time.time()
        completed_count = len(completed_ids)

        write_header_summary = not self.summary_csv.exists()
        write_header_failed = not self.failed_csv.exists()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_id = {executor.submit(run_3param_simulation_worker, t): t[0] for t in tasks}

            for i, future in enumerate(as_completed(future_to_id), 1):
                res = future.result()
                sim_id = res["simulation_id"]

                if res["status"] == "SUCCESS":
                    del res["status"]
                    success_records.append(res)
                else:
                    failed_records.append(res)
                    logger.error(f"Simulation {sim_id} failed: {res.get('error')}")

                # Incremental flush
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

        # Flush any remaining records
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

        # Write metadata JSON
        meta = {
            "total_simulations": completed_count,
            "target_simulations": self.n_samples,
            "fixed_diameter_subset_count": self.n_fixed,
            "variable_3param_subset_count": self.n_var,
            "fixed_diameter_value": D_FIXED,
            "variable_diameter_range": [D_MIN, D_MAX],
            "adaptive_mesh_rule": "h_defect = D / 12.0",
            "central_hole_mesh_size": H_CENTER_PROD,
            "coarse_mesh_size": H_COARSE_PROD,
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
    parser = argparse.ArgumentParser(description="3-Parameter Automated FEM Dataset Pipeline")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit number of simulations (for smoke test)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no-fields", action="store_true", help="Disable saving per-simulation .npz field data")
    args = parser.parse_args()

    pipeline = DatasetPipeline3D(
        n_fixed=2075,
        n_var=8000,
        num_workers=args.workers
    )
    pipeline.run(
        batch_size=args.batch_size,
        save_fields=not args.no_fields
    )
