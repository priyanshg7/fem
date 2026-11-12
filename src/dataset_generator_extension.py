"""
Automated Parametric FEM Simulation and Dataset Generator for Stage 2S Multi-Geometry Extension.

Supported Geometries:
- RECT_700: 1000 mm x 700 mm (Aspect Ratio = 1.4286)
- RECT_800: 1000 mm x 800 mm (Aspect Ratio = 1.2500)
- RECT_900: 1000 mm x 900 mm (Aspect Ratio = 1.1111)

Generates 10,075 unique, valid FEM simulations per geometry (30,225 total for Stage 2S):
- 2,075 fixed comparison subset (D = H / 80)
- 8,000 variable subset (D in [H / 160, H / 40])
- Clearance: 1% of plate height (g = 0.01 * H)
- Adaptive meshing: h_defect = D / 12.0, h_center = 3 mm, h_coarse = 20 mm
- Compressed .npz full spatial fields stored in data/geometry_extension/{GEOM}/fields/
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
logger = logging.getLogger("DatasetGeneratorStage2S")

PLATE_WIDTH_M = 1.0  # 1000 mm along X
PSI_TO_MPA = 0.006894757293168361
PA_TO_PSI = 1.4503773773020923e-4

GEOMETRY_CONFIGS = {
    "RECT_700": {
        "plate_length_m": 0.700,
        "height_mm": 700.0,
        "aspect_ratio": 1.0 / 0.700,
        "center_hole_x_m": 0.500,
        "center_hole_y_m": 0.350,
        "center_hole_diameter_m": 0.700 / 12.0,  # 58.333 mm
        "clearance_margin_m": 0.007,             # 7 mm (1% H)
        "d_fixed_m": 0.700 / 80.0,               # 8.750 mm (H/80)
        "d_min_m": 0.700 / 160.0,                # 4.375 mm (H/160)
        "d_max_m": 0.700 / 40.0,                 # 17.500 mm (H/40)
        "seed_fixed": 742,
        "seed_var": 10742,
        "fixed_subset_label": "fixed_d_8_75mm",
    },
    "RECT_800": {
        "plate_length_m": 0.800,
        "height_mm": 800.0,
        "aspect_ratio": 1.0 / 0.800,
        "center_hole_x_m": 0.500,
        "center_hole_y_m": 0.400,
        "center_hole_diameter_m": 0.800 / 12.0,  # 66.667 mm
        "clearance_margin_m": 0.008,             # 8 mm (1% H)
        "d_fixed_m": 0.800 / 80.0,               # 10.000 mm (H/80)
        "d_min_m": 0.800 / 160.0,                # 5.000 mm (H/160)
        "d_max_m": 0.800 / 40.0,                 # 20.000 mm (H/40)
        "seed_fixed": 842,
        "seed_var": 10842,
        "fixed_subset_label": "fixed_d_10mm",
    },
    "RECT_900": {
        "plate_length_m": 0.900,
        "height_mm": 900.0,
        "aspect_ratio": 1.0 / 0.900,
        "center_hole_x_m": 0.500,
        "center_hole_y_m": 0.450,
        "center_hole_diameter_m": 0.900 / 12.0,  # 75.000 mm
        "clearance_margin_m": 0.009,             # 9 mm (1% H)
        "d_fixed_m": 0.900 / 80.0,               # 11.250 mm (H/80)
        "d_min_m": 0.900 / 160.0,                # 5.625 mm (H/160)
        "d_max_m": 0.900 / 40.0,                 # 22.500 mm (H/40)
        "seed_fixed": 942,
        "seed_var": 10942,
        "fixed_subset_label": "fixed_d_11_25mm",
    },
}


def sample_extension_sobol(
    geom_id: str,
    n_fixed: int = 2075,
    n_var: int = 8000
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Deterministic Quasi-Monte Carlo space-filling sampling for a specified geometry.
    """
    gcfg = GEOMETRY_CONFIGS[geom_id]
    plate_w = PLATE_WIDTH_M
    plate_h = gcfg["plate_length_m"]
    xc = gcfg["center_hole_x_m"]
    yc = gcfg["center_hole_y_m"]
    rc = gcfg["center_hole_diameter_m"] / 2.0
    g = gcfg["clearance_margin_m"]
    d_fixed = gcfg["d_fixed_m"]
    d_min = gcfg["d_min_m"]
    d_max = gcfg["d_max_m"]

    # 1. Fixed Comparison Subset (D = H / 80)
    rd_f = d_fixed / 2.0
    x_min_f, x_max_f = rd_f + g, plate_w - rd_f - g
    y_min_f, y_max_f = rd_f + g, plate_h - rd_f - g
    r_excl_f = rc + rd_f + g

    sampler_f = qmc.Sobol(d=2, scramble=True, seed=gcfg["seed_fixed"])
    raw_f = sampler_f.random_base2(m=13)  # 8192
    pts_f = np.empty_like(raw_f)
    pts_f[:, 0] = x_min_f + raw_f[:, 0] * (x_max_f - x_min_f)
    pts_f[:, 1] = y_min_f + raw_f[:, 1] * (y_max_f - y_min_f)
    dist_f = np.hypot(pts_f[:, 0] - xc, pts_f[:, 1] - yc)
    valid_mask_f = dist_f >= r_excl_f
    valid_f = pts_f[valid_mask_f][:n_fixed]

    if len(valid_f) < n_fixed:
        raise RuntimeError(f"Insufficient valid fixed-D samples for {geom_id}")

    fixed_data = np.hstack([valid_f, np.full((n_fixed, 1), d_fixed)])
    fixed_types = [gcfg["fixed_subset_label"]] * n_fixed
    rej_f = len(raw_f) - np.sum(valid_mask_f)

    # 2. Variable 3-Parameter Subset (D in [H/160, H/40])
    sampler_v = qmc.Sobol(d=3, scramble=True, seed=gcfg["seed_var"])
    raw_v = sampler_v.random_base2(m=15)  # 32768

    d_candidates = d_min + raw_v[:, 2] * (d_max - d_min)
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
        raise RuntimeError(f"Insufficient valid variable-3D samples for {geom_id}")

    var_data = np.column_stack([valid_v_x, valid_v_y, valid_v_d])
    var_types = ["variable_3param"] * n_var
    rej_v = len(raw_v) - np.sum(valid_mask_v)

    total_data = np.vstack([fixed_data, var_data])
    total_types = fixed_types + var_types

    rejections_dict = {
        "fixed_subset_rejections": int(rej_f),
        "variable_subset_rejections": int(rej_v),
        "total_candidates_drawn": int(len(raw_f) + len(raw_v)),
        "total_accepted": int(len(total_data)),
        "seed_fixed": gcfg["seed_fixed"],
        "seed_var": gcfg["seed_var"]
    }

    return total_data, total_types, rejections_dict


def run_extension_simulation_worker(task_args: Tuple) -> Dict[str, Any]:
    """
    Worker function executed in parallel across CPU cores.
    
    Arguments:
        (geom_id, sim_id, x, y, d, subset_type, save_fields, fields_dir)
    """
    (geom_id, sim_id, x, y, d, subset_type, save_fields, fields_dir) = task_args
    gcfg = GEOMETRY_CONFIGS[geom_id]
    t_start = time.perf_counter()

    try:
        h_defect = float(d / 12.0)
        h_center = 0.0030
        h_coarse = 0.0200

        geom_config = PlateGeometryConfig(
            plate_width=PLATE_WIDTH_M,
            plate_length=gcfg["plate_length_m"],
            center_hole_x=gcfg["center_hole_x_m"],
            center_hole_y=gcfg["center_hole_y_m"],
            center_hole_diameter=gcfg["center_hole_diameter_m"],
            has_defect=True,
            defect_hole_diameter=float(d),
            defect_hole_x=float(x),
            defect_hole_y=float(y),
            h_coarse=h_coarse,
            h_center_hole=h_center,
            h_defect_hole=h_defect,
            clearance_margin=gcfg["clearance_margin_m"]
        )

        if not validate_defect_position(geom_config, margin=gcfg["clearance_margin_m"]):
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

        rc = gcfg["center_hole_diameter_m"] / 2.0
        rd = d / 2.0
        dc = np.hypot(centroids[:, 0] - gcfg["center_hole_x_m"], centroids[:, 1] - gcfg["center_hole_y_m"])
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

        # Scale-invariant metrics
        p_applied_psi = mat_config.applied_traction_psi      # 100.0 psi
        p_applied_pa = mat_config.applied_traction_pa        # 689,475.7 Pa
        
        normalized_max_stress_elem = result.max_element_von_mises_psi / p_applied_psi
        normalized_max_stress_nodal = result.max_von_mises_psi / p_applied_psi
        normalized_displacement = result.max_displacement / p_applied_pa
        normalized_strain_energy = result.strain_energy / (p_applied_pa ** 2)

        # Save compressed full-field data
        field_rel_path = f"data/geometry_extension/{geom_id}/fields/sim_{sim_id:05d}.npz"
        if save_fields and fields_dir is not None:
            field_abs_path = Path(fields_dir) / f"sim_{sim_id:05d}.npz"
            np.savez_compressed(
                field_abs_path,
                sim_id=np.int32(sim_id),
                geometry_family=geom_id,
                plate_length_mm=np.float32(PLATE_WIDTH_M * 1000.0),
                plate_height_mm=np.float32(gcfg["plate_length_m"] * 1000.0),
                defect_coords_mm=np.array([x * 1000.0, y * 1000.0], dtype=np.float32),
                defect_diameter_mm=np.float32(d * 1000.0),
                subset_type=subset_type,
                element_centroids_mm=(centroids * 1000.0).astype(np.float32),
                element_von_mises_psi=vm_elem_psi.astype(np.float32),
                nodal_coords_mm=(result.nodes * 1000.0).astype(np.float32),
                nodal_von_mises_psi=(result.nodal_von_mises * PA_TO_PSI).astype(np.float32),
                displacements_m=result.displacements.astype(np.float32)
            )

        # Summary record
        record = {
            "simulation_id": int(sim_id),
            "status": "SUCCESS",
            "geometry_family": geom_id,
            "plate_length_mm": 1000.0,
            "plate_height_mm": float(gcfg["height_mm"]),
            "aspect_ratio": round(float(gcfg["aspect_ratio"]), 6),
            "central_hole_center_x_mm": float(gcfg["center_hole_x_m"] * 1000.0),
            "central_hole_center_y_mm": float(gcfg["center_hole_y_m"] * 1000.0),
            "central_hole_diameter_mm": round(float(gcfg["center_hole_diameter_m"] * 1000.0), 4),
            "defect_center_x_mm": round(float(x * 1000.0), 4),
            "defect_center_y_mm": round(float(y * 1000.0), 4),
            "defect_diameter_mm": round(float(d * 1000.0), 4),
            "normalized_defect_x": round(float(x / PLATE_WIDTH_M), 6),
            "normalized_defect_y": round(float(y / gcfg["plate_length_m"]), 6),
            "normalized_defect_diameter": round(float(d / gcfg["plate_length_m"]), 6),
            "normalized_central_hole_diameter": round(float(gcfg["center_hole_diameter_m"] / gcfg["plate_length_m"]), 6),
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
            "random_seed": int(gcfg["seed_fixed"]),
            "field_file": field_rel_path
        }
        return record

    except Exception as e:
        return {
            "simulation_id": sim_id,
            "status": "FAILED",
            "error": str(e)
        }


class ExtensionDatasetPipeline:
    """
    Orchestrates dataset generation for a specific Stage 2S geometry.
    """
    def __init__(
        self,
        geom_id: str,
        base_dir: str = "data/geometry_extension",
        n_fixed: int = 2075,
        n_var: int = 8000,
        num_workers: Optional[int] = None
    ):
        if geom_id not in GEOMETRY_CONFIGS:
            raise ValueError(f"Unknown geometry {geom_id}. Must be one of {list(GEOMETRY_CONFIGS.keys())}")

        self.geom_id = geom_id
        self.output_dir = Path(base_dir) / geom_id
        self.fields_dir = self.output_dir / "fields"
        self.n_fixed = n_fixed
        self.n_var = n_var
        self.n_samples = n_fixed + n_var
        self.num_workers = num_workers or min(os.cpu_count() or 4, 12)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fields_dir.mkdir(parents=True, exist_ok=True)

        self.summary_csv = self.output_dir / f"fem_dataset_10075_{geom_id}.csv"
        self.failed_csv = self.output_dir / f"failed_simulations_{geom_id}.csv"
        self.parameters_csv = self.output_dir / f"sampled_parameters_{geom_id}.csv"
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

        logger.info(f"Sampling {self.n_samples} parameter sets for {self.geom_id}...")
        params, subtypes, rejections = sample_extension_sobol(
            self.geom_id,
            n_fixed=self.n_fixed,
            n_var=self.n_var
        )

        gcfg = GEOMETRY_CONFIGS[self.geom_id]
        df_p = pd.DataFrame({
            "simulation_id": np.arange(self.n_samples),
            "geometry_family": self.geom_id,
            "defect_x_m": params[:, 0],
            "defect_y_m": params[:, 1],
            "defect_diameter_m": params[:, 2],
            "defect_x_mm": params[:, 0] * 1000.0,
            "defect_y_mm": params[:, 1] * 1000.0,
            "defect_diameter_mm": params[:, 2] * 1000.0,
            "normalized_defect_x": params[:, 0] / PLATE_WIDTH_M,
            "normalized_defect_y": params[:, 1] / gcfg["plate_length_m"],
            "normalized_defect_diameter": params[:, 2] / gcfg["plate_length_m"],
            "subset_type": subtypes
        })
        df_p.to_csv(self.parameters_csv, index=False)
        logger.info(f"Saved {self.n_samples} parameters to {self.parameters_csv}")
        return params, subtypes, rejections

    def get_completed_ids(self) -> set:
        if not self.summary_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.summary_csv, usecols=["simulation_id"])
            return set(df["simulation_id"].tolist())
        except Exception as e:
            logger.warning(f"Error reading summary {self.summary_csv}: {e}")
            return set()

    def run(
        self,
        batch_size: Optional[int] = None,
        save_fields: bool = True,
        save_interval: int = 50
    ):
        completed_ids = self.get_completed_ids()
        logger.info(f"[{self.geom_id}] Checkpoint check: {len(completed_ids)} / {self.n_samples} already completed.")

        total_to_run = self.n_samples if batch_size is None else min(batch_size, self.n_samples)
        tasks = []
        for sim_id in range(total_to_run):
            if sim_id not in completed_ids:
                x, y, d = self.parameters[sim_id]
                subtype = self.subset_types[sim_id]
                tasks.append((
                    self.geom_id, sim_id, x, y, d, subtype,
                    save_fields, str(self.fields_dir)
                ))

        if not tasks:
            logger.info(f"[{self.geom_id}] All requested simulations have already been successfully completed!")
            return

        logger.info(f"[{self.geom_id}] Launching {len(tasks)} simulations using {self.num_workers} worker processes...")

        success_records = []
        failed_records = []
        t0 = time.time()
        completed_count = len(completed_ids)

        write_header_summary = not self.summary_csv.exists()
        write_header_failed = not self.failed_csv.exists()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_id = {executor.submit(run_extension_simulation_worker, t): t[1] for t in tasks}

            for i, future in enumerate(as_completed(future_to_id), 1):
                res = future.result()
                sim_id = res["simulation_id"]

                if res["status"] == "SUCCESS":
                    del res["status"]
                    success_records.append(res)
                else:
                    failed_records.append(res)
                    logger.error(f"[{self.geom_id}] Simulation {sim_id} failed: {res.get('error')}")

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
                        f"[{self.geom_id}] Progress: {completed_count}/{self.n_samples} ({i}/{len(tasks)} this run) | "
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
            f"[{self.geom_id}] Run completed. Processed {len(tasks)} tasks in {total_elapsed:.1f}s "
            f"({len(tasks)/total_elapsed:.2f} sims/s). Total successful: {completed_count}."
        )

        gcfg = GEOMETRY_CONFIGS[self.geom_id]
        meta = {
            "geometry_family": self.geom_id,
            "plate_length_mm": 1000.0,
            "plate_height_mm": gcfg["height_mm"],
            "aspect_ratio": gcfg["aspect_ratio"],
            "central_hole_center_mm": [gcfg["center_hole_x_m"] * 1000.0, gcfg["center_hole_y_m"] * 1000.0],
            "central_hole_diameter_mm": gcfg["center_hole_diameter_m"] * 1000.0,
            "clearance_margin_mm": gcfg["clearance_margin_m"] * 1000.0,
            "total_simulations": completed_count,
            "target_simulations": self.n_samples,
            "fixed_diameter_subset_count": self.n_fixed,
            "variable_3param_subset_count": self.n_var,
            "fixed_diameter_value_mm": gcfg["d_fixed_m"] * 1000.0,
            "variable_diameter_range_mm": [gcfg["d_min_m"] * 1000.0, gcfg["d_max_m"] * 1000.0],
            "adaptive_mesh_rule": "h_defect = D / 12.0, h_center = 3mm, h_coarse = 20mm",
            "solver": "scipy_sparse_cst_plane_stress",
            "seed_fixed": gcfg["seed_fixed"],
            "seed_var": gcfg["seed_var"],
            "rejections": self.rejections,
            "summary_csv": str(self.summary_csv),
            "fields_dir": str(self.fields_dir)
        }
        with open(self.metadata_json, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2S Multi-Geometry Dataset Pipeline")
    parser.add_argument("--geom", type=str, choices=["RECT_700", "RECT_800", "RECT_900", "ALL"], default="ALL", help="Geometry family to run")
    parser.add_argument("--batch-size", type=int, default=None, help="Limit number of simulations (e.g. 100 for smoke test)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--no-fields", action="store_true", help="Disable saving per-simulation .npz field data")
    args = parser.parse_args()

    geoms_to_run = ["RECT_700", "RECT_800", "RECT_900"] if args.geom == "ALL" else [args.geom]

    for gid in geoms_to_run:
        logger.info(f"\n{'='*30} STARTING PIPELINE: {gid} {'='*30}")
        pipeline = ExtensionDatasetPipeline(
            geom_id=gid,
            num_workers=args.workers
        )
        pipeline.run(
            batch_size=args.batch_size,
            save_fields=not args.no_fields
        )

