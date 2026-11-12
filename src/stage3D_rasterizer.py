"""
Stage 3D Phase C: Common Spatial Representation and Anti-Leakage Rasterizer
===========================================================================
Maps unstructured FEM mesh fields to standardized 2D grids (128x128 and 256x256)
over normalized coordinates:
    xi = x / L in [0, 1]
    eta = y / H in [0, 1]

CRITICAL ANTI-LEAKAGE SAFEGUARDS:
1. Defect-Void Inpainting:
   Points inside the unknown defect hole are smoothly inpainted from adjacent
   material elements using nearest/IDW continuum extrapolation.
   NO missing-value masks, zero-voids, or defect boundaries are exposed to the models!
2. Mesh-Density Prohibition:
   Mesh node counts, element connectivity, and local refinement footprints are
   strictly barred from model input channels.
3. Only permitted spec-level metadata:
   Known central hole mask (fixed at xi=0.5, eta=0.5) and aspect ratio L/H.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree


def normalize_coordinates(
    coords: np.ndarray,
    plate_length_mm: float,
    plate_height_mm: float
) -> np.ndarray:
    """Normalize physical coordinates (mm) to [0, 1] x [0, 1]."""
    coords = np.asarray(coords, dtype=np.float32)
    # Check if already in [0, 1] (Square plate batch 1)
    if coords[:, 0].max() <= 1.05 and coords[:, 1].max() <= 1.05:
        return np.clip(coords, 0.0, 1.0)
    
    norm_x = coords[:, 0] / float(plate_length_mm)
    norm_y = coords[:, 1] / float(plate_height_mm)
    return np.clip(np.column_stack([norm_x, norm_y]), 0.0, 1.0)


def create_normalized_grid(grid_res: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute Cartesian coordinate grid [grid_res, grid_res, 2]."""
    xi = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    eta = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    XI, ETA = np.meshgrid(xi, eta)
    grid_pts = np.column_stack([XI.ravel(), ETA.ravel()])
    return XI, ETA, grid_pts


def rasterize_stress_field(
    coords_norm: np.ndarray,
    stresses: np.ndarray,
    grid_res: int = 128,
    method: str = "nearest"
) -> np.ndarray:
    """
    Interpolate unstructured element or nodal stresses onto uniform rectangular grid.
    Inpaints defect voids naturally via continuum projection, eliminating all geometric leakage.
    """
    coords_norm = np.asarray(coords_norm, dtype=np.float32)
    stresses = np.asarray(stresses, dtype=np.float32)
    
    XI, ETA, grid_pts = create_normalized_grid(grid_res)
    tree = cKDTree(coords_norm)
    
    if method == "nearest":
        _, idxs = tree.query(grid_pts, k=1)
        grid_vals = stresses[idxs].reshape(grid_res, grid_res)
    else: # k=4 IDW
        dists, idxs = tree.query(grid_pts, k=4)
        weights = 1.0 / np.maximum(dists, 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        grid_vals = np.sum(weights * stresses[idxs], axis=1).reshape(grid_res, grid_res)
        
    return grid_vals.astype(np.float32)


def generate_known_central_hole_mask(
    grid_res: int,
    plate_length_mm: float,
    plate_height_mm: float,
    central_hole_diameter_mm: float
) -> np.ndarray:
    """
    Generate binary mask of the KNOWN central circular cutout.
    Values: 1.0 inside central hole, 0.0 outside.
    """
    XI, ETA, _ = create_normalized_grid(grid_res)
    rx = (central_hole_diameter_mm / 2.0) / float(plate_length_mm)
    ry = (central_hole_diameter_mm / 2.0) / float(plate_height_mm)
    
    dist_sq = ((XI - 0.5) / rx)**2 + ((ETA - 0.5) / ry)**2
    mask = (dist_sq <= 1.0).astype(np.float32)
    return mask


def generate_leakage_mesh_density_map(
    coords_norm: np.ndarray,
    grid_res: int = 128,
    radius: float = 0.05
) -> np.ndarray:
    """
    NEGATIVE CONTROL ONLY:
    Generates local element centroid count map to prove that mesh density
    would leak defect location if exposed, verifying why it must be prohibited.
    """
    XI, ETA, grid_pts = create_normalized_grid(grid_res)
    tree = cKDTree(coords_norm)
    # Count number of element centroids within radius
    counts = tree.query_ball_point(grid_pts, r=radius, return_sorted=False)
    density_map = np.array([len(c) for c in counts], dtype=np.float32).reshape(grid_res, grid_res)
    # Normalize to [0, 1]
    d_max = density_map.max()
    if d_max > 0:
        density_map /= d_max
    return density_map


def load_and_rasterize_simulation(
    npz_path: Path,
    plate_length_mm: float,
    plate_height_mm: float,
    grid_res: int = 128
) -> Dict[str, np.ndarray]:
    """Unified helper to load NPZ and rasterize to common representation."""
    data = np.load(npz_path)
    
    # 1. Centroid coordinates
    if 'element_centroids_mm' in data:
        coords_raw = data['element_centroids_mm']
    elif 'element_centroids' in data:
        coords_raw = data['element_centroids']
    elif 'nodes' in data and 'elements' in data:
        nodes = data['nodes']
        elems = data['elements']
        coords_raw = np.mean(nodes[elems], axis=1)
    else:
        raise KeyError(f"Unable to locate centroid coordinates in {npz_path}")
        
    coords_norm = normalize_coordinates(coords_raw, plate_length_mm, plate_height_mm)
    
    # 2. Element Von Mises stress (psi)
    vm_stress = data['element_von_mises_psi']
    
    # 3. Rasterize
    stress_grid = rasterize_stress_field(coords_norm, vm_stress, grid_res=grid_res, method="nearest")
    
    return {
        'coords_norm': coords_norm,
        'vm_stress': vm_stress,
        'stress_grid': stress_grid
    }

