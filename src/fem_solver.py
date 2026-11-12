"""
Finite Element Elasticity Solver for 2D Plane-Stress Analysis.

Supports:
- Direct solution using Continuum Plane-Stress Formulation with Gmsh Triangular Mesh
- Boundary Conditions matching experimental setup of Singh & Pamnani (2025):
  * Top boundary (y=1.0): Uniaxial tensile traction P = 100 psi (689,475.7 Pa)
  * Bottom boundary (y=0.0): Roller support (v = u_y = 0)
  * Rigid body motion prevention: Midpoint bottom node pinned in x (u = u_x = 0)
  * Traction-free interior hole boundaries and lateral vertical sides
- Computation of displacement, strain tensor, Cauchy stress tensor, and Von Mises stress.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.geometry import PlateGeometryConfig, generate_mesh


# Conversion constants
PSI_TO_PA = 6894.757293168361
PA_TO_PSI = 1.0 / PSI_TO_PA


@dataclass
class MaterialConfig:
    youngs_modulus_psi: float = 1e7          # E = 10^7 psi (from paper Table 2)
    poisson_ratio: float = 0.3               # mu / nu = 0.3 (from paper Table 2)
    applied_traction_psi: float = 100.0      # P = 100 psi (from paper Section 2.4)
    thickness: float = 1.0                   # Unit thickness in plane stress

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_psi * PSI_TO_PA

    @property
    def applied_traction_pa(self) -> float:
        return self.applied_traction_psi * PSI_TO_PA


@dataclass
class FEMResult:
    nodes: np.ndarray                        # (N, 2) array of nodal coords [x, y]
    elements: np.ndarray                     # (M, 3) array of element triangle node indices
    displacements: np.ndarray                # (N, 2) array of [u_x, u_y] at each node
    element_stresses: np.ndarray             # (M, 3) array of [sigma_xx, sigma_yy, tau_xy] (in Pa)
    element_von_mises: np.ndarray            # (M,) array of element von Mises stress (in Pa)
    nodal_stresses: np.ndarray               # (N, 3) array of recovered nodal [sigma_xx, sigma_yy, tau_xy] (in Pa)
    nodal_von_mises: np.ndarray              # (N,) array of nodal von Mises stress (in Pa)
    max_von_mises_pa: float                  # Maximum recovered nodal von Mises stress in Pa
    max_von_mises_psi: float                 # Maximum recovered nodal von Mises stress in psi
    max_element_von_mises_pa: float          # Maximum element centroid von Mises stress in Pa
    max_element_von_mises_psi: float         # Maximum element centroid von Mises stress in psi
    max_principal_stress_pa: float           # Maximum principal stress (sigma_1) in Pa
    max_principal_stress_psi: float          # Maximum principal stress (sigma_1) in psi
    max_stress_location: Tuple[float, float] # Coordinates (x, y) of peak nodal stress
    max_element_location: Tuple[float, float]# Centroid (x, y) of peak element stress
    max_displacement: float                  # Maximum resultant displacement magnitude (m)
    strain_energy: float                     # Total strain energy in the plate (J)
    num_nodes: int
    num_elements: int
    applied_force_total_n: float             # Total applied vertical tensile force (N)
    applied_force_total_lbf: float           # Total applied vertical tensile force (lbf)
    reaction_force_bottom_n: float           # Total vertical reaction force at bottom support (N)
    horizontal_reaction_n: float             # Total horizontal reaction at pin constraint (N)
    equilibrium_error_n: float               # Net vertical force imbalance |F_top + R_bottom| (N)
    mesh_generation_time: float = 0.0        # Time taken for Gmsh CAD & meshing (s)
    solve_time: float = 0.0                  # Time taken for stiffness assembly & direct solve (s)


def constitutive_matrix_plane_stress(E: float, nu: float) -> np.ndarray:
    """
    Constitutive matrix D for isotropic plane stress condition.
    sigma = D * epsilon
    sigma = [sigma_xx, sigma_yy, tau_xy]^T
    epsilon = [eps_xx, eps_yy, gamma_xy]^T
    """
    factor = E / (1.0 - nu**2)
    D = factor * np.array([
        [1.0,  nu,        0.0],
        [nu,  1.0,        0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=np.float64)
    return D


def solve_plane_stress_fem(
    geom_config: PlateGeometryConfig,
    mat_config: MaterialConfig,
    verbose: bool = False
) -> FEMResult:
    """
    Execute high-fidelity 2D plane-stress finite element simulation on the plate with defect.
    
    Uses consistent SI units internally for the linear elasticity solve, then outputs both
    SI (Pa, m) and conventional engineering units (psi, in) for direct comparison with the paper.
    """
    # 1. Generate refined Gmsh mesh
    t_mesh_start = time.perf_counter()
    nodes, elements, tags = generate_mesh(geom_config, verbose=verbose)
    t_mesh = time.perf_counter() - t_mesh_start

    t_solve_start = time.perf_counter()
    num_nodes = len(nodes)
    num_elements = len(elements)
    num_dofs = 2 * num_nodes

    if verbose:
        print(f"Generated Mesh: {num_nodes} nodes, {num_elements} elements, {num_dofs} DOFs.")

    # Material properties in SI units
    E = mat_config.youngs_modulus_pa
    nu = mat_config.poisson_ratio
    t = mat_config.thickness
    traction_y = mat_config.applied_traction_pa
    D = constitutive_matrix_plane_stress(E, nu)

    # 2. Assemble Global Stiffness Matrix
    # Pre-allocate sparse matrix COO arrays
    # Each CST element has 6 DOFs -> 36 entries in element stiffness matrix
    entries_per_elem = 36
    total_entries = num_elements * entries_per_elem
    rows = np.empty(total_entries, dtype=np.int32)
    cols = np.empty(total_entries, dtype=np.int32)
    vals = np.empty(total_entries, dtype=np.float64)

    elem_areas = np.empty(num_elements, dtype=np.float64)
    elem_B = np.empty((num_elements, 3, 6), dtype=np.float64)

    entry_idx = 0
    for e in range(num_elements):
        elem_node_ids = elements[e]
        x1, y1 = nodes[elem_node_ids[0]]
        x2, y2 = nodes[elem_node_ids[1]]
        x3, y3 = nodes[elem_node_ids[2]]

        # Area = 0.5 * det([1, x1, y1; 1, x2, y2; 1, x3, y3])
        two_area = (x2 * y3 - x3 * y2) - (x1 * y3 - x3 * y1) + (x1 * y2 - x2 * y1)
        
        # Enforce counter-clockwise node ordering to guarantee positive Jacobian
        if two_area < 0:
            elements[e] = elem_node_ids[[0, 2, 1]]
            elem_node_ids = elements[e]
            x2, y2, x3, y3 = x3, y3, x2, y2
            two_area = -two_area

        area = 0.5 * two_area
        elem_areas[e] = area

        if area <= 1e-14:
            raise ValueError(f"Degenerate triangle element detected at element index {e}")

        # Shape function spatial derivatives
        # N_i = (a_i + b_i*x + c_i*y) / (2A)
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2

        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        # Strain-displacement matrix B (3 x 6)
        inv_two_area = 1.0 / two_area
        B = inv_two_area * np.array([
            [b1, 0,  b2, 0,  b3, 0],
            [0,  c1, 0,  c2, 0,  c3],
            [c1, b1, c2, b2, c3, b3]
        ], dtype=np.float64)
        elem_B[e] = B

        # Element stiffness: K_e = t * area * (B^T * D * B)
        # Note: B has inv_two_area factor. Integral over area gives:
        # B^T * D * B * area * t
        Ke = t * area * (B.T @ D @ B)

        # Map to global DOFs: node i -> (2*i, 2*i + 1)
        elem_dofs = np.array([
            2 * elem_node_ids[0], 2 * elem_node_ids[0] + 1,
            2 * elem_node_ids[1], 2 * elem_node_ids[1] + 1,
            2 * elem_node_ids[2], 2 * elem_node_ids[2] + 1
        ], dtype=np.int32)

        # Meshgrid of indices
        I_mat = np.repeat(elem_dofs, 6)
        J_mat = np.tile(elem_dofs, 6)

        rows[entry_idx : entry_idx + entries_per_elem] = I_mat
        cols[entry_idx : entry_idx + entries_per_elem] = J_mat
        vals[entry_idx : entry_idx + entries_per_elem] = Ke.ravel()
        entry_idx += entries_per_elem

    K_global = sp.coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    # 3. Assemble Global Force Vector (Neumann Boundary Conditions)
    # Applied tensile traction on top edge (y = plate_length)
    F_global = np.zeros(num_dofs, dtype=np.float64)
    top_y = geom_config.plate_length
    eps = 1e-5

    # Find element edges lying on top boundary
    top_force_total = 0.0
    for e in range(num_elements):
        n_ids = elements[e]
        coords = nodes[n_ids]
        
        # Check all 3 edges: (0,1), (1,2), (2,0)
        edges = [(0, 1), (1, 2), (2, 0)]
        for i1, i2 in edges:
            if abs(coords[i1, 1] - top_y) < eps and abs(coords[i2, 1] - top_y) < eps:
                # This edge is on the top loaded boundary
                edge_len = abs(coords[i2, 0] - coords[i1, 0])
                # Distributed traction: consistent nodal load is (traction_y * edge_len * t) / 2 per node
                f_y_nodal = 0.5 * traction_y * edge_len * t
                dof1 = 2 * n_ids[i1] + 1
                dof2 = 2 * n_ids[i2] + 1
                F_global[dof1] += f_y_nodal
                F_global[dof2] += f_y_nodal
                top_force_total += 2.0 * f_y_nodal

    # 4. Apply Dirichlet Boundary Conditions
    # Bottom boundary (y = 0): roller support (u_y = 0)
    # Center-bottom node: pinned in x (u_x = 0) to eliminate rigid body translation
    bottom_y = 0.0
    fixed_dofs = []
    fixed_vals = []

    # Find all nodes on bottom boundary
    bottom_nodes = np.where(abs(nodes[:, 1] - bottom_y) < eps)[0]
    for n in bottom_nodes:
        # Constrain v (u_y) = 0
        fixed_dofs.append(2 * n + 1)
        fixed_vals.append(0.0)

    # Pin u_x = 0 at node closest to bottom midpoint (plate_width / 2.0, 0.0)
    mid_bottom_x = geom_config.plate_width / 2.0
    center_bottom_node = bottom_nodes[np.argmin(abs(nodes[bottom_nodes, 0] - mid_bottom_x))]
    fixed_dofs.append(2 * center_bottom_node)
    fixed_vals.append(0.0)

    fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
    fixed_vals = np.array(fixed_vals, dtype=np.float64)

    # Partition system into free and prescribed DOFs
    all_dofs = np.arange(num_dofs, dtype=np.int32)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Solve reduced linear system K_ff * u_f = F_f - K_fp * u_p
    K_ff = K_global[free_dofs, :][:, free_dofs]
    F_f = F_global[free_dofs]
    
    if len(fixed_dofs) > 0 and np.any(fixed_vals != 0.0):
        K_fp = K_global[free_dofs, :][:, fixed_dofs]
        rhs = F_f - K_fp.dot(fixed_vals)
    else:
        rhs = F_f

    # Direct solve using SuperLU
    u_f = spla.spsolve(K_ff, rhs)

    # Reconstruct full displacement vector
    U_full = np.zeros(num_dofs, dtype=np.float64)
    U_full[free_dofs] = u_f
    U_full[fixed_dofs] = fixed_vals

    displacements = U_full.reshape((num_nodes, 2))

    # 5. Compute Element Strains and Cauchy Stresses
    element_stresses = np.empty((num_elements, 3), dtype=np.float64)
    element_von_mises = np.empty(num_elements, dtype=np.float64)

    for e in range(num_elements):
        n_ids = elements[e]
        u_elem = displacements[n_ids].ravel()
        B = elem_B[e]
        
        # Strain: epsilon = B * u_elem  ([eps_xx, eps_yy, gamma_xy])
        eps_elem = B @ u_elem
        
        # Stress: sigma = D * eps_elem  ([sigma_xx, sigma_yy, tau_xy])
        sig_elem = D @ eps_elem
        element_stresses[e] = sig_elem

        # Von Mises stress (Plane Stress):
        # sigma_vM = sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*tau_xy^2)
        s_xx, s_yy, t_xy = sig_elem[0], sig_elem[1], sig_elem[2]
        vM = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * t_xy**2)
        element_von_mises[e] = vM

    # 6. Superconvergent Patch Recovery for Nodal Stresses
    # Area-weighted averaging from adjacent elements to nodes
    nodal_stresses = np.zeros((num_nodes, 3), dtype=np.float64)
    nodal_weights = np.zeros(num_nodes, dtype=np.float64)

    for e in range(num_elements):
        w = elem_areas[e]
        sig = element_stresses[e]
        for n_id in elements[e]:
            nodal_stresses[n_id] += w * sig
            nodal_weights[n_id] += w

    valid_mask = nodal_weights > 0
    nodal_stresses[valid_mask] /= nodal_weights[valid_mask, np.newaxis]

    # Nodal von Mises stress
    ns_xx = nodal_stresses[:, 0]
    ns_yy = nodal_stresses[:, 1]
    nt_xy = nodal_stresses[:, 2]
    nodal_von_mises = np.sqrt(ns_xx**2 - ns_xx * ns_yy + ns_yy**2 + 3.0 * nt_xy**2)

    # Principal stresses: sigma_1 = 0.5*(s_xx + s_yy) + sqrt((0.5*(s_xx - s_yy))**2 + tau_xy**2)
    nodal_sigma_1 = 0.5 * (ns_xx + ns_yy) + np.sqrt((0.5 * (ns_xx - ns_yy))**2 + nt_xy**2)
    max_p1_pa = float(np.max(nodal_sigma_1))
    max_p1_psi = float(max_p1_pa * PA_TO_PSI)

    max_vm_pa = float(np.max(nodal_von_mises))
    max_vm_psi = float(max_vm_pa * PA_TO_PSI)
    max_node_idx = int(np.argmax(nodal_von_mises))
    max_stress_loc = (float(nodes[max_node_idx, 0]), float(nodes[max_node_idx, 1]))

    # Element-level peak stress and centroid
    max_elem_vm_pa = float(np.max(element_von_mises))
    max_elem_vm_psi = float(max_elem_vm_pa * PA_TO_PSI)
    max_elem_idx = int(np.argmax(element_von_mises))
    elem_centroids = np.mean(nodes[elements], axis=1)
    max_elem_loc = (float(elem_centroids[max_elem_idx, 0]), float(elem_centroids[max_elem_idx, 1]))

    max_disp = float(np.max(np.linalg.norm(displacements, axis=1)))
    strain_energy = float(0.5 * U_full.dot(K_global.dot(U_full)))

    # Global equilibrium and reaction force calculation: R = K * U - F
    R_internal = K_global.dot(U_full)
    bottom_reaction_n = float(np.sum(R_internal[2 * bottom_nodes + 1]))
    pin_reaction_x = float(R_internal[2 * center_bottom_node])
    eq_error = float(abs(top_force_total + bottom_reaction_n))

    total_force_n = float(top_force_total)
    total_force_lbf = float(total_force_n * 0.224808943)
    t_solve = time.perf_counter() - t_solve_start

    return FEMResult(
        nodes=nodes,
        elements=elements,
        displacements=displacements,
        element_stresses=element_stresses,
        element_von_mises=element_von_mises,
        nodal_stresses=nodal_stresses,
        nodal_von_mises=nodal_von_mises,
        max_von_mises_pa=max_vm_pa,
        max_von_mises_psi=max_vm_psi,
        max_element_von_mises_pa=max_elem_vm_pa,
        max_element_von_mises_psi=max_elem_vm_psi,
        max_principal_stress_pa=max_p1_pa,
        max_principal_stress_psi=max_p1_psi,
        max_stress_location=max_stress_loc,
        max_element_location=max_elem_loc,
        max_displacement=max_disp,
        strain_energy=strain_energy,
        num_nodes=num_nodes,
        num_elements=num_elements,
        applied_force_total_n=total_force_n,
        applied_force_total_lbf=total_force_lbf,
        reaction_force_bottom_n=bottom_reaction_n,
        horizontal_reaction_n=pin_reaction_x,
        equilibrium_error_n=eq_error,
        mesh_generation_time=t_mesh,
        solve_time=t_solve
    )


if __name__ == "__main__":
    geom = PlateGeometryConfig(defect_hole_x=0.35, defect_hole_y=0.65)
    mat = MaterialConfig()
    result = solve_plane_stress_fem(geom, mat, verbose=True)
    print("=" * 60)
    print("FEM SIMULATION RESULT SUMMARY:")
    print(f"Max Von Mises Stress: {result.max_von_mises_psi:.2f} psi ({result.max_von_mises_pa / 1e6:.4f} MPa)")
    print(f"Max Resultant Displacement: {result.max_displacement:.6e} m")
    print(f"Total Applied Force: {result.applied_force_total_lbf:.2f} lbf ({result.applied_force_total_n:.2f} N)")
    print(f"Total Strain Energy: {result.strain_energy:.6e} J")
    print("=" * 60)

