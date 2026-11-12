"""
Unit and Physics Validation Tests for FEM Plate Simulation Pipeline.
Includes regression checks for central-hole Kirsch benchmark, equilibrium,
and Poisson lateral contraction.
"""

import unittest
import numpy as np

from src.geometry import PlateGeometryConfig, validate_defect_position, generate_mesh
from src.fem_solver import MaterialConfig, solve_plane_stress_fem, PSI_TO_PA, PA_TO_PSI


class TestFEMSimulation(unittest.TestCase):
    def setUp(self):
        self.geom_cfg = PlateGeometryConfig(
            defect_hole_x=0.35,
            defect_hole_y=0.65,
            h_coarse=0.030,
            h_center_hole=0.005,
            h_defect_hole=0.0015
        )
        self.mat_cfg = MaterialConfig()

    def test_geometry_bounds_validation(self):
        """Test defect boundary checking and collision avoidance."""
        # Valid position
        valid_cfg = PlateGeometryConfig(defect_hole_x=0.3, defect_hole_y=0.7)
        self.assertTrue(validate_defect_position(valid_cfg))

        # Single-hole mode (defect disabled)
        single_cfg = PlateGeometryConfig(has_defect=False)
        self.assertTrue(validate_defect_position(single_cfg))

        # Outside left
        invalid_cfg1 = PlateGeometryConfig(defect_hole_x=0.002, defect_hole_y=0.5)
        self.assertFalse(validate_defect_position(invalid_cfg1))

        # Overlapping central hole
        invalid_cfg2 = PlateGeometryConfig(defect_hole_x=0.505, defect_hole_y=0.505)
        self.assertFalse(validate_defect_position(invalid_cfg2))

    def test_mesh_generation_two_hole(self):
        """Test Gmsh generates valid triangles and physical groups for 2-hole plate."""
        nodes, elements, tags = generate_mesh(self.geom_cfg, verbose=False)
        self.assertGreater(len(nodes), 500)
        self.assertGreater(len(elements), 1000)
        self.assertIn("PlateSurface", tags)
        self.assertIn("Top", tags)
        self.assertIn("Bottom", tags)
        self.assertIn("CenterHole", tags)
        self.assertIn("DefectHole", tags)

    def test_mesh_generation_single_hole(self):
        """Test Gmsh generates valid mesh for single central hole benchmark."""
        single_cfg = PlateGeometryConfig(has_defect=False, h_center_hole=0.005, h_coarse=0.03)
        nodes, elements, tags = generate_mesh(single_cfg, verbose=False)
        self.assertGreater(len(nodes), 400)
        self.assertIn("CenterHole", tags)
        self.assertEqual(tags.get("DefectHole", -1), -1)

    def test_fem_solution_physics(self):
        """Verify FEM physics: equilibrium, boundary conditions, and stress ranges."""
        res = solve_plane_stress_fem(self.geom_cfg, self.mat_cfg, verbose=False)

        # 1. Non-zero, finite results
        self.assertTrue(np.isfinite(res.max_von_mises_psi))
        self.assertGreater(res.max_von_mises_psi, 0.0)
        self.assertGreater(res.strain_energy, 0.0)
        self.assertGreater(res.max_displacement, 0.0)

        # 2. Boundary condition satisfaction
        # Bottom nodes (y=0) must have u_y = 0 within 1e-12
        eps = 1e-5
        bottom_node_indices = np.where(abs(res.nodes[:, 1] - 0.0) < eps)[0]
        uy_bottom = res.displacements[bottom_node_indices, 1]
        np.testing.assert_allclose(uy_bottom, 0.0, atol=1e-12, err_msg="Bottom boundary roller condition violated!")

        # Center-bottom node must have u_x = 0
        center_bottom_idx = bottom_node_indices[np.argmin(abs(res.nodes[bottom_node_indices, 0] - 0.5))]
        self.assertAlmostEqual(res.displacements[center_bottom_idx, 0], 0.0, places=12)

        # 3. Stress concentration factor range (Kirsch ~ 3.0, finite width ~ 2.5 - 3.2)
        self.assertGreaterEqual(res.max_von_mises_psi, 200.0)
        self.assertLessEqual(res.max_von_mises_psi, 340.0)

        # 4. Total applied vertical load equilibrium check
        expected_force_n = self.mat_cfg.applied_traction_pa * self.geom_cfg.plate_width * self.mat_cfg.thickness
        self.assertAlmostEqual(res.applied_force_total_n, expected_force_n, delta=expected_force_n * 0.001)

    def test_global_equilibrium_reactions(self):
        """Verify sum of vertical forces and support reactions satisfies equilibrium."""
        res = solve_plane_stress_fem(self.geom_cfg, self.mat_cfg, verbose=False)
        # Sum of vertical forces: F_top + R_bottom = 0
        self.assertLess(res.equilibrium_error_n, 1e-6, "Vertical force equilibrium violated!")
        # Horizontal reaction at pin: Rx ~ 0
        self.assertLess(abs(res.horizontal_reaction_n), 1e-6, "Horizontal pin equilibrium violated!")

    def test_unrestrained_poisson_lateral_contraction(self):
        """Verify that bottom edge contracts inward freely without lateral clamping."""
        res = solve_plane_stress_fem(self.geom_cfg, self.mat_cfg, verbose=False)
        bot_mask = np.isclose(res.nodes[:, 1], 0.0, atol=1e-5)
        bot_nodes = res.nodes[bot_mask]
        bot_ux = res.displacements[bot_mask, 0]

        # Left corner (x=0, y=0) should move right (ux > 0)
        left_corner_idx = np.argmin(bot_nodes[:, 0])
        self.assertGreater(bot_ux[left_corner_idx], 0.0)

        # Right corner (x=1, y=0) should move left (ux < 0)
        right_corner_idx = np.argmax(bot_nodes[:, 0])
        self.assertLess(bot_ux[right_corner_idx], 0.0)

    def test_single_central_hole_kirsch_benchmark(self):
        """Verify single central hole converges to Peterson/Kirsch concentration factor ~ 3.0."""
        single_cfg = PlateGeometryConfig(has_defect=False, h_center_hole=0.002, h_coarse=0.02)
        res = solve_plane_stress_fem(single_cfg, self.mat_cfg, verbose=False)
        kt_nodal = res.max_von_mises_psi / 100.0
        kt_elem = res.max_element_von_mises_psi / 100.0

        # Peterson theoretical Kt is 3.015. Nodal recovery at h=0.002 achieves > 2.90, Element vM achieves ~ 3.05
        self.assertGreater(kt_nodal, 2.85)
        self.assertLess(kt_nodal, 3.15)
        self.assertGreater(kt_elem, 2.95)
        self.assertLess(kt_elem, 3.15)

        # Peak stress location must be on the transverse diameter y ~ 0.5, x ~ 0.5417 or 0.4583
        self.assertAlmostEqual(res.max_stress_location[1], 0.5, delta=0.01)
        self.assertTrue(
            abs(res.max_stress_location[0] - 0.5417) < 0.01 or abs(res.max_stress_location[0] - 0.4583) < 0.01
        )


if __name__ == "__main__":
    unittest.main()
