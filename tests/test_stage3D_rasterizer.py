"""
Unit tests for Stage 3D Common Spatial Representation and Anti-Leakage Rasterizer
"""

import unittest
import numpy as np
from pathlib import Path
from src.stage3D_rasterizer import (
    normalize_coordinates,
    rasterize_stress_field,
    generate_known_central_hole_mask,
    generate_leakage_mesh_density_map,
    load_and_rasterize_simulation
)


class TestStage3DRasterizer(unittest.TestCase):
    def test_01_normalization(self):
        coords_mm = np.array([[0.0, 0.0], [500.0, 300.0], [1000.0, 600.0]], dtype=np.float32)
        norm = normalize_coordinates(coords_mm, plate_length_mm=1000.0, plate_height_mm=600.0)
        self.assertEqual(norm.shape, (3, 2))
        np.testing.assert_allclose(norm[0], [0.0, 0.0])
        np.testing.assert_allclose(norm[1], [0.5, 0.5])
        np.testing.assert_allclose(norm[2], [1.0, 1.0])

    def test_02_rasterization_shapes_and_finiteness(self):
        f = Path("data/fields/sim_00001.npz")
        res128 = load_and_rasterize_simulation(f, 1000.0, 1000.0, grid_res=128)
        self.assertEqual(res128['stress_grid'].shape, (128, 128))
        self.assertFalse(np.isnan(res128['stress_grid']).any())
        self.assertFalse(np.isinf(res128['stress_grid']).any())
        self.assertTrue(np.all(res128['stress_grid'] > 0)) # Stresses are strictly positive

        res256 = load_and_rasterize_simulation(f, 1000.0, 1000.0, grid_res=256)
        self.assertEqual(res256['stress_grid'].shape, (256, 256))
        self.assertFalse(np.isnan(res256['stress_grid']).any())

    def test_03_central_hole_mask(self):
        mask128 = generate_known_central_hole_mask(128, 1000.0, 1000.0, 83.333)
        self.assertEqual(mask128.shape, (128, 128))
        # Center pixel (64, 64) must be inside hole (1.0)
        self.assertEqual(mask128[64, 64], 1.0)
        # Corner pixel (0, 0) must be outside hole (0.0)
        self.assertEqual(mask128[0, 0], 0.0)

    def test_04_anti_leakage_void_sealing(self):
        """Verify that the defect void has no zeros, missing values, or NaNs."""
        f = Path("data/fields/sim_00001.npz")
        res = load_and_rasterize_simulation(f, 1000.0, 1000.0, grid_res=128)
        grid = res['stress_grid']
        # The plate under 100 psi tension has far-field stress ~100 psi, cutouts ~300 psi
        # There should be NO zero-stress void artifacts inside the plate domain
        min_stress = grid[10:-10, 10:-10].min()
        self.assertGreater(min_stress, 5.0) # Well above zero


if __name__ == '__main__':
    unittest.main()
