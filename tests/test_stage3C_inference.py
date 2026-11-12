"""
Unit Tests for Stage 3C Production Inverse Defect Localizer
============================================================
Runs natively using Python's built-in `unittest` framework.
Tests:
- Initialization of InverseDefectLocalizer
- Input range validation and collision checks
- Experiment C1: Reference square plate inversion
- Experiment C2: Extended multi-geometry scalar inversion
- Experiment C3: Conditioned multi-model inversion (RF, ET, HGB, KNN, Ridge, ANNs)
- Phase F: Inverse Stacking (Ridge and PCA meta-models)
- Phase G: Forward-assisted candidate contour search
- Phase L: Stress noise sensitivity and Monte Carlo uncertainty estimation
- Identifiability diagnostics and mathematical ill-posedness warnings
"""

import unittest
import numpy as np
from pathlib import Path
from src.stage3C_inference import InverseDefectLocalizer


class TestStage3CInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.localizer = InverseDefectLocalizer()

    def test_01_initialization(self):
        """Verify localizer initializes and loads preprocessing scalers."""
        self.assertIsNotNone(self.localizer.scaler_c1)
        self.assertIsNotNone(self.localizer.scaler_c2)
        self.assertIsNotNone(self.localizer.scaler_c3)
        self.assertGreater(len(self.localizer.conv_models), 0)

    def test_02_input_validation(self):
        """Verify strictly positive stress and dimensions are enforced."""
        with self.assertRaises(ValueError):
            self.localizer.predict_reference_square(max_von_mises_psi=-10.0)
            
        with self.assertRaises(ValueError):
            self.localizer.predict_scalar_multigeom(max_von_mises_psi=0.0)
            
        with self.assertRaises(ValueError):
            self.localizer.predict_conditioned(
                max_von_mises_psi=350.0,
                plate_length_mm=-1000.0,
                plate_height_mm=1000.0,
                central_hole_diameter_mm=83.33,
                defect_diameter_mm=12.5
            )
            
        with self.assertRaises(ValueError):
            self.localizer.predict_conditioned(
                max_von_mises_psi=350.0,
                plate_length_mm=1000.0,
                plate_height_mm=1000.0,
                central_hole_diameter_mm=0.0,
                defect_diameter_mm=12.5
            )

    def test_03_reference_square_prediction(self):
        """Verify Experiment C1 reference inversion."""
        res = self.localizer.predict_reference_square(max_von_mises_psi=350.0, model_type="ANN-512")
        self.assertEqual(res['formulation'], 'Experiment_C1_Reference_Square')
        self.assertTrue(0.0 <= res['predicted_norm_coords']['x_over_L'] <= 1.0)
        self.assertTrue(0.0 <= res['predicted_norm_coords']['y_over_H'] <= 1.0)
        self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['x_mm'] <= 1000.0)
        self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['y_mm'] <= 1000.0)
        self.assertEqual(len(res['symmetry_candidates_mm']), 4)
        self.assertIn('ILL-POSED INVERSION', res['identifiability_warning'])

    def test_04_scalar_multigeom_prediction(self):
        """Verify Experiment C2 scalar-only inversion for rectangular plate."""
        res = self.localizer.predict_scalar_multigeom(
            max_von_mises_psi=320.0,
            plate_length_mm=1000.0,
            plate_height_mm=700.0,
            model_type="ANN-512"
        )
        self.assertEqual(res['formulation'], 'Experiment_C2_Extended_Scalar_Only')
        self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['x_mm'] <= 1000.0)
        self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['y_mm'] <= 700.0)
        self.assertIn('ILL-POSED INVERSION', res['identifiability_warning'])

    def test_05_conditioned_prediction(self):
        """Verify Experiment C3 conditioned inversion across multiple models."""
        for model_name in ["RandomForest", "ExtraTrees", "HistGradientBoosting", "Ridge", "LinearRegression"]:
            res = self.localizer.predict_conditioned(
                max_von_mises_psi=340.0,
                plate_length_mm=1000.0,
                plate_height_mm=800.0,
                central_hole_diameter_mm=83.33,
                defect_diameter_mm=25.0,
                model_type=model_name
            )
            self.assertEqual(res['formulation'], 'Experiment_C3_Geometry_Conditioned')
            self.assertTrue(0.0 <= res['predicted_norm_coords']['x_over_L'] <= 1.0)
            self.assertTrue(0.0 <= res['predicted_norm_coords']['y_over_H'] <= 1.0)
            self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['x_mm'] <= 1000.0)
            self.assertTrue(0.0 <= res['predicted_phys_coords_mm']['y_mm'] <= 800.0)

    def test_06_inverse_stacking(self):
        """Verify Phase F inverse stacking with Ridge and PCA meta-models."""
        res_ridge = self.localizer.predict_stacked(
            max_von_mises_psi=330.0,
            plate_length_mm=1000.0,
            plate_height_mm=900.0,
            central_hole_diameter_mm=83.33,
            defect_diameter_mm=15.0,
            meta_type="ridge"
        )
        self.assertEqual(res_ridge['formulation'], 'Phase_F_Inverse_Stacking')
        self.assertIn('base_predictions', res_ridge)
        self.assertTrue(0.0 <= res_ridge['predicted_phys_coords_mm']['x_mm'] <= 1000.0)
        
        res_pca = self.localizer.predict_stacked(
            max_von_mises_psi=330.0,
            plate_length_mm=1000.0,
            plate_height_mm=900.0,
            central_hole_diameter_mm=83.33,
            defect_diameter_mm=15.0,
            meta_type="pca"
        )
        self.assertEqual(res_pca['formulation'], 'Phase_F_Inverse_Stacking')
        self.assertTrue(0.0 <= res_pca['predicted_phys_coords_mm']['x_mm'] <= 1000.0)

    def test_07_forward_assisted_contour_search(self):
        """Verify Phase G forward-assisted iso-stress level-set contour mapping."""
        res = self.localizer.generate_candidate_contours(
            max_von_mises_psi=315.0,
            plate_length_mm=1000.0,
            plate_height_mm=1000.0,
            central_hole_diameter_mm=83.33,
            defect_diameter_mm=12.5,
            tolerance_psi=3.0,
            grid_density=12  # Small grid for fast unit test
        )
        self.assertEqual(res['formulation'], 'Phase_G_Forward_Assisted_Level_Set_Search')
        self.assertEqual(res['tolerance_psi'], 3.0)
        self.assertIn('candidate_count', res)
        self.assertIn('candidate_centroid_mm', res)
        self.assertIn('bounding_box_mm', res)
        self.assertIn('SUCCESSFUL LEVEL-SET RECOVERY', res['scientific_takeaway'])

    def test_08_stress_noise_uncertainty(self):
        """Verify Phase L Monte Carlo stress perturbation analysis."""
        res = self.localizer.predict_with_uncertainty(
            max_von_mises_psi=325.0,
            plate_length_mm=1000.0,
            plate_height_mm=1000.0,
            central_hole_diameter_mm=83.33,
            defect_diameter_mm=12.5,
            noise_std_pct=2.0,
            n_samples=15
        )
        self.assertEqual(res['formulation'], 'Phase_L_Stress_Noise_Sensitivity')
        self.assertEqual(res['n_monte_carlo_samples'], 15)
        self.assertGreaterEqual(res['mean_coordinate_shift_mm'], 0.0)
        self.assertGreaterEqual(res['max_coordinate_shift_mm'], res['mean_coordinate_shift_mm'])


if __name__ == '__main__':
    unittest.main()
