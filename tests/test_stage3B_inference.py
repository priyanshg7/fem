"""
Unit and Integration Tests for Stage 3B Forward Stress Ensemble Predictor
"""

import unittest
import numpy as np
from src.stage3B_inference import ForwardStressEnsemblePredictor

class TestStage3BInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = ForwardStressEnsemblePredictor()
        
    def test_single_prediction_heterogeneous(self):
        res = self.predictor.predict(
            plate_length_mm=1000.0,
            plate_height_mm=800.0,
            central_hole_diameter_mm=66.6667,
            defect_diameter_mm=12.0,
            defect_center_x_mm=300.0,
            defect_center_y_mm=250.0,
            applied_traction_psi=100.0,
            framework="heterogeneous"
        )
        self.assertIn('predicted_Kt_element', res)
        self.assertIn('predicted_max_stress_psi', res)
        self.assertIn('ensemble_std_uncertainty_psi', res)
        self.assertGreater(res['predicted_Kt_element'], 2.5)
        self.assertLess(res['predicted_Kt_element'], 6.0)
        self.assertAlmostEqual(res['predicted_max_stress_psi'], res['predicted_Kt_element'] * 100.0, places=4)
        print(f"\n[Test Heterogeneous] Predicted Kt: {res['predicted_Kt_element']:.4f}, Stress: {res['predicted_max_stress_psi']:.2f} psi, Unc: {res['ensemble_std_uncertainty_psi']:.2f} psi")

    def test_dnn_and_pca_frameworks(self):
        res_dnn = self.predictor.predict(
            plate_length_mm=1000.0,
            plate_height_mm=1000.0,
            central_hole_diameter_mm=83.3333,
            defect_diameter_mm=15.0,
            defect_center_x_mm=350.0,
            defect_center_y_mm=650.0,
            applied_traction_psi=100.0,
            framework="dnn"
        )
        self.assertGreater(res_dnn['predicted_Kt_element'], 2.5)
        
        res_pca = self.predictor.predict(
            plate_length_mm=1000.0,
            plate_height_mm=1000.0,
            central_hole_diameter_mm=83.3333,
            defect_diameter_mm=15.0,
            defect_center_x_mm=350.0,
            defect_center_y_mm=650.0,
            applied_traction_psi=100.0,
            framework="pca"
        )
        self.assertGreater(res_pca['predicted_Kt_element'], 2.5)
        print(f"[Test DNN Stack] Predicted Kt: {res_dnn['predicted_Kt_element']:.4f}")
        print(f"[Test PCA Stack] Predicted Kt: {res_pca['predicted_Kt_element']:.4f}")

    def test_geometric_collision_validation(self):
        # Case 1: Defect inside central hole
        with self.assertRaises(ValueError):
            self.predictor.predict(
                plate_length_mm=1000.0,
                plate_height_mm=1000.0,
                central_hole_diameter_mm=83.3333,
                defect_diameter_mm=20.0,
                defect_center_x_mm=500.0, # At plate center!
                defect_center_y_mm=500.0
            )
            
        # Case 2: Defect outside outer boundary
        with self.assertRaises(ValueError):
            self.predictor.predict(
                plate_length_mm=1000.0,
                plate_height_mm=1000.0,
                central_hole_diameter_mm=83.3333,
                defect_diameter_mm=20.0,
                defect_center_x_mm=995.0, # Crossing edge
                defect_center_y_mm=500.0
            )

if __name__ == '__main__':
    unittest.main()
