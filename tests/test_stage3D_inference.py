"""
Unit tests for Stage 3D Production Defect Localizer across all model families:
- Regional (Model D1)
- Sensors (Model D2)
- PCA (Model D3)
- CNN (Model D4)
- U-Net (Model D5) with uncertainty ellipses and abstention
"""
import unittest
import numpy as np
from src.stage3D_inference import SpatialDefectLocalizer

class TestSpatialDefectLocalizer(unittest.TestCase):
    def setUp(self):
        self.localizer_reg = SpatialDefectLocalizer(model_type='regional')
        self.localizer_sens = SpatialDefectLocalizer(model_type='sensors')
        self.localizer_pca = SpatialDefectLocalizer(model_type='pca')
        self.localizer_cnn = SpatialDefectLocalizer(model_type='cnn')
        self.localizer_unet = SpatialDefectLocalizer(model_type='unet')

    def test_regional_inference_dict(self):
        dummy_feat = {
            'D1_Kt_max': 2.85,
            'D1_Kt_Q1': 2.85,
            'D1_Kt_Q2': 1.15,
            'D1_Kt_Q3': 1.08,
            'D1_Kt_Q4': 1.12,
            'D1_mean_Q1': 1.25,
            'D1_mean_Q2': 0.98,
            'D1_mean_Q3': 0.95,
            'D1_mean_Q4': 0.97,
            'D1_p90_Q1': 1.65,
            'D1_p90_Q2': 1.05,
            'D1_p90_Q3': 1.02,
            'D1_p90_Q4': 1.04,
            'D1_Kt_ring': 2.10,
            'D1_mean_ring': 1.45,
            'D1_p90_ring': 1.85
        }
        res = self.localizer_reg.predict(dummy_feat, plate_length_mm=1000.0, plate_height_mm=800.0, defect_diameter_mm=25.0)
        self.assertIn('predicted_x_mm', res)
        self.assertIn('predicted_y_mm', res)
        self.assertGreaterEqual(res['predicted_x_mm'], 0.0)
        self.assertLessEqual(res['predicted_x_mm'], 1000.0)
        self.assertGreaterEqual(res['predicted_y_mm'], 0.0)
        self.assertLessEqual(res['predicted_y_mm'], 800.0)
        self.assertGreater(res['predicted_x_mm'], 400.0)
        self.assertGreater(res['predicted_y_mm'], 300.0)

    def test_sensors_inference_array(self):
        dummy_sensors = np.ones(64, dtype=np.float32) * 1.0
        dummy_sensors[:16] = 2.4
        res = self.localizer_sens.predict(dummy_sensors, plate_length_mm=1000.0, plate_height_mm=600.0)
        self.assertIn('predicted_x_mm', res)
        self.assertIn('predicted_y_mm', res)
        self.assertGreaterEqual(res['predicted_x_mm'], 0.0)
        self.assertLessEqual(res['predicted_x_mm'], 1000.0)

    def test_pca_inference_array(self):
        dummy_pca = np.zeros(120, dtype=np.float32)
        dummy_pca[0] = 1.5
        res = self.localizer_pca.predict(dummy_pca, plate_length_mm=1000.0, plate_height_mm=900.0)
        self.assertIn('predicted_x_mm', res)
        self.assertIn('predicted_y_mm', res)
        self.assertGreaterEqual(res['predicted_x_mm'], 0.0)
        self.assertLessEqual(res['predicted_x_mm'], 1000.0)

    def test_cnn_inference_tensor(self):
        dummy_field = np.ones((64, 64), dtype=np.float32)
        res = self.localizer_cnn.predict(dummy_field, plate_length_mm=1000.0, plate_height_mm=700.0)
        self.assertIn('predicted_x_mm', res)
        self.assertIn('predicted_y_mm', res)
        self.assertGreaterEqual(res['predicted_x_mm'], 0.0)
        self.assertLessEqual(res['predicted_x_mm'], 1000.0)

    def test_unet_inference_with_uncertainty_and_abstention(self):
        dummy_field = np.ones((64, 64), dtype=np.float32)
        res = self.localizer_unet.predict(
            dummy_field,
            plate_length_mm=1000.0,
            plate_height_mm=800.0,
            confidence_abstention_threshold_mm=25.0
        )
        self.assertIn('predicted_x_mm', res)
        self.assertIn('predicted_y_mm', res)
        self.assertIn('uncertainty_radius_mm', res)
        self.assertIn('uncertainty_ellipse', res)
        self.assertIn('abstain', res)
        self.assertIn('confidence_flag', res)
        
        # Verify uncertainty ellipse dictionary fields
        ellipse = res['uncertainty_ellipse']
        self.assertIn('semi_major_axis_mm', ellipse)
        self.assertIn('semi_minor_axis_mm', ellipse)
        self.assertIn('rotation_angle_deg', ellipse)
        self.assertGreater(ellipse['semi_major_axis_mm'], 0.0)
        self.assertGreater(ellipse['semi_minor_axis_mm'], 0.0)
        self.assertGreaterEqual(ellipse['semi_major_axis_mm'], ellipse['semi_minor_axis_mm'] - 1e-4)

    def test_input_validation(self):
        with self.assertRaises(FileNotFoundError):
            SpatialDefectLocalizer(model_type='regional', model_path='nonexistent_model.joblib')

if __name__ == '__main__':
    unittest.main()
