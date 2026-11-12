"""
Stage 3D Production Inference Interface:
SpatialDefectLocalizer provides a unified, production-grade interface
for spatial defect localization and sizing across all Stage 3D model families.
"""
import os
import joblib
import numpy as np
import torch
from typing import Dict, Any, Optional, Union

MODEL_DIR = "models/stage3D_spatial_inverse"

class SpatialDefectLocalizer:
    """
    Unified Production Defect Localizer supporting:
    - 'regional': Model D1 (Regional Peak Stress Features)
    - 'sensors': Model D2 (Discrete Virtual Strain Gauge Arrays)
    - 'pca': Model D3 (PCA Modal Reduction)
    - 'cnn': Model D4 (Convolutional Neural Network)
    - 'unet': Model D5 (U-Net Heatmap & Spatial Uncertainty)
    """
    def __init__(self, model_type: str = 'regional', model_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self.model = None
        self.meta = {}
        
        if model_path is None:
            if self.model_type == 'regional':
                model_path = os.path.join(MODEL_DIR, "model_D1_E1_best.joblib")
            elif self.model_type == 'sensors':
                model_path = os.path.join(MODEL_DIR, "model_D2_sensors_64.joblib")
            elif self.model_type == 'pca':
                model_path = os.path.join(MODEL_DIR, "model_D3_pca_k120.joblib")
            elif self.model_type == 'cnn':
                model_path = os.path.join(MODEL_DIR, "model_D4_cnn.pt")
            elif self.model_type == 'unet':
                model_path = os.path.join(MODEL_DIR, "model_D5_unet.pt")
            else:
                raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'regional', 'sensors', 'pca', 'cnn', 'unet'.")
                
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
        if self.model_type in ['regional', 'sensors', 'pca']:
            self.model = joblib.load(self.model_path)
        elif self.model_type == 'cnn':
            from src.stage3D_cnn_unet import DefectInverseCNN
            self.model = DefectInverseCNN(in_channels=1, meta_dim=2, out_dim=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
        elif self.model_type == 'unet':
            from src.stage3D_cnn_unet import DefectHeatmapUNet
            self.model = DefectHeatmapUNet(in_channels=1)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()

    def predict(
        self,
        features: Union[np.ndarray, Dict[str, float]],
        plate_length_mm: float,
        plate_height_mm: float,
        defect_diameter_mm: Optional[float] = None,
        confidence_abstention_threshold_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict defect location (and uncertainty if available).
        
        Parameters:
        -----------
        features : np.ndarray or dict containing required stress features or rasterized field
        plate_length_mm : specimen length in mm (L)
        plate_height_mm : specimen height in mm (H)
        defect_diameter_mm : optional known diameter in mm (defaults to nominal canonical d_ratio)
        confidence_abstention_threshold_mm : optional uncertainty radius limit to flag high-confidence
        """
        L = float(plate_length_mm)
        H = float(plate_height_mm)
        ar = L / H
        Dc = H / 12.0
        d_val = float(defect_diameter_mm) if defect_diameter_mm is not None else (0.15 * Dc)
        d_ratio = d_val / Dc
        
        if self.model_type == 'regional':
            if isinstance(features, dict):
                d1_cols = [
                    'D1_Kt_max', 'D1_Kt_Q1', 'D1_Kt_Q2', 'D1_Kt_Q3', 'D1_Kt_Q4',
                    'D1_mean_Q1', 'D1_mean_Q2', 'D1_mean_Q3', 'D1_mean_Q4',
                    'D1_p90_Q1', 'D1_p90_Q2', 'D1_p90_Q3', 'D1_p90_Q4',
                    'D1_Kt_ring', 'D1_mean_ring', 'D1_p90_ring'
                ]
                vals = [features[c] for c in d1_cols] + [ar, d_ratio]
                X = np.array([vals], dtype=np.float32)
            else:
                X = np.asarray(features, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                    
            pred_norm = self.model.predict(X)[0]
            pred_x_mm = float(pred_norm[0] * L)
            pred_y_mm = float(pred_norm[1] * H)
            
            return {
                'model_family': 'Model D1 (Regional Peak Stress)',
                'predicted_x_mm': float(np.clip(pred_x_mm, 0.0, L)),
                'predicted_y_mm': float(np.clip(pred_y_mm, 0.0, H)),
                'normalized_xi': float(np.clip(pred_norm[0], 0.0, 1.0)),
                'normalized_eta': float(np.clip(pred_norm[1], 0.0, 1.0)),
                'plate_length_mm': L,
                'plate_height_mm': H
            }
            
        elif self.model_type == 'sensors':
            X = np.asarray(features, dtype=np.float32)
            if X.ndim == 1:
                if len(X) == 64:
                    X = np.hstack([X, [ar, d_ratio]])
                X = X.reshape(1, -1)
            pred_norm = self.model.predict(X)[0]
            pred_x_mm = float(pred_norm[0] * L)
            pred_y_mm = float(pred_norm[1] * H)
            return {
                'model_family': 'Model D2 (Virtual Sensor Array)',
                'predicted_x_mm': float(np.clip(pred_x_mm, 0.0, L)),
                'predicted_y_mm': float(np.clip(pred_y_mm, 0.0, H)),
                'normalized_xi': float(np.clip(pred_norm[0], 0.0, 1.0)),
                'normalized_eta': float(np.clip(pred_norm[1], 0.0, 1.0)),
                'plate_length_mm': L,
                'plate_height_mm': H
            }
            
        elif self.model_type == 'pca':
            X = np.asarray(features, dtype=np.float32)
            if X.ndim == 1:
                if len(X) == 120:
                    X = np.hstack([X, [ar, d_ratio]])
                X = X.reshape(1, -1)
            pred_norm = self.model.predict(X)[0]
            pred_x_mm = float(pred_norm[0] * L)
            pred_y_mm = float(pred_norm[1] * H)
            return {
                'model_family': 'Model D3 (PCA Modal Coefficients)',
                'predicted_x_mm': float(np.clip(pred_x_mm, 0.0, L)),
                'predicted_y_mm': float(np.clip(pred_y_mm, 0.0, H)),
                'normalized_xi': float(np.clip(pred_norm[0], 0.0, 1.0)),
                'normalized_eta': float(np.clip(pred_norm[1], 0.0, 1.0)),
                'plate_length_mm': L,
                'plate_height_mm': H
            }
            
        elif self.model_type == 'cnn':
            arr = np.asarray(features, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            elif arr.ndim == 3:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
            x_field = torch.tensor(arr, dtype=torch.float32)
            x_meta = torch.tensor([[ar, d_ratio]], dtype=torch.float32)
            with torch.no_grad():
                pred_t = self.model(x_field, x_meta)
                pred_norm = pred_t[0].cpu().numpy()
            pred_x_mm = float(pred_norm[0] * L)
            pred_y_mm = float(pred_norm[1] * H)
            return {
                'model_family': 'Model D4 (Convolutional Neural Network)',
                'predicted_x_mm': float(np.clip(pred_x_mm, 0.0, L)),
                'predicted_y_mm': float(np.clip(pred_y_mm, 0.0, H)),
                'normalized_xi': float(np.clip(pred_norm[0], 0.0, 1.0)),
                'normalized_eta': float(np.clip(pred_norm[1], 0.0, 1.0)),
                'plate_length_mm': L,
                'plate_height_mm': H
            }
            
        elif self.model_type == 'unet':
            from src.stage3D_cnn_unet import soft_argmax_2d
            arr = np.asarray(features, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            elif arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.unsqueeze(0) if hasattr(arr, 'unsqueeze') else arr.reshape(1, 1, arr.shape[1], arr.shape[2])
            x_field = torch.tensor(arr, dtype=torch.float32)
            
            with torch.no_grad():
                logits = self.model(x_field)
                coords_norm, unc_norm, cov_norm = soft_argmax_2d(logits, beta=40.0)
                
            norm_x = float(coords_norm[0, 0].cpu().item())
            norm_y = float(coords_norm[0, 1].cpu().item())
            pred_x_mm = float(np.clip(norm_x * L, 0.0, L))
            pred_y_mm = float(np.clip(norm_y * H, 0.0, H))
            
            # Physical covariance scaling
            c_norm = cov_norm[0].cpu().numpy()
            cov_phys = np.array([
                [c_norm[0, 0] * (L**2), c_norm[0, 1] * (L * H)],
                [c_norm[1, 0] * (L * H), c_norm[1, 1] * (H**2)]
            ], dtype=np.float64)
            
            unc_radius_phys = float(np.sqrt(max(cov_phys[0, 0] + cov_phys[1, 1], 1e-6)))
            
            # Eigen-decomposition for uncertainty ellipse
            eigvals, eigvecs = np.linalg.eigh(cov_phys)
            semi_major = float(np.sqrt(max(eigvals[1], 1e-6)))
            semi_minor = float(np.sqrt(max(eigvals[0], 1e-6)))
            angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
            
            res = {
                'model_family': 'Model D5 (Spatial Heatmap U-Net)',
                'predicted_x_mm': pred_x_mm,
                'predicted_y_mm': pred_y_mm,
                'normalized_xi': norm_x,
                'normalized_eta': norm_y,
                'uncertainty_radius_mm': unc_radius_phys,
                'uncertainty_covariance_mm2': cov_phys.tolist(),
                'uncertainty_ellipse': {
                    'semi_major_axis_mm': semi_major,
                    'semi_minor_axis_mm': semi_minor,
                    'rotation_angle_deg': angle_deg
                },
                'plate_length_mm': L,
                'plate_height_mm': H
            }
            
            if confidence_abstention_threshold_mm is not None:
                res['abstain'] = bool(unc_radius_phys > confidence_abstention_threshold_mm)
                res['confidence_flag'] = 'REJECT_ABSTAIN' if res['abstain'] else 'ACCEPT_HIGH_CONFIDENCE'
                
            return res
            
        else:
            raise NotImplementedError(f"Inference for model_type '{self.model_type}' not supported.")
