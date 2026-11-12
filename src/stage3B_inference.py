"""
Stage 3B: Reusable Production Forward Stress Ensemble Predictor
Provides unified inference interface for:
- Framework 1: DNN-Only Stacking
- Framework 2: PCA-Based Stacking
- Framework 3: Multi-Level Meta-Stacking
- Framework 4: Modern Heterogeneous Stacking (Default Best Model)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

FEATURE_COLS = [
    'defect_x_over_L',
    'defect_y_over_H',
    'defect_d_over_Dc',
    'aspect_ratio',
    'ligament_clearance_over_Dc',
    'min_boundary_clearance_over_H'
]

# Neural network architectures
class ShallowANN(nn.Module):
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D1(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D2(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class ResMLP(nn.Module):
    def __init__(self, in_features: int, width: int = 256, num_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, width),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(width, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_proj(h)

class ForwardStressEnsemblePredictor:
    """Production Forward Stress Predictor for Perforated Plates."""
    
    def __init__(self, models_dir: Union[str, Path] = "models/stage3B_forward_stacking"):
        self.models_dir = Path(models_dir)
        self.scaler = joblib.load(self.models_dir / "stage3B_feature_scaler.pkl")
        
        # Load base estimators
        self.ann_512 = ShallowANN(len(FEATURE_COLS), 512)
        self.ann_512.load_state_dict(torch.load(self.models_dir / "baseline_ANN_512.pt", map_location='cpu'))
        self.ann_512.eval()
        
        self.deep_d1 = DeepMLP_D1(len(FEATURE_COLS))
        self.deep_d1.load_state_dict(torch.load(self.models_dir / "baseline_DeepMLP_D1.pt", map_location='cpu'))
        self.deep_d1.eval()
        
        self.deep_d2 = DeepMLP_D2(len(FEATURE_COLS))
        self.deep_d2.load_state_dict(torch.load(self.models_dir / "baseline_DeepMLP_D2.pt", map_location='cpu'))
        self.deep_d2.eval()
        
        self.resmlp = ResMLP(len(FEATURE_COLS), width=256, num_blocks=3)
        self.resmlp.load_state_dict(torch.load(self.models_dir / "baseline_ResMLP.pt", map_location='cpu'))
        self.resmlp.eval()
        
        self.rf = joblib.load(self.models_dir / "baseline_RandomForest.pkl")
        self.et = joblib.load(self.models_dir / "baseline_ExtraTrees.pkl")
        self.hgb = joblib.load(self.models_dir / "baseline_HistGradientBoosting.pkl")
        
        # Load meta learners
        self.meta_dnn = joblib.load(self.models_dir / "meta_framework1_dnn_ridge.pkl")
        self.meta_pca = joblib.load(self.models_dir / "meta_framework2_pca_ridge.pkl")
        self.pca = joblib.load(self.models_dir / "meta_framework2_pca.pkl")
        self.pca_scaler = joblib.load(self.models_dir / "meta_framework2_pca_scaler.pkl")
        self.meta_het = joblib.load(self.models_dir / "meta_framework4_heterogeneous.pkl")
        
    def _extract_features(
        self,
        plate_length_mm: float,
        plate_height_mm: float,
        central_hole_diameter_mm: float,
        defect_diameter_mm: float,
        defect_center_x_mm: float,
        defect_center_y_mm: float
    ) -> np.ndarray:
        L = float(plate_length_mm)
        H = float(plate_height_mm)
        Dc = float(central_hole_diameter_mm)
        d = float(defect_diameter_mm)
        x = float(defect_center_x_mm)
        y = float(defect_center_y_mm)
        
        Xc = L / 2.0
        Yc = H / 2.0
        
        # Validations
        dist_c = np.hypot(x - Xc, y - Yc)
        c_lig = dist_c - (Dc + d) / 2.0
        if c_lig <= 0:
            raise ValueError(f"Geometric collision: Defect intersects central hole (clearance: {c_lig:.2f} mm)")
            
        c_left = x - d / 2.0
        c_right = L - x - d / 2.0
        c_bot = y - d / 2.0
        c_top = H - y - d / 2.0
        min_bnd = min(c_left, c_right, c_bot, c_top)
        if min_bnd <= 0:
            raise ValueError(f"Geometric collision: Defect penetrates outer plate boundary (clearance: {min_bnd:.2f} mm)")
            
        feat = np.array([
            x / L,
            y / H,
            d / Dc,
            L / H,
            c_lig / Dc,
            min_bnd / H
        ], dtype=np.float32).reshape(1, -1)
        
        return self.scaler.transform(feat)

    def predict(
        self,
        plate_length_mm: float,
        plate_height_mm: float,
        central_hole_diameter_mm: float,
        defect_diameter_mm: float,
        defect_center_x_mm: float,
        defect_center_y_mm: float,
        applied_traction_psi: float = 100.0,
        framework: str = "heterogeneous"
    ) -> Dict[str, Any]:
        """Predict maximum stress concentration factor and stress in psi."""
        X_norm = self._extract_features(
            plate_length_mm, plate_height_mm, central_hole_diameter_mm,
            defect_diameter_mm, defect_center_x_mm, defect_center_y_mm
        )
        
        t_X = torch.tensor(X_norm, dtype=torch.float32)
        with torch.no_grad():
            p_ann = float(self.ann_512(t_X).item())
            p_d1 = float(self.deep_d1(t_X).item())
            p_d2 = float(self.deep_d2(t_X).item())
            p_res = float(self.resmlp(t_X).item())
            
        p_rf = float(self.rf.predict(X_norm)[0])
        p_et = float(self.et.predict(X_norm)[0])
        p_hgb = float(self.hgb.predict(X_norm)[0])
        
        # Base vector [ANN_512, D1, D2, ResMLP, RF, ET, HGB]
        base_preds = np.array([p_ann, p_d1, p_d2, p_res, p_rf, p_et, p_hgb], dtype=np.float32).reshape(1, -1)
        
        fw = framework.lower()
        if "dnn" in fw:
            pred_kt = float(self.meta_dnn.predict(base_preds[:, [0, 1, 2, 3]])[0])
            active_members = [p_ann, p_d1, p_d2, p_res]
        elif "pca" in fw:
            z_norm = self.pca_scaler.transform(base_preds)
            k_comp = getattr(self.meta_pca, "n_features_in_", 5)
            z_pca = self.pca.transform(z_norm)[:, :k_comp]
            pred_kt = float(self.meta_pca.predict(z_pca)[0])
            active_members = [p_ann, p_d1, p_d2, p_res, p_rf, p_et, p_hgb]
        else: # Default: heterogeneous
            pred_kt = float(self.meta_het.predict(base_preds[:, [2, 3, 4, 5, 6]])[0])
            active_members = [p_d2, p_res, p_rf, p_et, p_hgb]
            
        pred_stress_psi = pred_kt * applied_traction_psi
        uncertainty_psi = float(np.std(active_members) * applied_traction_psi)
        
        return {
            'framework': framework,
            'predicted_Kt_element': pred_kt,
            'predicted_max_stress_psi': pred_stress_psi,
            'ensemble_std_uncertainty_psi': uncertainty_psi,
            'applied_traction_psi': applied_traction_psi,
            'base_predictions_Kt': {
                'ANN_512': p_ann,
                'DeepMLP_D1': p_d1,
                'DeepMLP_D2': p_d2,
                'ResMLP': p_res,
                'RandomForest': p_rf,
                'ExtraTrees': p_et,
                'HistGradientBoosting': p_hgb
            }
        }
