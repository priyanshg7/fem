"""
Inverse defect localization service orchestrator.
Executes the audited Stage 3D Model D5 U-Net Heatmap localizer.
"""
import time
import uuid
import numpy as np
import torch
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.core.errors import SimuStructAPIException, InferenceExecutionError
from backend.app.services.registry import model_registry
from backend.app.schemas.inverse import (
    InversePredictionRequest, InversePredictionResponse,
    UncertaintyEllipse, AbstentionDecision
)
from src.stage3D_cnn_unet import soft_argmax_2d
from src.stage3D_rasterizer import normalize_coordinates, rasterize_stress_field

class InversePredictionService:
    @staticmethod
    def predict(request: InversePredictionRequest) -> InversePredictionResponse:
        t0 = time.perf_counter()
        
        L = float(request.plate_length_mm)
        H = float(request.plate_height_mm)
        traction = float(request.applied_traction_psi or settings.DEFAULT_TRACTION_PSI)
        threshold_mm = float(request.confidence_abstention_threshold_mm or settings.D5_ABSTENTION_THRESHOLD_MM)
        
        # 1. Dimension Validation
        if L <= 0 or H <= 0:
            raise SimuStructAPIException(
                message=f"Plate dimensions must be positive (got L={L}, H={H}).",
                code="INVALID_DIMENSIONS",
                status_code=400
            )
            
        ar = L / H
        warnings = []
        if ar < settings.MIN_ASPECT_RATIO or ar > settings.MAX_ASPECT_RATIO + 0.05:
            warnings.append(
                f"[OUT_OF_DOMAIN_ASPECT_RATIO] Aspect ratio L/H = {ar:.3f} is outside the validated domain [1.000, 1.667]."
            )
            
        # 2. Preprocess Stress Input into (1, 1, 64, 64) normalized tensor
        grid_norm: np.ndarray
        input_mode = "FULL_FIELD_64x64"
        if request.stress_grid_64x64 is not None:
            raw_grid = np.array(request.stress_grid_64x64, dtype=np.float32)
            if raw_grid.ndim != 2 or raw_grid.shape != (settings.GRID_RESOLUTION, settings.GRID_RESOLUTION):
                raise SimuStructAPIException(
                    message=f"stress_grid_64x64 must have dimensions ({settings.GRID_RESOLUTION}, {settings.GRID_RESOLUTION}), got {raw_grid.shape}.",
                    code="INVALID_STRESS_GRID_SHAPE",
                    status_code=400
                )
            if not np.all(np.isfinite(raw_grid)):
                raise SimuStructAPIException(
                    message="stress_grid_64x64 contains non-finite (NaN or Inf) values.",
                    code="NON_FINITE_STRESS_VALUES",
                    status_code=400
                )
            # Normalize by reference far-field traction
            grid_norm = raw_grid / traction
            
        elif request.stress_coordinates is not None and request.stress_values_psi is not None:
            input_mode = "EXPERIMENTAL_SCATTERED_SENSORS"
            warnings.append(
                "[EXPERIMENTAL_INPUT_MODE] Sensor-point interpolation mode is experimental and has not been validated "
                "to the same benchmark standard as full 64x64 rasterized stress fields. Use with caution."
            )
            coords = np.array(request.stress_coordinates, dtype=np.float32)
            values = np.array(request.stress_values_psi, dtype=np.float32)
            
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise SimuStructAPIException(
                    message="stress_coordinates must be an N x 2 array of [x, y] positions.",
                    code="INVALID_COORDINATES_SHAPE",
                    status_code=400
                )
            if len(coords) != len(values):
                raise SimuStructAPIException(
                    message=f"Mismatch in length: got {len(coords)} coordinates and {len(values)} stress values.",
                    code="ARRAY_LENGTH_MISMATCH",
                    status_code=400
                )
            if len(coords) > settings.MAX_COORDINATE_POINTS:
                raise SimuStructAPIException(
                    message=f"Number of points {len(coords)} exceeds maximum limit {settings.MAX_COORDINATE_POINTS}.",
                    code="PAYLOAD_TOO_LARGE",
                    status_code=400
                )
                
            # Rasterize via authoritative research pipeline
            cnorm = normalize_coordinates(coords, L, H)
            grid_raw = rasterize_stress_field(cnorm, values, grid_res=settings.GRID_RESOLUTION, method="nearest")
            grid_norm = grid_raw / traction
            
        else:
            raise SimuStructAPIException(
                message="Must provide either 'stress_grid_64x64' or both 'stress_coordinates' and 'stress_values_psi'.",
                code="MISSING_STRESS_FIELD_INPUT",
                status_code=400
            )
            
        # 3. Model Retrieval & Inference Execution
        model_id = request.model_id or settings.ACTIVE_INVERSE_MODEL_ID
        localizer = model_registry.get_model_instance(model_id)
        model_meta = model_registry.get_model_detail(model_id)
        
        try:
            # Model D5 PyTorch U-Net execution
            x_field = torch.tensor(grid_norm.reshape(1, 1, 64, 64), dtype=torch.float32)
            with torch.no_grad():
                logits = localizer.model(x_field)
                coords_norm, unc_norm, cov_norm = soft_argmax_2d(logits, beta=settings.D5_SOFT_ARGMAX_BETA)
                heatmap_probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
                
            norm_x = float(coords_norm[0, 0].item())
            norm_y = float(coords_norm[0, 1].item())
            pred_x_mm = float(np.clip(norm_x * L, 0.0, L))
            pred_y_mm = float(np.clip(norm_y * H, 0.0, H))
            
            # Anisotropic physical covariance scaling
            c_norm = cov_norm[0].cpu().numpy()
            cov_phys = np.array([
                [c_norm[0, 0] * (L**2), c_norm[0, 1] * (L * H)],
                [c_norm[1, 0] * (L * H), c_norm[1, 1] * (H**2)]
            ], dtype=np.float64)
            
            unc_radius_phys = float(np.sqrt(max(cov_phys[0, 0] + cov_phys[1, 1], 1e-6)))
            
            # Uncertainty Ellipse Geometry
            eigvals, eigvecs = np.linalg.eigh(cov_phys)
            semi_major = float(np.sqrt(max(eigvals[1], 1e-6)))
            semi_minor = float(np.sqrt(max(eigvals[0], 1e-6)))
            angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1])))
            
        except Exception as e:
            raise InferenceExecutionError(
                message=str(e),
                details=[{"model_id": model_id}]
            )
            
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        # 4. Abstention Decision
        is_abstain = bool(unc_radius_phys > threshold_mm)
        abst_status = "FLAGGED_FOR_REVIEW" if is_abstain else "ACCEPTED"
        if is_abstain:
            advice = (
                f"Prediction uncertainty ({unc_radius_phys:.2f} mm) exceeds verified threshold ({threshold_mm:.2f} mm). "
                "Defect is likely in an extreme ligament or boundary interaction regime; manual NDE verification recommended."
            )
            warnings.append(f"[UNCERTAINTY_ABSTENTION_FLAG] {advice}")
        else:
            advice = "Prediction uncertainty lies within verified high-precision operating envelope (<= 25 mm guarantee)."
            
        abst_decision = AbstentionDecision(
            abstain=is_abstain,
            status=abst_status,
            threshold_used_mm=threshold_mm,
            uncertainty_radius_mm=round(unc_radius_phys, 2),
            advice=advice
        )
        
        ellipse_geo = UncertaintyEllipse(
            semi_major_axis_mm=round(semi_major, 2),
            semi_minor_axis_mm=round(semi_minor, 2),
            rotation_angle_deg=round(angle_deg, 2)
        )
        
        # Format serializable heatmap
        heatmap_grid = None
        if request.return_heatmap:
            heatmap_grid = [[round(float(v), 4) for v in row] for row in heatmap_probs]
            
        prediction_id = f"inv_{uuid.uuid4().hex[:12]}"
        
        return InversePredictionResponse(
            success=True,
            prediction_id=prediction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance="ML_SURROGATE",
            operation="INVERSE_SPATIAL_DEFECT_LOCALIZATION",
            model_id=model_id,
            model_version=model_meta.version,
            predicted_defect_x_mm=round(pred_x_mm, 2),
            predicted_defect_y_mm=round(pred_y_mm, 2),
            normalized_xi=round(norm_x, 6),
            normalized_eta=round(norm_y, 6),
            uncertainty_radius_mm=round(unc_radius_phys, 2),
            uncertainty_covariance_mm2=[[round(float(v), 2) for v in r] for r in cov_phys],
            uncertainty_ellipse=ellipse_geo,
            abstention=abst_decision,
            input_mode=input_mode,
            heatmap_grid_64x64=heatmap_grid,
            execution_time_ms=round(elapsed_ms, 2),
            warnings=warnings
        )

inverse_service = InversePredictionService()
