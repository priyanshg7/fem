"""
Forward prediction service orchestrator.
Executes the verified Stage 3B Forward Ensemble with rigorous physical validation.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

from backend.app.core.config import settings
from backend.app.core.errors import InvalidGeometryError, InferenceExecutionError
from backend.app.services.registry import model_registry
from backend.app.services.validation_service import validation_service
from backend.app.schemas.validation import GeometryValidationRequest, ValidationSeverity
from backend.app.schemas.forward import (
    ForwardPredictionRequest, ForwardPredictionResponse, ModelInputSummary
)

class ForwardPredictionService:
    @staticmethod
    def predict(request: ForwardPredictionRequest) -> ForwardPredictionResponse:
        t0 = time.perf_counter()
        
        # 1. Physical Input Validation
        val_req = GeometryValidationRequest(
            plate_length_mm=request.plate_length_mm,
            plate_height_mm=request.plate_height_mm,
            central_hole_diameter_mm=request.central_hole_diameter_mm,
            defect_diameter_mm=request.defect_diameter_mm,
            defect_center_x_mm=request.defect_center_x_mm,
            defect_center_y_mm=request.defect_center_y_mm
        )
        val_res = validation_service.validate(val_req)
        
        if not val_res.can_proceed_to_inference:
            err_issues = [i.dict() for i in val_res.issues if i.severity == ValidationSeverity.ERROR]
            raise InvalidGeometryError(
                message="Cannot execute forward prediction: physical geometry collision or domain violation detected.",
                details=err_issues
            )
            
        # Collect operational warnings
        warnings = [
            f"[{issue.code}] {issue.message}"
            for issue in val_res.issues
            if issue.severity == ValidationSeverity.WARNING
        ]
        
        traction = float(request.applied_traction_psi or settings.DEFAULT_TRACTION_PSI)
        if abs(traction - settings.DEFAULT_TRACTION_PSI) > 1e-3:
            warnings.append(
                f"[NON_STANDARD_TRACTION] Applied traction {traction:.1f} psi departs from standard 100.0 psi training baseline; "
                "stress prediction applies linear elastic superposition."
            )
            
        # 2. Select Model & Stacking Framework
        model_id = request.model_id or settings.ACTIVE_FORWARD_MODEL_ID
        predictor = model_registry.get_model_instance(model_id)
        model_meta = model_registry.get_model_detail(model_id)
        framework = request.framework or "heterogeneous"
        
        # 3. Execute Authoritative Research Inference
        try:
            pred_res = predictor.predict(
                plate_length_mm=request.plate_length_mm,
                plate_height_mm=request.plate_height_mm,
                central_hole_diameter_mm=request.central_hole_diameter_mm,
                defect_diameter_mm=request.defect_diameter_mm,
                defect_center_x_mm=request.defect_center_x_mm,
                defect_center_y_mm=request.defect_center_y_mm,
                applied_traction_psi=traction,
                framework=framework
            )
        except Exception as e:
            raise InferenceExecutionError(
                message=str(e),
                details=[{"model_id": model_id, "framework": framework}]
            )
            
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        # 4. Construct Input Summary
        L = request.plate_length_mm
        H = request.plate_height_mm
        Dc = request.central_hole_diameter_mm
        d = request.defect_diameter_mm
        c_lig = val_res.clearances.hole_to_hole_clearance_mm if val_res.clearances else 0.0
        min_bnd = val_res.clearances.min_boundary_clearance_mm if val_res.clearances else 0.0
        
        input_summary = ModelInputSummary(
            normalized_coordinates={
                "defect_x_over_L": round(request.defect_center_x_mm / L, 6),
                "defect_y_over_H": round(request.defect_center_y_mm / H, 6)
            },
            aspect_ratio=round(L / H, 4),
            defect_diameter_ratio=round(d / Dc, 4),
            ligament_clearance_ratio=round(c_lig / Dc, 4),
            boundary_clearance_ratio=round(min_bnd / H, 4),
            applied_traction_psi=traction,
            fixed_material=model_meta.operating_domain.fixed_material
        )
        
        prediction_id = f"fwd_{uuid.uuid4().hex[:12]}"
        
        return ForwardPredictionResponse(
            success=True,
            prediction_id=prediction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance="ML_SURROGATE",
            operation="FORWARD_MAX_STRESS_PREDICTION",
            model_id=model_id,
            model_version=model_meta.version,
            feature_contract_version=model_meta.feature_contract_version,
            predicted_max_stress_psi=round(pred_res["predicted_max_stress_psi"], 3),
            predicted_Kt_element=round(pred_res["predicted_Kt_element"], 4),
            ensemble_uncertainty_psi=round(pred_res["ensemble_std_uncertainty_psi"], 3),
            applied_traction_psi=traction,
            base_model_predictions_Kt={k: round(v, 4) for k, v in pred_res["base_predictions_Kt"].items()},
            input_summary=input_summary,
            execution_time_ms=round(elapsed_ms, 2),
            warnings=warnings
        )

forward_service = ForwardPredictionService()
