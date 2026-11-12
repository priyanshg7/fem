"""
Forward stress prediction schemas.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ForwardPredictionRequest(BaseModel):
    plate_length_mm: float = Field(..., gt=0, example=1000.0, description="Plate length in mm")
    plate_height_mm: float = Field(..., gt=0, example=800.0, description="Plate height in mm")
    central_hole_diameter_mm: float = Field(..., gt=0, example=66.67, description="Central hole diameter in mm (canonical Dc = H/12)")
    defect_diameter_mm: float = Field(..., gt=0, example=12.0, description="Defect hole diameter in mm")
    defect_center_x_mm: float = Field(..., example=250.0, description="Defect X coordinate in mm")
    defect_center_y_mm: float = Field(..., example=300.0, description="Defect Y coordinate in mm")
    applied_traction_psi: Optional[float] = Field(default=100.0, gt=0, example=100.0, description="Far-field tensile traction in psi")
    model_id: Optional[str] = Field(default=None, description="Optional specific model ID (defaults to active forward model)")
    framework: Optional[str] = Field(default="heterogeneous", description="Stacking framework ('heterogeneous', 'dnn', 'pca')")

class ModelInputSummary(BaseModel):
    normalized_coordinates: Dict[str, float]
    aspect_ratio: float
    defect_diameter_ratio: float
    ligament_clearance_ratio: float
    boundary_clearance_ratio: float
    applied_traction_psi: float
    fixed_material: Dict[str, Any]

class ForwardPredictionResponse(BaseModel):
    success: bool = True
    prediction_id: str
    timestamp: str
    provenance: str = "ML_SURROGATE"
    operation: str = "FORWARD_MAX_STRESS_PREDICTION"
    model_id: str
    model_version: str
    feature_contract_version: str
    predicted_max_stress_psi: float = Field(..., description="Predicted maximum element-centroid Von Mises stress in psi")
    predicted_Kt_element: float = Field(..., description="Predicted stress concentration factor Kt = sigma_max / sigma_0")
    ensemble_uncertainty_psi: float = Field(..., description="Ensemble dispersion standard deviation across base estimators in psi")
    applied_traction_psi: float
    base_model_predictions_Kt: Dict[str, float]
    input_summary: ModelInputSummary
    execution_time_ms: float
    warnings: List[str] = []
    scope_notice: str = (
        "This forward surrogate predicts scalar maximum element-centroid Von Mises stress. "
        "It does not reconstruct complete 2D spatial stress fields or replace numerical FEM convergence analysis."
    )
