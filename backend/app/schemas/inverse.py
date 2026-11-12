"""
Inverse defect localization schemas.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class InversePredictionRequest(BaseModel):
    plate_length_mm: float = Field(..., gt=0, example=1000.0, description="Plate length in mm (L)")
    plate_height_mm: float = Field(..., gt=0, example=800.0, description="Plate height in mm (H)")
    
    # Stress Field Input: Either 64x64 grid OR scattered coordinates + values
    stress_grid_64x64: Optional[List[List[float]]] = Field(
        default=None,
        description="64x64 2D matrix of observed Von Mises stress in psi (continuous inpainted field)"
    )
    stress_coordinates: Optional[List[List[float]]] = Field(
        default=None,
        description="Scattered measurement coordinates [x, y] in mm (N x 2)"
    )
    stress_values_psi: Optional[List[float]] = Field(
        default=None,
        description="Corresponding Von Mises stress values at coordinates in psi (length N)"
    )
    
    applied_traction_psi: Optional[float] = Field(default=100.0, gt=0, description="Reference far-field traction for normalization (default: 100.0 psi)")
    model_id: Optional[str] = Field(default=None, description="Optional specific model ID (defaults to active inverse model)")
    confidence_abstention_threshold_mm: Optional[float] = Field(
        default=9.07,
        gt=0,
        description="Uncertainty radius threshold for abstention recommendation in mm (default: 9.07 mm, audited 95% retention)"
    )
    return_heatmap: Optional[bool] = Field(default=True, description="Whether to include full 64x64 predicted heatmap matrix in response")

class UncertaintyEllipse(BaseModel):
    semi_major_axis_mm: float
    semi_minor_axis_mm: float
    rotation_angle_deg: float

class AbstentionDecision(BaseModel):
    abstain: bool = Field(..., description="True if uncertainty exceeds threshold; client advised to withhold automated decision")
    status: str = Field(..., description="'ACCEPTED' or 'FLAGGED_FOR_REVIEW'")
    threshold_used_mm: float
    uncertainty_radius_mm: float
    advice: str

class InversePredictionResponse(BaseModel):
    success: bool = True
    prediction_id: str
    timestamp: str
    provenance: str = "ML_SURROGATE"
    operation: str = "INVERSE_SPATIAL_DEFECT_LOCALIZATION"
    model_id: str
    model_version: str
    predicted_defect_x_mm: float = Field(..., description="Predicted defect center X coordinate in mm")
    predicted_defect_y_mm: float = Field(..., description="Predicted defect center Y coordinate in mm")
    normalized_xi: float = Field(..., description="Predicted dimensionless X/L in [0, 1]")
    normalized_eta: float = Field(..., description="Predicted dimensionless Y/H in [0, 1]")
    uncertainty_radius_mm: float = Field(..., description="Estimated physical spatial uncertainty radius sqrt(Var(X) + Var(Y)) in mm")
    uncertainty_covariance_mm2: List[List[float]] = Field(..., description="2x2 physical covariance matrix in mm^2")
    uncertainty_ellipse: UncertaintyEllipse
    abstention: AbstentionDecision
    input_mode: str = Field(
        default="FULL_FIELD_64x64",
        description="Operational input mode: 'FULL_FIELD_64x64' (audited benchmarked mode) or 'EXPERIMENTAL_SCATTERED_SENSORS' (experimental interpolation mode)"
    )
    heatmap_grid_64x64: Optional[List[List[float]]] = None
    execution_time_ms: float
    warnings: List[str] = []
    statistical_disclaimer: str = (
        "Notice: The spatial uncertainty radius and covariance ellipse represent empirical heatmap dispersion. "
        "Due to audited nominal under-coverage in raw deep softmax outputs, they must not be interpreted as statistically calibrated 95% confidence intervals."
    )
