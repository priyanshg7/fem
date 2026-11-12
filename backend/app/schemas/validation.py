"""
Geometry validation schemas.
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"

class ValidationStatus(str, Enum):
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INVALID = "INVALID"

class ValidationIssue(BaseModel):
    field: str
    code: str
    message: str
    severity: ValidationSeverity
    value: Any
    constraint: Optional[str] = None

class GeometryValidationRequest(BaseModel):
    plate_length_mm: float = Field(..., gt=0, example=1000.0, description="Plate length in mm (default canonical: 1000.0)")
    plate_height_mm: float = Field(..., gt=0, example=800.0, description="Plate height in mm (canonical: 600.0 to 1000.0)")
    central_hole_diameter_mm: float = Field(..., gt=0, example=66.67, description="Central circular hole diameter in mm (canonical Dc = H/12)")
    defect_diameter_mm: float = Field(..., gt=0, example=12.0, description="Defect hole diameter in mm")
    defect_center_x_mm: float = Field(..., example=250.0, description="Defect center X coordinate in mm")
    defect_center_y_mm: float = Field(..., example=300.0, description="Defect center Y coordinate in mm")

class GeometryClearances(BaseModel):
    aspect_ratio: float
    hole_to_hole_clearance_mm: float
    min_boundary_clearance_mm: float
    clearance_to_left_edge_mm: float
    clearance_to_right_edge_mm: float
    clearance_to_bottom_edge_mm: float
    clearance_to_top_edge_mm: float
    defect_to_central_hole_diameter_ratio: float
    min_allowed_margin_mm: float

class GeometryValidationResponse(BaseModel):
    status: ValidationStatus
    is_valid: bool
    can_proceed_to_inference: bool
    clearances: Optional[GeometryClearances] = None
    issues: List[ValidationIssue] = []
    canonical_geometry_compliance: bool
    operating_domain_summary: Dict[str, Any]
