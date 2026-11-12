"""
Physical geometry validation route handlers.
"""
from fastapi import APIRouter

from backend.app.schemas.validation import GeometryValidationRequest, GeometryValidationResponse
from backend.app.services.validation_service import validation_service

router = APIRouter()

@router.post(
    "/validate/geometry",
    response_model=GeometryValidationResponse,
    summary="Validate Plate and Defect Geometry",
    description="Validates physical non-penetration constraints, canonical dimensions (Dc=H/12), and clearance regimes before running inference."
)
def validate_geometry(request: GeometryValidationRequest) -> GeometryValidationResponse:
    return validation_service.validate(request)
