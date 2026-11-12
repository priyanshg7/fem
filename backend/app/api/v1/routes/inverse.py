"""
Inverse defect localization route handlers.
"""
from fastapi import APIRouter

from backend.app.schemas.inverse import InversePredictionRequest, InversePredictionResponse
from backend.app.services.inverse_service import inverse_service

router = APIRouter()

@router.post(
    "/predict/inverse",
    response_model=InversePredictionResponse,
    summary="Localize Defect from Stress Field",
    description="Executes the audited Stage 3D Model D5 U-Net Heatmap Localizer to predict defect coordinates, spatial covariance, and abstention status."
)
def predict_inverse(request: InversePredictionRequest) -> InversePredictionResponse:
    return inverse_service.predict(request)
