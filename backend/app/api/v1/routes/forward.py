"""
Forward stress prediction route handlers.
"""
from fastapi import APIRouter

from backend.app.schemas.forward import ForwardPredictionRequest, ForwardPredictionResponse
from backend.app.services.forward_service import forward_service

router = APIRouter()

@router.post(
    "/predict/forward",
    response_model=ForwardPredictionResponse,
    summary="Predict Maximum Von Mises Stress",
    description="Executes the verified Stage 3B Forward Ensemble to predict the maximum stress and Kt for a known defect configuration."
)
def predict_forward(request: ForwardPredictionRequest) -> ForwardPredictionResponse:
    return forward_service.predict(request)
