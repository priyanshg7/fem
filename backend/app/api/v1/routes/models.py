"""
Model catalog and metadata inspection routes.
"""
from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.schemas.models import ModelRegistryResponse, ModelDetail
from backend.app.services.registry import model_registry

router = APIRouter()

@router.get(
    "/models",
    response_model=ModelRegistryResponse,
    summary="List Registered Models",
    description="Lists all certified forward and inverse machine learning models available in the registry."
)
def list_models() -> ModelRegistryResponse:
    summaries = model_registry.list_models()
    return ModelRegistryResponse(
        total_models=len(summaries),
        active_forward_model=settings.ACTIVE_FORWARD_MODEL_ID,
        active_inverse_model=settings.ACTIVE_INVERSE_MODEL_ID,
        models=summaries
    )

@router.get(
    "/models/{model_id}",
    response_model=ModelDetail,
    summary="Get Model Details & Specifications",
    description="Returns detailed metadata, architecture, input-output contracts, and operating domain limits for a model."
)
def get_model_details(model_id: str) -> ModelDetail:
    return model_registry.get_model_detail(model_id)
