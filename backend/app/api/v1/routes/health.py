"""
Health check and readiness route handlers.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Response, status

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse, ReadinessResponse
from backend.app.services.registry import model_registry

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Model Status",
    description="Reports the overall health of the service and detailed loading readiness for all registered ML models."
)
def get_health(response: Response) -> HealthResponse:
    health_info = model_registry.get_health_status()
    if not health_info["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return HealthResponse(
        status=health_info["status"],
        ready=health_info["ready"],
        service=settings.PROJECT_NAME,
        version=settings.API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=settings.ENVIRONMENT,
        active_forward_model=settings.ACTIVE_FORWARD_MODEL_ID,
        active_inverse_model=settings.ACTIVE_INVERSE_MODEL_ID,
        models=health_info["models"]
    )

@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Lightweight probe to verify backend availability and model inference readiness."
)
def get_readiness(response: Response) -> ReadinessResponse:
    health_info = model_registry.get_health_status()
    is_ready = health_info["ready"]
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    model_flags = {m_id: detail.loaded for m_id, detail in health_info["models"].items()}
    return ReadinessResponse(
        ready=is_ready,
        service_healthy=True,
        all_models_loaded=is_ready,
        details=model_flags
    )
