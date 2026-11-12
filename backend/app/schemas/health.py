"""
Health check and system readiness schemas.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelStatusDetail(BaseModel):
    loaded: bool
    status: str = Field(..., description="'ready', 'failed', or 'unloaded'")
    architecture: str
    checkpoint: str
    device: str
    load_time_ms: Optional[float] = None
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    ready: bool = Field(..., description="True if all primary models are ready for inference")
    service: str = "SimuStruct ML Surrogate Backend"
    version: str
    timestamp: str
    environment: str
    active_forward_model: str
    active_inverse_model: str
    models: Dict[str, ModelStatusDetail]

class ReadinessResponse(BaseModel):
    ready: bool
    service_healthy: bool
    all_models_loaded: bool
    details: Dict[str, bool]
