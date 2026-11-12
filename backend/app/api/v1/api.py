"""
API v1 Router aggregation.
"""
from fastapi import APIRouter

from backend.app.api.v1.routes import health, models, validation, forward, inverse

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Readiness"])
api_router.include_router(models.router, tags=["Model Registry"])
api_router.include_router(validation.router, tags=["Geometry Validation"])
api_router.include_router(forward.router, tags=["Forward Stress Prediction"])
api_router.include_router(inverse.router, tags=["Inverse Defect Localization"])
