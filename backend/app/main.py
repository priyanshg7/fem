"""
SimuStruct ML Surrogate Backend Service: Main FastAPI Application Entrypoint.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.errors import (
    SimuStructAPIException,
    simustruct_exception_handler,
    generic_exception_handler
)
from backend.app.api.v1.api import api_router
from backend.app.services.registry import model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("simustruct_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads and checks all certified models on startup."""
    logger.info("Starting up SimuStruct ML Backend...")
    model_registry.load_all_models()
    yield
    logger.info("Shutting down SimuStruct ML Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(SimuStructAPIException, simustruct_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include v1 API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Production Single-Origin Static Serving (React SPA)
from pathlib import Path
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

project_root = Path(__file__).resolve().parent.parent.parent
frontend_dist = project_root / "frontend" / "dist"

if frontend_dist.exists() and (frontend_dist / "assets").exists():
    logger.info(f"Mounting compiled frontend SPA assets from: {frontend_dist / 'assets'}")
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static_assets")

@app.get("/", tags=["Root"])
def root(request: Request):
    accept = request.headers.get("accept", "")
    # If a browser requests HTML at root, serve compiled React index.html
    if "text/html" in accept and frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(frontend_dist / "index.html")
    
    # Default API Metadata for programmatic clients and test suites
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "status": "online",
        "documentation": "/docs",
        "api_v1": settings.API_V1_STR,
        "health_check": f"{settings.API_V1_STR}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
