"""
Application configuration for SimuStruct ML Surrogate Backend.
"""
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "SimuStruct ML Surrogate Service"
    API_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "Production-grade FastAPI service exposing verified Stage 3B forward stress prediction "
        "and audited Stage 3D Model D5 spatial inverse defect localization models for perforated plates."
    )
    
    # Environment & Server
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]
    
    # Model Artifact Paths
    MODELS_ROOT: Path = PROJECT_ROOT / "models"
    STAGE3B_DIR: Path = MODELS_ROOT / "stage3B_forward_stacking"
    STAGE3D_DIR: Path = MODELS_ROOT / "stage3D_spatial_inverse"
    
    # Active Primary Models
    ACTIVE_FORWARD_MODEL_ID: str = "forward_ensemble_stage3b"
    ACTIVE_INVERSE_MODEL_ID: str = "inverse_unet_d5_stage3d"
    
    # Hardware Configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")
    NUM_THREADS: int = int(os.getenv("NUM_THREADS", str(min(os.cpu_count() or 4, 8))))
    
    # Mechanical & Operating Domain Constraints
    CANONICAL_LENGTH_MM: float = 1000.0
    CANONICAL_MIN_HEIGHT_MM: float = 600.0
    CANONICAL_MAX_HEIGHT_MM: float = 1000.0
    MIN_ASPECT_RATIO: float = 1.000
    MAX_ASPECT_RATIO: float = 1.667
    VALIDATED_ASPECT_RATIOS: List[float] = [1.000, 1.111, 1.250, 1.429, 1.667]
    CANONICAL_DC_RATIO: float = 1.0 / 12.0 # Dc = H / 12
    MIN_CLEARANCE_MARGIN_FACTOR: float = 0.012 # gm = 0.012 * H
    
    # Defect Diameter Validated Envelope
    MIN_DIAMETER_RATIO: float = 0.08 # d / Dc
    MAX_DIAMETER_RATIO: float = 0.45 # d / Dc
    
    # Baseline Reference Material & Loading
    DEFAULT_TRACTION_PSI: float = 100.0
    FIXED_YOUNGS_MODULUS_GPA: float = 200.0
    FIXED_POISSONS_RATIO: float = 0.30
    FIXED_THICKNESS_MM: float = 1.0
    
    # Uncertainty & Abstention Settings
    D5_ABSTENTION_THRESHOLD_MM: float = 9.07 # Top 5% uncertainty rejection threshold
    D5_SOFT_ARGMAX_BETA: float = 40.0
    GRID_RESOLUTION: int = 64
    
    # Request Payload Safety Limits
    MAX_STRESS_GRID_SIZE: int = 64
    MAX_COORDINATE_POINTS: int = 25000
    
    class Config:
        case_sensitive = True

settings = Settings()
