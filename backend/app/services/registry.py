"""
Central Model Registry and Lifecycle Manager for SimuStruct ML Backend.
Manages versioning, loading, device assignment, and inference contracts.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.app.core.config import settings
from backend.app.core.errors import ModelNotFoundError, ModelNotReadyError
from backend.app.schemas.models import (
    ModelCategory, ModelSummary, ModelDetail,
    OperatingDomainSpec, InputOutputSpec
)
from backend.app.schemas.health import ModelStatusDetail

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Central Singleton Model Registry for loading, versioning, and managing ML surrogates.
    """
    _instance: Optional["ModelRegistry"] = None
    
    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self._models_meta: Dict[str, Dict[str, Any]] = {}
        self._loaded_instances: Dict[str, Any] = {}
        self._status_details: Dict[str, ModelStatusDetail] = {}
        
        self._register_known_models()
        self._initialized = True
        self.load_all_models()

    def _register_known_models(self):
        """Define specifications for all certified model artifacts."""
        
        # 1. Primary Forward Model: Stage 3B Heterogeneous Ensemble
        self._models_meta["forward_ensemble_stage3b"] = {
            "model_id": "forward_ensemble_stage3b",
            "name": "Stage 3B Forward Multi-Level Meta-Stacking Ensemble",
            "category": ModelCategory.FORWARD,
            "version": "3.2.0-frozen",
            "is_active": True,
            "architecture": "Heterogeneous Stacking (Ridge Meta-Learner over ResMLP, DeepMLP_D2, ExtraTrees, RandomForest, HistGradientBoosting)",
            "description": (
                "Authoritative forward stress prediction ensemble combining deep residual neural networks "
                "and gradient-boosted decision trees. Predicts maximum element-centroid Von Mises stress and Kt."
            ),
            "checkpoint_file": "meta_framework4_heterogeneous.pkl",
            "checkpoint_version": "v3.2-confirmatory-r2-0.9634",
            "feature_contract_version": "v3B-6features-scaled",
            "research_stage": "Stage 3B",
            "validation_benchmark": {
                "confirmatory_R2": 0.9634,
                "confirmatory_RMSE_psi": 4.69,
                "historical_test_R2": 0.9340,
                "historical_test_RMSE_psi": 2.65,
                "evaluation_samples": 500
            },
            "operating_domain": OperatingDomainSpec(
                aspect_ratios_supported=settings.VALIDATED_ASPECT_RATIOS,
                canonical_central_hole_rule="Dc = H / 12.0",
                length_range_mm=[1000.0, 1000.0],
                height_range_mm=[settings.CANONICAL_MIN_HEIGHT_MM, settings.CANONICAL_MAX_HEIGHT_MM],
                defect_diameter_ratio_range=[settings.MIN_DIAMETER_RATIO, settings.MAX_DIAMETER_RATIO],
                min_clearance_factor=settings.MIN_CLEARANCE_MARGIN_FACTOR,
                fixed_material={
                    "material_name": "Structural Steel (Linear Elastic)",
                    "youngs_modulus_GPa": settings.FIXED_YOUNGS_MODULUS_GPA,
                    "poissons_ratio": settings.FIXED_POISSONS_RATIO,
                    "thickness_mm": settings.FIXED_THICKNESS_MM
                },
                fixed_loading={
                    "type": "Uniaxial Longitudinal Tensile Traction",
                    "reference_traction_psi": settings.DEFAULT_TRACTION_PSI
                }
            ),
            "io_specification": InputOutputSpec(
                input_format="Structured Physical Parameters",
                input_dimensions="6 normalized features",
                input_features_or_channels=[
                    "defect_x_over_L", "defect_y_over_H", "defect_d_over_Dc",
                    "aspect_ratio", "ligament_clearance_over_Dc", "min_boundary_clearance_over_H"
                ],
                output_format="Scalar Quantities with Ensemble Dispersion",
                output_quantities=[
                    "predicted_max_stress_psi", "predicted_Kt_element", "ensemble_std_uncertainty_psi"
                ],
                output_units={
                    "predicted_max_stress_psi": "psi",
                    "predicted_Kt_element": "dimensionless (sigma_max / sigma_0)",
                    "ensemble_std_uncertainty_psi": "psi"
                }
            )
        }
        
        # 2. Primary Inverse Model: Stage 3D Model D5 U-Net Heatmap Localizer
        self._models_meta["inverse_unet_d5_stage3d"] = {
            "model_id": "inverse_unet_d5_stage3d",
            "name": "Stage 3D Model D5 Spatial Heatmap U-Net Localizer",
            "category": ModelCategory.INVERSE,
            "version": "3.5.0-audited",
            "is_active": True,
            "architecture": "U-Net with Lateral Skip Connections + Differentiable Soft-Argmax 2D (beta=40.0)",
            "description": (
                "Authoritative full-field inverse defect localizer. Predicts a 2D spatial defect likelihood heatmap "
                "from a continuous normalized Von Mises stress field, computing coordinates and spatial uncertainty."
            ),
            "checkpoint_file": "model_D5_unet.pt",
            "checkpoint_version": "v3.5-audited-logo-8.58mm",
            "feature_contract_version": "v3D-single-channel-64x64-inpainted",
            "research_stage": "Stage 3D & 3E",
            "validation_benchmark": {
                "5_seed_mean_euclidean_mm": 8.43,
                "5_seed_median_euclidean_mm": 5.94,
                "success_rate_le_25mm": 98.8,
                "logo_cross_val_mean_mm": 8.58,
                "untouched_confirmatory_median_mm": 6.00
            },
            "operating_domain": OperatingDomainSpec(
                aspect_ratios_supported=settings.VALIDATED_ASPECT_RATIOS,
                canonical_central_hole_rule="Dc = H / 12.0",
                length_range_mm=[1000.0, 1000.0],
                height_range_mm=[settings.CANONICAL_MIN_HEIGHT_MM, settings.CANONICAL_MAX_HEIGHT_MM],
                defect_diameter_ratio_range=[settings.MIN_DIAMETER_RATIO, settings.MAX_DIAMETER_RATIO],
                min_clearance_factor=settings.MIN_CLEARANCE_MARGIN_FACTOR,
                fixed_material={
                    "material_name": "Structural Steel (Linear Elastic)",
                    "youngs_modulus_GPa": settings.FIXED_YOUNGS_MODULUS_GPA,
                    "poissons_ratio": settings.FIXED_POISSONS_RATIO,
                    "thickness_mm": settings.FIXED_THICKNESS_MM
                },
                fixed_loading={
                    "type": "Uniaxial Longitudinal Tensile Traction",
                    "reference_traction_psi": settings.DEFAULT_TRACTION_PSI
                }
            ),
            "io_specification": InputOutputSpec(
                input_format="Normalized 2D Continuous Stress Tensor (Grid or Scattered Points)",
                input_dimensions="1 x 64 x 64 single-channel float32",
                input_features_or_channels=["normalized_von_mises_stress (sigma_vm / sigma_0)"],
                output_format="Defect Center Coordinates, Spatial Covariance, Heatmap",
                output_quantities=[
                    "predicted_defect_x_mm", "predicted_defect_y_mm", "normalized_xi", "normalized_eta",
                    "uncertainty_radius_mm", "uncertainty_covariance_mm2", "uncertainty_ellipse", "abstention"
                ],
                output_units={
                    "predicted_defect_x_mm": "mm",
                    "predicted_defect_y_mm": "mm",
                    "normalized_xi": "dimensionless in [0, 1]",
                    "normalized_eta": "dimensionless in [0, 1]",
                    "uncertainty_radius_mm": "mm",
                    "uncertainty_covariance_mm2": "mm^2"
                }
            )
        }

    def load_all_models(self):
        """Loads and initializes all registered models into memory."""
        logger.info("Initializing Model Registry and loading certified model checkpoints...")
        
        # Load Forward Model
        fwd_meta = self._models_meta["forward_ensemble_stage3b"]
        t0 = time.perf_counter()
        try:
            from src.stage3B_inference import ForwardStressEnsemblePredictor
            fwd_predictor = ForwardStressEnsemblePredictor(models_dir=settings.STAGE3B_DIR)
            self._loaded_instances["forward_ensemble_stage3b"] = fwd_predictor
            load_ms = (time.perf_counter() - t0) * 1000.0
            self._status_details["forward_ensemble_stage3b"] = ModelStatusDetail(
                loaded=True,
                status="ready",
                architecture=fwd_meta["architecture"],
                checkpoint=fwd_meta["checkpoint_file"],
                device="cpu",
                load_time_ms=round(load_ms, 2)
            )
            logger.info("Forward model loaded successfully in %.2f ms", load_ms)
        except Exception as e:
            logger.error("Failed to load forward model: %s", str(e), exc_info=True)
            self._status_details["forward_ensemble_stage3b"] = ModelStatusDetail(
                loaded=False,
                status="failed",
                architecture=fwd_meta["architecture"],
                checkpoint=fwd_meta["checkpoint_file"],
                device="cpu",
                error_message=str(e)
            )
            
        # Load Inverse Model (Model D5)
        inv_meta = self._models_meta["inverse_unet_d5_stage3d"]
        t0 = time.perf_counter()
        try:
            from src.stage3D_inference import SpatialDefectLocalizer
            inv_localizer = SpatialDefectLocalizer(
                model_type="unet",
                model_path=str(settings.STAGE3D_DIR / "model_D5_unet.pt")
            )
            self._loaded_instances["inverse_unet_d5_stage3d"] = inv_localizer
            load_ms = (time.perf_counter() - t0) * 1000.0
            self._status_details["inverse_unet_d5_stage3d"] = ModelStatusDetail(
                loaded=True,
                status="ready",
                architecture=inv_meta["architecture"],
                checkpoint=inv_meta["checkpoint_file"],
                device=settings.DEVICE,
                load_time_ms=round(load_ms, 2)
            )
            logger.info("Inverse Model D5 loaded successfully in %.2f ms", load_ms)
        except Exception as e:
            logger.error("Failed to load inverse Model D5: %s", str(e), exc_info=True)
            self._status_details["inverse_unet_d5_stage3d"] = ModelStatusDetail(
                loaded=False,
                status="failed",
                architecture=inv_meta["architecture"],
                checkpoint=inv_meta["checkpoint_file"],
                device=settings.DEVICE,
                error_message=str(e)
            )

    def get_model_instance(self, model_id: str) -> Any:
        if model_id not in self._models_meta:
            raise ModelNotFoundError(model_id)
        if model_id not in self._loaded_instances or not self._status_details[model_id].loaded:
            raise ModelNotReadyError(model_id, self._status_details[model_id].error_message or "Not loaded")
        return self._loaded_instances[model_id]

    def get_active_forward_model(self) -> Any:
        return self.get_model_instance(settings.ACTIVE_FORWARD_MODEL_ID)

    def get_active_inverse_model(self) -> Any:
        return self.get_model_instance(settings.ACTIVE_INVERSE_MODEL_ID)

    def list_models(self) -> List[ModelSummary]:
        summaries = []
        for m_id, meta in self._models_meta.items():
            status_val = self._status_details.get(m_id, ModelStatusDetail(
                loaded=False, status="unloaded", architecture=meta["architecture"],
                checkpoint=meta["checkpoint_file"], device="cpu"
            )).status
            summaries.append(ModelSummary(
                model_id=meta["model_id"],
                name=meta["name"],
                category=meta["category"],
                version=meta["version"],
                is_active=meta["is_active"],
                status=status_val,
                architecture=meta["architecture"],
                description=meta["description"]
            ))
        return summaries

    def get_model_detail(self, model_id: str) -> ModelDetail:
        if model_id not in self._models_meta:
            raise ModelNotFoundError(model_id)
        meta = self._models_meta[model_id]
        status_val = self._status_details.get(model_id, ModelStatusDetail(
            loaded=False, status="unloaded", architecture=meta["architecture"],
            checkpoint=meta["checkpoint_file"], device="cpu"
        )).status
        return ModelDetail(
            model_id=meta["model_id"],
            name=meta["name"],
            category=meta["category"],
            version=meta["version"],
            is_active=meta["is_active"],
            status=status_val,
            architecture=meta["architecture"],
            description=meta["description"],
            checkpoint_file=meta["checkpoint_file"],
            checkpoint_version=meta["checkpoint_version"],
            feature_contract_version=meta["feature_contract_version"],
            research_stage=meta["research_stage"],
            validation_benchmark=meta["validation_benchmark"],
            operating_domain=meta["operating_domain"],
            io_specification=meta["io_specification"]
        )

    def get_health_status(self) -> Dict[str, Any]:
        all_ready = all(
            m_id in self._status_details and self._status_details[m_id].loaded
            for m_id in [settings.ACTIVE_FORWARD_MODEL_ID, settings.ACTIVE_INVERSE_MODEL_ID]
        )
        return {
            "ready": all_ready,
            "status": "healthy" if all_ready else "degraded",
            "models": self._status_details
        }

model_registry = ModelRegistry()
