"""
Model metadata and registry schemas.
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ModelCategory(str, Enum):
    FORWARD = "forward"
    INVERSE = "inverse"

class ModelSummary(BaseModel):
    model_id: str
    name: str
    category: ModelCategory
    version: str
    is_active: bool
    status: str
    architecture: str
    description: str

class OperatingDomainSpec(BaseModel):
    aspect_ratios_supported: List[float]
    canonical_central_hole_rule: str
    length_range_mm: List[float]
    height_range_mm: List[float]
    defect_diameter_ratio_range: List[float]
    min_clearance_factor: float
    fixed_material: Dict[str, Any]
    fixed_loading: Dict[str, Any]

class InputOutputSpec(BaseModel):
    input_format: str
    input_dimensions: str
    input_features_or_channels: List[str]
    output_format: str
    output_quantities: List[str]
    output_units: Dict[str, str]

class ModelDetail(BaseModel):
    model_id: str
    name: str
    category: ModelCategory
    version: str
    is_active: bool
    status: str
    architecture: str
    description: str
    checkpoint_file: str
    checkpoint_version: str
    feature_contract_version: str
    research_stage: str
    validation_benchmark: Dict[str, Any]
    operating_domain: OperatingDomainSpec
    io_specification: InputOutputSpec

class ModelRegistryResponse(BaseModel):
    total_models: int
    active_forward_model: str
    active_inverse_model: str
    models: List[ModelSummary]
