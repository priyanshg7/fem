/**
 * SimuStruct ML Backend API Contract Types (v1.0.0)
 * Strictly typed mirror of Stage 4D OpenAPI Schemas.
 */

export interface ModelStatusDetail {
  loaded: boolean;
  status: 'ready' | 'failed' | 'unloaded';
  architecture: string;
  checkpoint: string;
  device: string;
  load_time_ms?: number;
  error_message?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  ready: boolean;
  service: string;
  version: string;
  timestamp: string;
  environment: string;
  active_forward_model: string;
  active_inverse_model: string;
  models: Record<string, ModelStatusDetail>;
}

export interface ReadinessResponse {
  ready: boolean;
  service_healthy: boolean;
  all_models_loaded: boolean;
  details: Record<string, boolean>;
}

export interface ModelSummary {
  model_id: string;
  name: string;
  category: 'forward' | 'inverse';
  version: string;
  is_active: boolean;
  status: string;
  architecture: string;
  description: string;
}

export interface ModelDetail extends ModelSummary {
  checkpoint_file: string;
  checkpoint_version: string;
  feature_contract_version: string;
  research_stage: string;
  validation_benchmark: Record<string, any>;
  operating_domain: {
    aspect_ratios_supported: number[];
    canonical_central_hole_rule: string;
    length_range_mm: [number, number];
    height_range_mm: [number, number];
    defect_diameter_ratio_range: [number, number];
    min_clearance_factor: number;
    fixed_material: {
      material_name: string;
      youngs_modulus_GPa: number;
      poissons_ratio: number;
      thickness_mm: number;
    };
    fixed_loading: {
      type: string;
      reference_traction_psi: number;
    };
  };
  io_specification: {
    input_format: string;
    input_dimensions: string;
    input_features_or_channels: string[];
    output_format: string;
    output_quantities: string[];
    output_units: Record<string, string>;
  };
}

export interface GeometryValidationRequest {
  plate_length_mm: number;
  plate_height_mm: number;
  central_hole_diameter_mm: number;
  defect_diameter_mm: number;
  defect_center_x_mm: number;
  defect_center_y_mm: number;
}

export type ValidationSeverity = 'INFO' | 'WARNING' | 'ERROR';

export interface ValidationIssue {
  field: string;
  code: string;
  message: string;
  severity: ValidationSeverity;
  value?: any;
  constraint?: string;
}

export interface GeometryClearances {
  aspect_ratio: number;
  hole_to_hole_clearance_mm: number;
  min_boundary_clearance_mm: number;
  clearance_to_left_edge_mm: number;
  clearance_to_right_edge_mm: number;
  clearance_to_bottom_edge_mm: number;
  clearance_to_top_edge_mm: number;
  defect_to_central_hole_diameter_ratio: number;
  min_allowed_margin_mm: number;
}

export type ValidationStatus = 'VALID' | 'VALID_WITH_WARNINGS' | 'INVALID';

export interface GeometryValidationResponse {
  status: ValidationStatus;
  is_valid: boolean;
  can_proceed_to_inference: boolean;
  clearances?: GeometryClearances;
  issues: ValidationIssue[];
  canonical_geometry_compliance: boolean;
  operating_domain_summary: Record<string, any>;
}

export interface ForwardPredictionRequest {
  plate_length_mm: number;
  plate_height_mm: number;
  central_hole_diameter_mm: number;
  defect_diameter_mm: number;
  defect_center_x_mm: number;
  defect_center_y_mm: number;
  applied_traction_psi?: number;
  model_id?: string;
  framework?: 'heterogeneous' | 'dnn' | 'pca';
}

export interface ModelInputSummary {
  normalized_coordinates: {
    defect_x_over_L: number;
    defect_y_over_H: number;
  };
  aspect_ratio: number;
  defect_diameter_ratio: number;
  ligament_clearance_ratio: number;
  boundary_clearance_ratio: number;
  applied_traction_psi: number;
  fixed_material: Record<string, any>;
}

export interface ForwardPredictionResponse {
  success: boolean;
  prediction_id: string;
  timestamp: string;
  provenance: 'ML_SURROGATE';
  operation: 'FORWARD_MAX_STRESS_PREDICTION';
  model_id: string;
  model_version: string;
  feature_contract_version: string;
  predicted_max_stress_psi: number;
  predicted_Kt_element: number;
  ensemble_uncertainty_psi: number;
  applied_traction_psi: number;
  base_model_predictions_Kt: Record<string, number>;
  input_summary: ModelInputSummary;
  execution_time_ms: number;
  warnings: string[];
  scope_notice: string;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: any[];
}

export interface UncertaintyEllipse {
  semi_major_axis_mm: number;
  semi_minor_axis_mm: number;
  rotation_angle_deg: number;
}

export interface AbstentionDecision {
  abstain: boolean;
  status: 'ACCEPTED' | 'FLAGGED_FOR_REVIEW';
  threshold_used_mm: number;
  uncertainty_radius_mm: number;
  advice: string;
}

export interface InversePredictionRequest {
  plate_length_mm: number;
  plate_height_mm: number;
  stress_grid_64x64?: number[][];
  stress_coordinates?: number[][];
  stress_values_psi?: number[];
  applied_traction_psi?: number;
  model_id?: string;
  confidence_abstention_threshold_mm?: number;
  return_heatmap?: boolean;
}

export interface InversePredictionResponse {
  success: boolean;
  prediction_id: string;
  timestamp: string;
  provenance: 'ML_SURROGATE';
  operation: 'INVERSE_SPATIAL_DEFECT_LOCALIZATION';
  model_id: string;
  model_version: string;
  predicted_defect_x_mm: number;
  predicted_defect_y_mm: number;
  normalized_xi: number;
  normalized_eta: number;
  uncertainty_radius_mm: number;
  uncertainty_covariance_mm2: number[][];
  uncertainty_ellipse: UncertaintyEllipse;
  abstention: AbstentionDecision;
  input_mode: 'FULL_FIELD_64x64' | 'EXPERIMENTAL_SCATTERED_SENSORS';
  heatmap_grid_64x64?: number[][] | null;
  execution_time_ms: number;
  warnings: string[];
  statistical_disclaimer: string;
}

// Stage 5D: Case Records and Engineering Reports
export type AnalysisType = 'FORWARD' | 'INVERSE' | 'LINKED';
export type StressSourceProvenance =
  | 'EXTERNAL_FEM'
  | 'EXPERIMENTAL_SENSOR'
  | 'RESEARCH_DATASET'
  | 'DEMONSTRATION_SAMPLE';

export interface ReferenceGroundTruth {
  defect_center_x_mm?: number;
  defect_center_y_mm?: number;
  defect_diameter_mm?: number;
  source_notes?: string;
}

export interface CaseRecord {
  case_id: string;
  analysis_type: AnalysisType;
  title: string;
  created_at: string;
  content_hash: string;
  notes?: string;
  is_stale?: boolean;
  api_version: string;
  frontend_version: string;
  provenance: 'ML_SURROGATE';

  // Forward Run (if applicable)
  forward_request?: ForwardPredictionRequest;
  forward_response?: ForwardPredictionResponse;

  // Inverse Run (if applicable)
  inverse_request?: InversePredictionRequest;
  inverse_response?: InversePredictionResponse;
  stress_source_provenance?: StressSourceProvenance;
  stress_source_description?: string;
  reference_ground_truth?: ReferenceGroundTruth;
}

