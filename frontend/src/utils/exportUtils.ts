/**
 * Export Utilities for SimuStruct Engineering Reports and Data
 * Generates machine-readable JSON, CSV matrix tables, and print-ready formats.
 */
import { CaseRecord } from '../api/types';

export function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportCaseAsJson(record: CaseRecord) {
  const jsonStr = JSON.stringify(record, null, 2);
  const filename = `${record.case_id}.json`;
  downloadBlob(jsonStr, filename, 'application/json');
}

export function exportMatrixAsCsv(
  matrix: number[][],
  filename: string,
  rowLabelPrefix = 'Row',
  colLabelPrefix = 'Col'
) {
  if (!matrix || matrix.length === 0) return;
  const numRows = matrix.length;
  const numCols = matrix[0].length;

  const header = ['Axis', ...Array.from({ length: numCols }, (_, c) => `${colLabelPrefix}_${c}`)].join(',');
  const lines = [header];

  for (let r = 0; r < numRows; r++) {
    const rowValues = matrix[r].map((v) => (typeof v === 'number' ? v.toFixed(4) : String(v)));
    lines.push([`${rowLabelPrefix}_${r}`, ...rowValues].join(','));
  }

  downloadBlob(lines.join('\n'), filename, 'text/csv;charset=utf-8;');
}

export function exportTabularSummaryCsv(record: CaseRecord) {
  const lines: string[] = ['Parameter,Value,Unit,Provenance'];
  lines.push(`Case ID,${record.case_id},-,System Generated`);
  lines.push(`Analysis Type,${record.analysis_type},-,Input`);
  lines.push(`Created At,${record.created_at},UTC,System Timestamp`);
  lines.push(`Integrity Hash,${record.content_hash},SHA-256,Client Verification`);

  if (record.forward_request && record.forward_response) {
    const req = record.forward_request;
    const res = record.forward_response;
    lines.push(`Plate Length,${req.plate_length_mm},mm,Nominal Specimen`);
    lines.push(`Plate Height,${req.plate_height_mm},mm,Nominal Specimen`);
    lines.push(`Central Hole Diameter,${req.central_hole_diameter_mm},mm,Nominal Specimen`);
    lines.push(`Defect Center X,${req.defect_center_x_mm},mm,Input Coordinate`);
    lines.push(`Defect Center Y,${req.defect_center_y_mm},mm,Input Coordinate`);
    lines.push(`Defect Diameter,${req.defect_diameter_mm},mm,Input Dimension`);
    lines.push(`Applied Traction,${req.applied_traction_psi ?? 100.0},psi,Boundary Load`);
    lines.push(`Predicted Max Stress,${res.predicted_max_stress_psi},psi,ML_SURROGATE`);
    lines.push(`Predicted Kt,${res.predicted_Kt_element},dimensionless,ML_SURROGATE`);
    lines.push(`Ensemble Uncertainty,${res.ensemble_uncertainty_psi},psi,ML_SURROGATE Dispersion`);
    lines.push(`Model ID,${res.model_id},-,Stage 3B Stacking Ensemble`);
    lines.push(`Execution Time,${res.execution_time_ms},ms,Server Telemetry`);
  }

  if (record.inverse_request && record.inverse_response) {
    const req = record.inverse_request;
    const res = record.inverse_response;
    lines.push(`Plate Length,${req.plate_length_mm},mm,Nominal Specimen`);
    lines.push(`Plate Height,${req.plate_height_mm},mm,Nominal Specimen`);
    lines.push(`Applied Traction,${req.applied_traction_psi ?? 100.0},psi,Reference Load`);
    lines.push(`Stress Source,${record.stress_source_provenance || 'EXTERNAL'},-,User Metadata`);
    lines.push(`Predicted Defect X,${res.predicted_defect_x_mm},mm,ML_SURROGATE`);
    lines.push(`Predicted Defect Y,${res.predicted_defect_y_mm},mm,ML_SURROGATE`);
    lines.push(`Normalized Xi,${res.normalized_xi},[0-1],ML_SURROGATE`);
    lines.push(`Normalized Eta,${res.normalized_eta},[0-1],ML_SURROGATE`);
    lines.push(`Spatial Dispersion Radius,${res.uncertainty_radius_mm},mm,Empirical Heatmap Dispersion`);
    lines.push(`Abstention Decision,${res.abstention.status},-,Threshold Rule`);
    lines.push(`Abstention Advice,"${res.abstention.advice.replace(/"/g, '""')}",-,Safety System`);
    lines.push(`Model ID,${res.model_id},-,Stage 3D Model D5 U-Net`);
    lines.push(`Execution Time,${res.execution_time_ms},ms,Server Telemetry`);
  }

  downloadBlob(lines.join('\n'), `${record.case_id}_summary.csv`, 'text/csv;charset=utf-8;');
}
