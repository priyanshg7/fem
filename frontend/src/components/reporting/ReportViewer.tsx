import React from 'react';
import { CaseRecord } from '../../api/types';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import { exportCaseAsJson, exportTabularSummaryCsv } from '../../utils/exportUtils';
import { Printer, Download, ArrowLeft, FileSpreadsheet } from 'lucide-react';

interface ReportViewerProps {
  caseRecord: CaseRecord;
  onClose: () => void;
}

export const ReportViewer: React.FC<ReportViewerProps> = ({ caseRecord, onClose }) => {
  const isForward = caseRecord.analysis_type === 'FORWARD' || caseRecord.analysis_type === 'LINKED';
  const isInverse = caseRecord.analysis_type === 'INVERSE' || caseRecord.analysis_type === 'LINKED';

  const fwdReq = caseRecord.forward_request;
  const fwdRes = caseRecord.forward_response;
  const invReq = caseRecord.inverse_request;
  const invRes = caseRecord.inverse_response;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/85 backdrop-blur-sm flex justify-center p-4 sm:p-6 print:p-0 print:bg-white print:static print:inset-auto">
      <div className="w-full max-w-4xl bg-[#0f141f] border border-border rounded-xl shadow-2xl flex flex-col overflow-hidden text-text-primary print:border-none print:shadow-none print:bg-white print:text-black">
        {/* Top Action Bar (Hidden during print) */}
        <div className="bg-[#141b2a] border-b border-border px-6 py-3.5 flex items-center justify-between print:hidden">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={onClose} icon={<ArrowLeft className="w-4 h-4" />}>
              Back to Workspace
            </Button>
            <span className="text-xs font-mono text-text-muted">|</span>
            <span className="font-mono text-xs font-semibold text-accent-cyan">
              {caseRecord.case_id}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => exportTabularSummaryCsv(caseRecord)}
              icon={<FileSpreadsheet className="w-3.5 h-3.5" />}
            >
              Export CSV
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => exportCaseAsJson(caseRecord)}
              icon={<Download className="w-3.5 h-3.5" />}
            >
              Export JSON
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={handlePrint}
              icon={<Printer className="w-3.5 h-3.5" />}
            >
              Print / Save PDF
            </Button>
          </div>
        </div>

        {/* Printable Report Body */}
        <div className="p-8 space-y-6 print:p-6 print:space-y-4 font-sans text-sm leading-relaxed">
          {/* Header Title Section */}
          <div className="border-b border-border/80 pb-5 print:border-black/30">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-mono tracking-widest text-accent-cyan uppercase font-semibold print:text-blue-800">
                  SimuStruct Digital Twin Platform • Engineering Audit Record
                </span>
                <h1 className="text-2xl font-bold mt-1 text-text-primary print:text-black">
                  {caseRecord.title}
                </h1>
                <p className="text-xs text-text-muted mt-1 print:text-gray-600">
                  Machine Learning Surrogate Structural Evaluation Report
                </p>
              </div>
              <div className="text-right font-mono text-xs text-text-muted print:text-gray-700">
                <div className="font-bold text-accent-cyan print:text-black">{caseRecord.case_id}</div>
                <div>Generated: {new Date(caseRecord.created_at).toUTCString()}</div>
                <div>Hash: {caseRecord.content_hash}</div>
              </div>
            </div>

            {/* Badges Row */}
            <div className="flex flex-wrap items-center gap-2 mt-4">
              <Badge variant="cyan">{caseRecord.provenance}</Badge>
              <Badge variant="green">Analysis: {caseRecord.analysis_type}</Badge>
              <span className="text-xs text-text-muted font-mono print:text-gray-600">
                API Version: {caseRecord.api_version} • System Core: v{caseRecord.frontend_version}
              </span>
            </div>
          </div>

          {/* Executive Summary */}
          <div className="bg-[#121927] border border-border/60 p-4 rounded-lg print:bg-gray-50 print:border-gray-300">
            <h3 className="text-xs font-bold uppercase tracking-wider text-text-secondary print:text-gray-800 mb-1">
              Executive Engineering Summary
            </h3>
            <p className="text-xs text-text-muted leading-relaxed print:text-gray-700">
              This traceable case record archives computational surrogate predictions produced by the frozen SimuStruct
              machine-learning models. Results reflect empirical surrogate estimations under linear elastic boundary
              conditions and must be interpreted in conjunction with standard structural verification procedures.
            </p>
          </div>

          {/* Section A: Forward Analysis Record (if present) */}
          {isForward && fwdReq && fwdRes && (
            <div className="space-y-3 pt-2">
              <h2 className="text-sm font-bold uppercase tracking-wider text-accent-cyan print:text-blue-900 border-b border-border/60 pb-1.5 print:border-gray-300">
                1. Forward Stress Concentration Analysis (Stage 3B Stacking Ensemble)
              </h2>

              <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                {/* Inputs Table */}
                <div className="border border-border/60 rounded p-3 print:border-gray-300">
                  <h4 className="font-sans font-semibold text-text-secondary mb-2 print:text-black">
                    Specimen Geometry & Loading
                  </h4>
                  <table className="w-full text-[11px]">
                    <tbody>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Plate Length (L):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{fwdReq.plate_length_mm} mm</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Plate Height (H):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{fwdReq.plate_height_mm} mm</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Aspect Ratio (L/H):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {(fwdReq.plate_length_mm / fwdReq.plate_height_mm).toFixed(3)}
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Central Cutout (Dc):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {fwdReq.central_hole_diameter_mm.toFixed(2)} mm
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Defect Position (x, y):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          ({fwdReq.defect_center_x_mm.toFixed(1)}, {fwdReq.defect_center_y_mm.toFixed(1)}) mm
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Defect Diameter (d):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{fwdReq.defect_diameter_mm.toFixed(1)} mm</td>
                      </tr>
                      <tr>
                        <td className="py-1 text-text-muted">Applied Traction (σ₀):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {fwdReq.applied_traction_psi ?? 100.0} psi
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Outputs Table */}
                <div className="border border-border/60 rounded p-3 print:border-gray-300">
                  <h4 className="font-sans font-semibold text-text-secondary mb-2 print:text-black">
                    Surrogate Prediction Metrics
                  </h4>
                  <table className="w-full text-[11px]">
                    <tbody>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Max Von Mises Stress (σ_max):</td>
                        <td className="py-1 text-right text-accent-cyan font-bold print:text-black">
                          {fwdRes.predicted_max_stress_psi.toFixed(2)} psi
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Stress Concentration (Kt):</td>
                        <td className="py-1 text-right text-text-primary font-bold print:text-black">
                          {fwdRes.predicted_Kt_element.toFixed(3)}
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Ensemble Dispersion (±std):</td>
                        <td className="py-1 text-right text-amber-400 print:text-gray-800">
                          ±{fwdRes.ensemble_uncertainty_psi.toFixed(2)} psi
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Ensemble Model:</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{fwdRes.model_id}</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Model Checkpoint:</td>
                        <td className="py-1 text-right text-text-primary print:text-black">v{fwdRes.model_version}</td>
                      </tr>
                      <tr>
                        <td className="py-1 text-text-muted">Execution Latency:</td>
                        <td className="py-1 text-right text-emerald-400 print:text-gray-800">
                          {fwdRes.execution_time_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Base Estimators Breakdown */}
              {fwdRes.base_model_predictions_Kt && (
                <div className="p-3 rounded bg-surface-elevated/40 border border-border/40 print:bg-white print:border-gray-300">
                  <span className="text-[11px] font-semibold text-text-secondary block mb-1.5 print:text-black">
                    Constituent Base Model Estimates (Kt):
                  </span>
                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-[10px] font-mono text-center">
                    {Object.entries(fwdRes.base_model_predictions_Kt).map(([name, val]) => (
                      <div key={name} className="p-1 rounded bg-[#0a0d14] border border-border/40 print:border-gray-200">
                        <div className="text-text-muted truncate">{name}</div>
                        <div className="font-bold text-text-primary print:text-black">{Number(val).toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Section B: Inverse Analysis Record (if present) */}
          {isInverse && invReq && invRes && (
            <div className="space-y-3 pt-2">
              <h2 className="text-sm font-bold uppercase tracking-wider text-accent-cyan print:text-blue-900 border-b border-border/60 pb-1.5 print:border-gray-300">
                2. Inverse Defect Localization Analysis (Stage 3D Model D5 U-Net)
              </h2>

              <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                {/* Inputs & Source Table */}
                <div className="border border-border/60 rounded p-3 print:border-gray-300">
                  <h4 className="font-sans font-semibold text-text-secondary mb-2 print:text-black">
                    Stress Field Input & Provenance
                  </h4>
                  <table className="w-full text-[11px]">
                    <tbody>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Plate Length (L):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{invReq.plate_length_mm} mm</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Plate Height (H):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{invReq.plate_height_mm} mm</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Reference Traction (σ₀):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {invReq.applied_traction_psi ?? 100.0} psi
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Input Mode:</td>
                        <td className="py-1 text-right text-text-primary print:text-black">{invRes.input_mode}</td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Stress Source:</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {caseRecord.stress_source_provenance || 'EXTERNAL'}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-1 text-text-muted">Source Description:</td>
                        <td className="py-1 text-right text-text-muted truncate max-w-[150px] print:text-gray-700">
                          {caseRecord.stress_source_description || 'Unspecified'}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Outputs Table */}
                <div className="border border-border/60 rounded p-3 print:border-gray-300">
                  <h4 className="font-sans font-semibold text-text-secondary mb-2 print:text-black">
                    Predicted Defect Location & Uncertainty
                  </h4>
                  <table className="w-full text-[11px]">
                    <tbody>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Predicted Defect Center (X, Y):</td>
                        <td className="py-1 text-right text-accent-cyan font-bold print:text-black">
                          ({invRes.predicted_defect_x_mm.toFixed(2)}, {invRes.predicted_defect_y_mm.toFixed(2)}) mm
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Normalized Position (ξ, η):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          ({invRes.normalized_xi.toFixed(4)}, {invRes.normalized_eta.toFixed(4)})
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Heatmap Spatial Dispersion (ρ):</td>
                        <td className="py-1 text-right text-amber-400 font-bold print:text-gray-800">
                          ±{invRes.uncertainty_radius_mm.toFixed(2)} mm
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Dispersion Ellipse (a, b, θ):</td>
                        <td className="py-1 text-right text-text-primary print:text-black">
                          {invRes.uncertainty_ellipse.semi_major_axis_mm.toFixed(1)} mm,{' '}
                          {invRes.uncertainty_ellipse.semi_minor_axis_mm.toFixed(1)} mm,{' '}
                          {invRes.uncertainty_ellipse.rotation_angle_deg.toFixed(1)}°
                        </td>
                      </tr>
                      <tr className="border-b border-border/30">
                        <td className="py-1 text-text-muted">Abstention Status:</td>
                        <td className="py-1 text-right text-text-primary font-bold print:text-black">
                          {invRes.abstention.status}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-1 text-text-muted">Execution Latency:</td>
                        <td className="py-1 text-right text-emerald-400 print:text-gray-800">
                          {invRes.execution_time_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Reference Truth Verification Note if present */}
              {caseRecord.reference_ground_truth?.defect_center_x_mm !== undefined &&
                caseRecord.reference_ground_truth?.defect_center_y_mm !== undefined && (
                  <div className="p-3 rounded bg-emerald-950/20 border border-emerald-500/30 text-xs font-mono text-emerald-300 print:bg-green-50 print:border-green-300 print:text-green-900">
                    <span className="font-bold block">
                      Demonstration Reference Ground Truth Comparison:
                    </span>
                    Truth Coordinate: ({caseRecord.reference_ground_truth.defect_center_x_mm.toFixed(2)},{' '}
                    {caseRecord.reference_ground_truth.defect_center_y_mm.toFixed(2)}) mm • Defect Diameter:{' '}
                    {caseRecord.reference_ground_truth.defect_diameter_mm?.toFixed(2)} mm • Localization Error: Δr ={' '}
                    {Math.hypot(
                      invRes.predicted_defect_x_mm - caseRecord.reference_ground_truth.defect_center_x_mm,
                      invRes.predicted_defect_y_mm - caseRecord.reference_ground_truth.defect_center_y_mm
                    ).toFixed(2)}{' '}
                    mm
                  </div>
                )}

              {/* Abstention Advice Banner */}
              <div
                className={`p-3 rounded text-xs leading-relaxed border ${
                  invRes.abstention.status === 'ACCEPTED'
                    ? 'bg-blue-950/20 border-blue-500/30 text-blue-200 print:bg-blue-50 print:text-blue-900'
                    : 'bg-amber-950/20 border-amber-500/30 text-amber-200 print:bg-amber-50 print:text-amber-900'
                }`}
              >
                <div className="font-bold mb-0.5">
                  Safety System Decision: {invRes.abstention.status} (Threshold: {invRes.abstention.threshold_used_mm} mm)
                </div>
                <div>{invRes.abstention.advice}</div>
                <div className="mt-1 text-[10px] text-text-muted print:text-gray-600">
                  {invRes.statistical_disclaimer}
                </div>
              </div>
            </div>
          )}

          {/* Section C: Scientific Governance and Limitations */}
          <div className="border-t border-border/80 pt-4 space-y-2 text-xs text-text-muted print:border-gray-300 print:text-gray-600 leading-relaxed">
            <h4 className="font-bold text-text-secondary uppercase tracking-wider text-[11px] print:text-black">
              Scientific Disclaimers & Operational Limitations
            </h4>
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <strong>Machine Learning Surrogate Provenance:</strong> The reported stresses and defect coordinates
                are derived from frozen statistical surrogate models (Stage 3B Stacking Ensemble and Stage 3D Model D5 U-Net),
                not direct numerical solves (FEM) of the boundary value problem.
              </li>
              <li>
                <strong>Spatial Dispersion Semantics:</strong> The reported uncertainty radius and ellipse reflect empirical
                heatmap sharpness. They do not constitute statistically calibrated 95% confidence intervals or guaranteed
                tolerance limits.
              </li>
              <li>
                <strong>Structural Governance:</strong> This report is for engineering evaluation, research audit, and
                screening purposes. It does not certify structural fitness-for-service without independent numerical FEM
                analysis and physical non-destructive inspection (NDI).
              </li>
            </ul>
          </div>

          {/* Sign-Off Footer */}
          <div className="border-t border-border/60 pt-4 flex items-center justify-between text-[11px] font-mono text-text-muted print:text-gray-600">
            <span>SimuStruct Digital Twin Platform</span>
            <span>Record Hash: {caseRecord.content_hash} • End of Report</span>
          </div>
        </div>
      </div>
    </div>
  );
};
