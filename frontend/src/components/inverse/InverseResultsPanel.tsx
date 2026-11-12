import React from 'react';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';
import { Alert } from '../common/Alert';
import { InversePredictionResponse, ReferenceGroundTruth } from '../../api/types';
import { exportMatrixAsCsv } from '../../utils/exportUtils';
import {
  FileText,
  Download,
  Clock,
  Crosshair,
  HelpCircle,
  Zap,
  Target,
} from 'lucide-react';

interface InverseResultsPanelProps {
  prediction: InversePredictionResponse | null;
  stressGrid: number[][] | null;
  referenceGroundTruth?: ReferenceGroundTruth;
  onSaveCase: () => void;
  disabled?: boolean;
}

export const InverseResultsPanel: React.FC<InverseResultsPanelProps> = ({
  prediction,
  stressGrid,
  referenceGroundTruth,
  onSaveCase,
  disabled,
}) => {
  if (!prediction) {
    return (
      <Card title="Inverse Defect Localization Results">
        <div className="py-10 text-center space-y-2">
          <Target className="w-8 h-8 text-text-dim mx-auto stroke-1" />
          <div className="text-xs font-semibold text-text-muted">
            Awaiting Inversion Execution
          </div>
          <p className="text-[11px] text-text-dim max-w-sm mx-auto">
            Click &quot;Execute Inverse Localization (D5)&quot; to evaluate the continuous 64×64 U-Net surrogate model and extract defect coordinates.
          </p>
        </div>
      </Card>
    );
  }

  const {
    predicted_defect_x_mm,
    predicted_defect_y_mm,
    normalized_xi,
    normalized_eta,
    uncertainty_radius_mm,
    uncertainty_ellipse,
    abstention,
    execution_time_ms,
    model_id,
    model_version,
    provenance,
  } = prediction;

  const isAccepted = abstention.status === 'ACCEPTED';

  // Compare with reference ground truth if available
  let localizationErrorMm: number | null = null;
  if (
    referenceGroundTruth?.defect_center_x_mm !== undefined &&
    referenceGroundTruth?.defect_center_y_mm !== undefined
  ) {
    const dx = predicted_defect_x_mm - referenceGroundTruth.defect_center_x_mm;
    const dy = predicted_defect_y_mm - referenceGroundTruth.defect_center_y_mm;
    localizationErrorMm = Math.hypot(dx, dy);
  }

  // Export CSVs
  const handleExportHeatmapCsv = () => {
    if (prediction.heatmap_grid_64x64) {
      exportMatrixAsCsv(
        prediction.heatmap_grid_64x64,
        `heatmap_${prediction.prediction_id}.csv`,
        'Y_Row',
        'X_Col'
      );
    }
  };

  const handleExportStressCsv = () => {
    if (stressGrid) {
      exportMatrixAsCsv(stressGrid, `input_stress_${prediction.prediction_id}.csv`, 'Y_Row', 'X_Col');
    }
  };

  return (
    <Card
      title="Inverse Defect Localization Results"
      icon={<Crosshair className="w-4 h-4 text-accent-cyan" />}
      headerAction={
        <div className="flex items-center gap-2">
          <Badge variant="cyan">{provenance}</Badge>
          <Badge variant={isAccepted ? 'green' : 'amber'}>
            {isAccepted ? 'ACCEPTED' : 'FLAGGED_FOR_REVIEW'}
          </Badge>
        </div>
      }
    >
      <div className="space-y-4">
        {/* 1. Primary Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Metric 1: Defect Center */}
          <div className="p-3.5 rounded-lg bg-[#070b14] border border-[#1e293b] space-y-1">
            <span className="text-[11px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
              <Crosshair className="w-3.5 h-3.5 text-accent-cyan" />
              Predicted Defect Center (X, Y)
            </span>
            <div className="text-2xl font-bold font-mono text-accent-cyan">
              ({predicted_defect_x_mm.toFixed(2)}, {predicted_defect_y_mm.toFixed(2)}){' '}
              <span className="text-xs text-text-muted font-normal">mm</span>
            </div>
            <div className="text-[10px] text-text-dim font-mono">
              Normalized: (ξ: {normalized_xi.toFixed(4)}, η: {normalized_eta.toFixed(4)})
            </div>
          </div>

          {/* Metric 2: Spatial Dispersion */}
          <div className="p-3.5 rounded-lg bg-[#070b14] border border-[#1e293b] space-y-1">
            <span className="text-[11px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Spatial Dispersion Radius (ρ)
            </span>
            <div className="text-2xl font-bold font-mono text-amber-400">
              ±{uncertainty_radius_mm.toFixed(2)}{' '}
              <span className="text-xs text-text-muted font-normal">mm</span>
            </div>
            <div className="text-[10px] text-text-dim font-mono">
              Retention Threshold: ≤ {abstention.threshold_used_mm} mm
            </div>
          </div>
        </div>

        {/* 1b. Ground Truth Comparison Callout (if demonstration specimen) */}
        {localizationErrorMm !== null && (
          <div className="p-3 rounded-lg bg-[#0a1322] border border-emerald-500/30 flex items-center justify-between text-xs font-mono">
            <div>
              <span className="text-[10px] text-text-muted block uppercase">Audited Benchmark Ground Truth</span>
              <span className="text-emerald-400 font-bold text-sm">
                ({referenceGroundTruth?.defect_center_x_mm?.toFixed(2)},{' '}
                {referenceGroundTruth?.defect_center_y_mm?.toFixed(2)}) mm
              </span>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-text-muted block uppercase">Localization Error (Δr)</span>
              <span
                className={`font-bold text-sm ${
                  localizationErrorMm <= 25.0 ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                Δr = {localizationErrorMm.toFixed(2)} mm
                {localizationErrorMm <= 25.0 && (
                  <span className="ml-1.5 text-[10px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    &lt; 25 mm TOLERANCE
                  </span>
                )}
              </span>
            </div>
          </div>
        )}

        {/* 2. Dispersion Ellipse & Spatial Covariance */}
        <div className="p-3 rounded-lg bg-[#080c14] border border-[#1e293b] space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-text-muted">
            <span>Heatmap Spatial Covariance 2×2</span>
            <span className="text-[10px] font-mono text-text-dim">Empirical Softmax Spread</span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
            <div className="bg-[#0d131f] p-2 rounded border border-[#1e293b]">
              <span className="text-[9px] text-text-muted block">Semi-Major (a)</span>
              <span className="text-text-primary font-bold">
                {uncertainty_ellipse.semi_major_axis_mm.toFixed(2)} mm
              </span>
            </div>
            <div className="bg-[#0d131f] p-2 rounded border border-[#1e293b]">
              <span className="text-[9px] text-text-muted block">Semi-Minor (b)</span>
              <span className="text-text-primary font-bold">
                {uncertainty_ellipse.semi_minor_axis_mm.toFixed(2)} mm
              </span>
            </div>
            <div className="bg-[#0d131f] p-2 rounded border border-[#1e293b]">
              <span className="text-[9px] text-text-muted block">Orientation (θ)</span>
              <span className="text-text-primary font-bold">
                {uncertainty_ellipse.rotation_angle_deg.toFixed(1)}°
              </span>
            </div>
          </div>

          <div className="text-[10px] text-text-dim leading-relaxed flex items-start gap-1 pt-1">
            <HelpCircle className="w-3.5 h-3.5 text-text-dim shrink-0 mt-0.5" />
            <span>
              The dispersion metric reflects the empirical sharpness of the U-Net softmax heatmap. Because deep neural networks are uncalibrated, it is an engineering filter, <strong>not</strong> a statistically calibrated 95% confidence interval.
            </span>
          </div>
        </div>

        {/* 3. Abstention Status Banner */}
        {isAccepted ? (
          <Alert variant="info" title="Abstention Filter: ACCEPTED">
            <div className="space-y-1 text-xs">
              <p>
                Spatial dispersion ({uncertainty_radius_mm.toFixed(2)} mm) satisfies the audited ≤ 9.07 mm retention criterion.
              </p>
              <p className="text-[11px] text-text-muted">
                Acceptance confirms the model identified a unique, high-contrast stress gradient focal point.
              </p>
            </div>
          </Alert>
        ) : (
          <Alert variant="danger" title="Abstention Filter: FLAGGED FOR REVIEW">
            <div className="space-y-1 text-xs">
              <p>
                Estimated dispersion ({uncertainty_radius_mm.toFixed(2)} mm) exceeds the 9.07 mm retention threshold.
              </p>
              <p className="text-[11px] text-rose-300">
                The defect may lie in a weak perturbation regime or multi-defect interference zone. Independent NDE or numerical FEM verification is required.
              </p>
            </div>
          </Alert>
        )}

        {/* 4. Model Telemetry & Inference Speed */}
        <div className="flex items-center justify-between text-xs text-text-dim px-1 font-mono">
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            Inference: <strong className="text-text-primary">{execution_time_ms.toFixed(1)} ms</strong>
          </span>
          <span>
            Model: <strong className="text-text-primary">{model_id}</strong> (v{model_version})
          </span>
        </div>

        {/* 5. Action Buttons */}
        <div className="pt-2 border-t border-[#1e293b] flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleExportStressCsv}
              disabled={disabled || !stressGrid}
              icon={<Download className="w-3.5 h-3.5" />}
            >
              Export Stress CSV
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleExportHeatmapCsv}
              disabled={disabled || !prediction.heatmap_grid_64x64}
              icon={<Download className="w-3.5 h-3.5" />}
            >
              Export Heatmap CSV
            </Button>
          </div>

          <Button
            variant="primary"
            size="sm"
            onClick={onSaveCase}
            disabled={disabled}
            icon={<FileText className="w-3.5 h-3.5" />}
          >
            Save Case &amp; Generate Report
          </Button>
        </div>
      </div>
    </Card>
  );
};
