import React, { useEffect } from 'react';
import { useInverseAnalysis } from '../../hooks/useInverseAnalysis';
import { StressFieldInput } from './StressFieldInput';
import { HeatmapViewer } from './HeatmapViewer';
import { InverseResultsPanel } from './InverseResultsPanel';
import { Button } from '../common/Button';
import { Alert } from '../common/Alert';
import { caseHistoryService, generateCaseId } from '../../services/caseHistory';
import { CaseRecord } from '../../api/types';
import { Play, Sparkles, ShieldCheck, Cpu } from 'lucide-react';
import sampleStressData from '../../data/sample_stress_field.json';

interface InverseWorkspaceProps {
  onOpenReport: (caseRecord: CaseRecord) => void;
}

export const InverseWorkspace: React.FC<InverseWorkspaceProps> = ({ onOpenReport }) => {
  const {
    plateLengthMm,
    plateHeightMm,
    appliedTractionPsi,
    stressGrid,
    stressProvenance,
    stressDescription,
    diagnostics,
    referenceGroundTruth,
    prediction,
    isLoading,
    error,
    isStale,
    handleGeometryChange,
    handleTractionChange,
    handleStressGridChange,
    handleProvenanceChange,
    setReferenceGroundTruth,
    executeInverse,
  } = useInverseAnalysis();

  // Load sample dataset on initial mount so the user immediately has an operational demo ready
  useEffect(() => {
    if (!stressGrid) {
      const grid = sampleStressData.stress_grid_64x64 as number[][];
      handleStressGridChange(grid, {
        isValid: true,
        rows: 64,
        cols: 64,
        min: 26.38,
        max: 275.82,
        mean: 104.5,
        nonFiniteCount: 0,
      });
      handleGeometryChange(sampleStressData.plate_length_mm, sampleStressData.plate_height_mm);
      handleTractionChange(sampleStressData.applied_traction_psi);
      handleProvenanceChange('DEMONSTRATION_SAMPLE', sampleStressData.description);
      setReferenceGroundTruth(sampleStressData.reference_ground_truth);
    }
  }, [
    stressGrid,
    handleStressGridChange,
    handleGeometryChange,
    handleTractionChange,
    handleProvenanceChange,
    setReferenceGroundTruth,
  ]);

  const canExecute = Boolean(stressGrid && diagnostics?.isValid && !isLoading);

  const handleSaveAndReport = async () => {
    if (!prediction || !stressGrid) return;

    const caseId = generateCaseId('INVERSE');
    const newRecord = await caseHistoryService.saveCase({
      case_id: caseId,
      analysis_type: 'INVERSE',
      title: `Inverse Defect Localization (${plateLengthMm}×${plateHeightMm} mm)`,
      created_at: new Date().toISOString(),
      api_version: '1.0.0',
      frontend_version: '1.0.0',
      provenance: 'ML_SURROGATE',
      inverse_request: {
        plate_length_mm: plateLengthMm,
        plate_height_mm: plateHeightMm,
        applied_traction_psi: appliedTractionPsi,
        stress_grid_64x64: stressGrid,
        return_heatmap: true,
        confidence_abstention_threshold_mm: 9.07,
      },
      inverse_response: prediction,
      stress_source_provenance: stressProvenance,
      stress_source_description: stressDescription,
      reference_ground_truth: referenceGroundTruth,
      is_stale: isStale,
    });

    onOpenReport(newRecord);
  };

  return (
    <div className="space-y-4">
      {/* Workspace Status Banner */}
      <div className="bg-[#0b101b] border border-[#1e293b] rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-text-primary tracking-wide">
                Inverse Defect Inspection Workspace
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-bold">
                STAGE 3D MODEL D5 (FROZEN)
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Continuous spatial U-Net surrogate inverts 64×64 Von Mises stress fields into defect heatmaps P(x, y).
            </p>
          </div>
        </div>

        {/* Telemetry Pills */}
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <div className="px-2.5 py-1 rounded-md bg-[#070a12] border border-[#1e293b] flex items-center gap-1.5 text-text-muted">
            <Cpu className="w-3.5 h-3.5 text-accent-cyan" />
            <span>Resolution: <strong className="text-text-primary">64×64 (4,096)</strong></span>
          </div>
          <div className="px-2.5 py-1 rounded-md bg-[#070a12] border border-[#1e293b] flex items-center gap-1.5 text-text-muted">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Threshold: <strong className="text-text-primary">ρ ≤ 9.07 mm</strong></span>
          </div>
        </div>
      </div>

      {/* Main 2-Column Responsive Engineering Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (5 Cols): Inputs & Execution Controls */}
        <div className="lg:col-span-5 space-y-4">
          <StressFieldInput
            plateLengthMm={plateLengthMm}
            plateHeightMm={plateHeightMm}
            appliedTractionPsi={appliedTractionPsi}
            stressGrid={stressGrid}
            stressProvenance={stressProvenance}
            stressDescription={stressDescription}
            diagnostics={diagnostics}
            onGeometryChange={handleGeometryChange}
            onTractionChange={handleTractionChange}
            onStressGridChange={handleStressGridChange}
            onProvenanceChange={handleProvenanceChange}
            onReferenceTruthChange={setReferenceGroundTruth}
            disabled={isLoading}
          />

          {error && (
            <Alert variant="danger" title={error.code || 'Inversion Error'}>
              {error.message}
            </Alert>
          )}

          {/* Primary CTA Button */}
          <Button
            variant="primary"
            size="lg"
            onClick={executeInverse}
            disabled={!canExecute}
            loading={isLoading}
            icon={<Play className="w-4 h-4 fill-current" />}
            className="w-full text-sm font-bold tracking-wider uppercase py-3.5 shadow-lg shadow-accent-cyan/25"
          >
            {isLoading ? 'Inverting Stress Tensor...' : 'Execute Inverse Localization (D5)'}
          </Button>

          {/* Scientific Guardrail Disclosure */}
          <div className="p-3 rounded-lg bg-[#080c14] border border-[#1e293b] text-[11px] text-text-muted leading-relaxed space-y-1">
            <div className="flex items-center gap-1.5 text-text-primary font-semibold text-xs">
              <ShieldCheck className="w-3.5 h-3.5 text-accent-cyan" />
              <span>Scientific Disclosure &amp; Guardrails</span>
            </div>
            <p>
              Model D5 predicts defect coordinates solely from the continuous 64×64 surface stress field without prior knowledge of defect size or location. Ground truth comparison is displayed exclusively for documented demonstration benchmarks.
            </p>
          </div>
        </div>

        {/* Right Column (7 Cols): CAD / FEM Viewport & Localization Analytics */}
        <div className="lg:col-span-7 space-y-4">
          <HeatmapViewer
            plateLengthMm={plateLengthMm}
            plateHeightMm={plateHeightMm}
            heatmapGrid={prediction?.heatmap_grid_64x64}
            stressGrid={stressGrid}
            predictedX={prediction?.predicted_defect_x_mm}
            predictedY={prediction?.predicted_defect_y_mm}
            uncertaintyRadiusMm={prediction?.uncertainty_radius_mm}
            uncertaintyEllipse={prediction?.uncertainty_ellipse}
            referenceGroundTruth={referenceGroundTruth}
            abstentionStatus={prediction?.abstention.status}
            isStale={isStale}
          />

          <InverseResultsPanel
            prediction={prediction}
            stressGrid={stressGrid}
            referenceGroundTruth={referenceGroundTruth}
            onSaveCase={handleSaveAndReport}
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  );
};
