import React, { useState } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Alert } from '../common/Alert';
import { StressSourceProvenance, ReferenceGroundTruth } from '../../api/types';
import { validateStressGrid, parseCsvToMatrix, parseJsonToMatrix, MatrixDiagnostics } from '../../utils/matrixValidator';
import sampleStressData from '../../data/sample_stress_field.json';
import {
  FileUp,
  Database,
  CheckCircle2,
  Sparkles,
  Sliders,
  Layers,
  Radio,
  FileText,
} from 'lucide-react';

interface StressFieldInputProps {
  plateLengthMm: number;
  plateHeightMm: number;
  appliedTractionPsi: number;
  stressGrid?: number[][] | null;
  stressProvenance: StressSourceProvenance;
  stressDescription: string;
  diagnostics: MatrixDiagnostics | null;
  onGeometryChange: (length: number, height: number) => void;
  onTractionChange: (traction: number) => void;
  onStressGridChange: (grid: number[][], diagnostics: MatrixDiagnostics) => void;
  onProvenanceChange: (prov: StressSourceProvenance, desc: string) => void;
  onReferenceTruthChange: (truth?: ReferenceGroundTruth) => void;
  disabled?: boolean;
}

const CANONICAL_PRESETS = [
  { label: 'Square 1000×1000', L: 1000, H: 1000, alpha: '1.00', desc: 'Symmetric reference' },
  { label: 'Rect 1000×900', L: 1000, H: 900, alpha: '1.11', desc: 'Mild elongation' },
  { label: 'Rect 1000×800', L: 1000, H: 800, alpha: '1.25', desc: 'Moderate aspect' },
  { label: 'Rect 1000×700', L: 1000, H: 700, alpha: '1.43', desc: 'Audited benchmark', isDemo: true },
  { label: 'Rect 1000×600', L: 1000, H: 600, alpha: '1.67', desc: 'High aspect ratio' },
];

export const StressFieldInput: React.FC<StressFieldInputProps> = ({
  plateLengthMm,
  plateHeightMm,
  appliedTractionPsi,
  stressProvenance,
  stressDescription,
  diagnostics,
  onGeometryChange,
  onTractionChange,
  onStressGridChange,
  onProvenanceChange,
  onReferenceTruthChange,
  disabled,
}) => {
  const [activeTab, setActiveTab] = useState<'demo' | 'upload' | 'paste' | 'sensors'>('demo');
  const [pasteText, setPasteText] = useState('');
  const [parseError, setParseError] = useState<string | null>(null);

  // Load Audited Demonstration Sample (SPEC-I02)
  const handleLoadSample = () => {
    try {
      const grid = sampleStressData.stress_grid_64x64 as number[][];
      const diag = validateStressGrid(grid);
      onGeometryChange(sampleStressData.plate_length_mm, sampleStressData.plate_height_mm);
      onTractionChange(sampleStressData.applied_traction_psi);
      onStressGridChange(grid, diag);
      onProvenanceChange('DEMONSTRATION_SAMPLE', sampleStressData.description);
      onReferenceTruthChange(sampleStressData.reference_ground_truth);
      setParseError(null);
      setPasteText('');
    } catch (err: any) {
      setParseError(`Failed to load sample dataset: ${err.message}`);
    }
  };

  // File Upload Handler (JSON or CSV)
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      try {
        let grid: number[][];
        if (file.name.endsWith('.json')) {
          grid = parseJsonToMatrix(content);
        } else if (file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
          grid = parseCsvToMatrix(content);
        } else {
          throw new Error('Unsupported format. Please upload a .json or .csv matrix file.');
        }

        const diag = validateStressGrid(grid);
        if (!diag.isValid) {
          setParseError(diag.errorMessage || 'Invalid matrix dimensions or format.');
          return;
        }

        onStressGridChange(grid, diag);
        onProvenanceChange(
          'EXTERNAL_FEM',
          `Imported file: ${file.name} (${diag.rows}×${diag.cols})`
        );
        onReferenceTruthChange(undefined);
        setParseError(null);
      } catch (err: any) {
        setParseError(`Parse error: ${err.message}`);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Parse Textarea text
  const handleParseText = () => {
    if (!pasteText.trim()) return;
    try {
      let grid: number[][];
      const trimmed = pasteText.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        grid = parseJsonToMatrix(trimmed);
      } else {
        grid = parseCsvToMatrix(trimmed);
      }

      const diag = validateStressGrid(grid);
      if (!diag.isValid) {
        setParseError(diag.errorMessage || 'Invalid matrix dimensions or format.');
        return;
      }

      onStressGridChange(grid, diag);
      onProvenanceChange('EXTERNAL_FEM', 'Direct clipboard / textarea input');
      onReferenceTruthChange(undefined);
      setParseError(null);
    } catch (err: any) {
      setParseError(`Manual parse error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-4">
      {/* 1. Specimen Geometry Selector */}
      <Card
        title="Plate Specimen Geometry"
        icon={<Sliders className="w-4 h-4 text-accent-cyan" />}
        headerAction={
          <span className="text-[11px] font-mono text-accent-cyan bg-accent-cyan/10 px-2 py-0.5 rounded border border-accent-cyan/20">
            Canonical 1000 mm Series
          </span>
        }
      >
        <div className="space-y-3.5">
          {/* Preset Buttons Grid */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">
                Select Canonical Geometry
              </span>
              <span className="text-[11px] font-mono text-text-muted">
                Aspect ratio α = L/H
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {CANONICAL_PRESETS.map((p) => {
                const isSelected = plateHeightMm === p.H;
                return (
                  <button
                    key={p.label}
                    type="button"
                    disabled={disabled}
                    onClick={() => onGeometryChange(p.L, p.H)}
                    className={`p-2.5 rounded-lg text-left transition-all border relative flex items-center justify-between ${
                      isSelected
                        ? 'bg-accent-cyan/15 border-accent-cyan text-text-primary shadow-sm shadow-accent-cyan/20'
                        : 'bg-[#0d131f] border-[#1e293b] text-text-muted hover:border-[#33466d] hover:text-text-primary'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-mono font-semibold text-text-primary">
                          {p.label}
                        </span>
                        {p.isDemo && (
                          <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
                            DEMO
                          </span>
                        )}
                      </div>
                      <span className="text-[10px] text-text-muted block mt-0.5">
                        {p.desc}
                      </span>
                    </div>

                    <div className="text-right font-mono">
                      <span className={`text-xs font-bold ${isSelected ? 'text-accent-cyan' : 'text-text-muted'}`}>
                        α = {p.alpha}
                      </span>
                      <span className="text-[10px] text-text-dim block">
                        Dc = {(p.H / 12).toFixed(1)} mm
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Geometry Details Readout */}
          <div className="grid grid-cols-3 gap-2 p-2.5 rounded-lg bg-[#080c14] border border-[#1e293b] text-center font-mono">
            <div>
              <span className="text-[10px] text-text-muted block uppercase">Length (L)</span>
              <span className="text-sm font-bold text-text-primary">{plateLengthMm} mm</span>
            </div>
            <div>
              <span className="text-[10px] text-text-muted block uppercase">Height (H)</span>
              <span className="text-sm font-bold text-accent-cyan">{plateHeightMm} mm</span>
            </div>
            <div>
              <span className="text-[10px] text-text-muted block uppercase">Central Hole</span>
              <span className="text-sm font-bold text-text-secondary">{(plateHeightMm / 12).toFixed(1)} mm</span>
            </div>
          </div>

          {/* Applied Traction Setting */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-text-muted uppercase tracking-wider">
                Far-Field Traction (σ₀)
              </label>
              <span className="text-xs font-mono font-bold text-accent-cyan">
                {appliedTractionPsi} psi
              </span>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="50"
                max="150"
                step="5"
                value={appliedTractionPsi}
                disabled={disabled}
                onChange={(e) => onTractionChange(Number(e.target.value))}
                className="w-full accent-accent-cyan cursor-pointer h-1.5 bg-[#192237] rounded-lg"
              />
              <span className="text-[11px] font-mono text-text-muted whitespace-nowrap bg-[#0d131f] px-2 py-1 rounded border border-[#1e293b]">
                {appliedTractionPsi} psi
              </span>
            </div>
            <span className="text-[10px] text-text-muted mt-1 block">
              Reference load for stress field normalization (Stage 4D standard: 100.0 psi).
            </span>
          </div>
        </div>
      </Card>

      {/* 2. Stress Field Ingestion */}
      <Card
        title="Spatial Stress Field Input (64×64)"
        icon={<Layers className="w-4 h-4 text-accent-cyan" />}
        headerAction={
          <span className="text-[11px] font-mono text-text-muted bg-[#080c14] px-2 py-0.5 rounded border border-[#1e293b]">
            4,096 Sensor Points
          </span>
        }
      >
        <div className="space-y-3.5">
          {/* Ingestion Source Tabs */}
          <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-[#080c14] border border-[#1e293b]">
            <button
              type="button"
              onClick={() => setActiveTab('demo')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition-all text-center flex items-center justify-center gap-1 ${
                activeTab === 'demo'
                  ? 'bg-accent-cyan text-[#080c14] font-bold shadow-sm'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              <Sparkles className="w-3 h-3" />
              <span>Demo</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('upload')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition-all text-center flex items-center justify-center gap-1 ${
                activeTab === 'upload'
                  ? 'bg-accent-cyan text-[#080c14] font-bold shadow-sm'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              <FileUp className="w-3 h-3" />
              <span>Upload</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('paste')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition-all text-center flex items-center justify-center gap-1 ${
                activeTab === 'paste'
                  ? 'bg-accent-cyan text-[#080c14] font-bold shadow-sm'
                  : 'text-text-muted hover:text-text-primary'
              }`}
            >
              <FileText className="w-3 h-3" />
              <span>Paste</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('sensors')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition-all text-center flex items-center justify-center gap-1 ${
                activeTab === 'sensors'
                  ? 'bg-amber-500 text-[#080c14] font-bold shadow-sm'
                  : 'text-amber-400/70 hover:text-amber-300'
              }`}
            >
              <Radio className="w-3 h-3" />
              <span>Sensors</span>
            </button>
          </div>

          {/* TAB 1: DEMO SPECIMEN */}
          {activeTab === 'demo' && (
            <div className="p-3.5 rounded-lg bg-[#0a1120] border border-accent-cyan/30 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-bold text-accent-cyan">
                      Audited Specimen SPEC-I02
                    </span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-mono">
                      CONFIRMATORY
                    </span>
                  </div>
                  <p className="text-[11px] text-text-muted mt-1 leading-relaxed">
                    Plate 1000×700 mm with defect (d=28.1 mm) at (507.8, 265.4) mm. Ground truth is embedded for rigorous verification.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono p-2 rounded bg-[#070b14] border border-[#1e293b]">
                <div>
                  <span className="text-[10px] text-text-muted block">True Position</span>
                  <span className="text-emerald-400 font-bold">(507.8, 265.4) mm</span>
                </div>
                <div>
                  <span className="text-[10px] text-text-muted block">Defect Diameter</span>
                  <span className="text-text-primary font-bold">d = 28.1 mm</span>
                </div>
              </div>

              <Button
                variant="primary"
                size="sm"
                onClick={handleLoadSample}
                disabled={disabled}
                icon={<Database className="w-3.5 h-3.5" />}
                className="w-full font-semibold"
              >
                Reload Audited Benchmark Specimen
              </Button>
            </div>
          )}

          {/* TAB 2: FILE UPLOAD */}
          {activeTab === 'upload' && (
            <div className="space-y-2">
              <label className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-[#222f4c] hover:border-accent-cyan rounded-lg bg-[#080c14] hover:bg-[#0c1424] cursor-pointer transition-all group">
                <FileUp className="w-8 h-8 text-accent-cyan mb-2 group-hover:scale-110 transition-transform" />
                <span className="text-xs font-semibold text-text-primary">
                  Click to browse or drop matrix file
                </span>
                <span className="text-[11px] font-mono text-text-muted mt-1">
                  Accepts .json (array 64×64) or .csv (64 rows × 64 cols)
                </span>
                <input
                  type="file"
                  accept=".json,.csv,.txt"
                  disabled={disabled}
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          )}

          {/* TAB 3: PASTE TEXT */}
          {activeTab === 'paste' && (
            <div className="space-y-2">
              <textarea
                rows={4}
                value={pasteText}
                disabled={disabled}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder="Paste raw JSON matrix [[v0, v1, ...], ...] or comma-delimited 64x64 lines..."
                className="w-full bg-[#080c14] border border-[#1e293b] rounded-lg p-2.5 text-xs font-mono text-text-primary placeholder:text-text-dim focus:border-accent-cyan focus:outline-none resize-none"
              />
              <Button
                variant="secondary"
                size="sm"
                onClick={handleParseText}
                disabled={disabled || !pasteText.trim()}
                className="w-full font-semibold"
              >
                Validate &amp; Ingest Pasted Field
              </Button>
            </div>
          )}

          {/* TAB 4: SENSORS (EXP.) */}
          {activeTab === 'sensors' && (
            <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 space-y-2 text-xs">
              <div className="flex items-center gap-1.5 text-amber-400 font-bold">
                <Radio className="w-3.5 h-3.5" />
                <span>Discrete Strain Sensor Inversion (Stage 3D-B2)</span>
              </div>
              <p className="text-[11px] text-text-muted leading-relaxed">
                Stage 3D proved that sparse sensor arrays (8–32 sensors) achieve mean error of ~42 mm after RBF interpolation into the 64×64 grid.
              </p>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleLoadSample}
                disabled={disabled}
                className="w-full text-amber-400 border-amber-500/40 hover:bg-amber-500/20"
              >
                Simulate 32-Sensor Synthetic Reconstitution
              </Button>
            </div>
          )}

          {/* Parse Error Alert */}
          {parseError && (
            <Alert variant="danger" title="Matrix Parsing Failed">
              {parseError}
            </Alert>
          )}

          {/* Matrix Diagnostics HUD */}
          {diagnostics && diagnostics.isValid ? (
            <div className="p-3 rounded-lg bg-[#080c14] border border-[#1e293b] space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold font-mono">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Valid 64×64 Spatial Tensor</span>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20">
                  {stressProvenance}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-center font-mono">
                <div className="p-1.5 rounded bg-[#0d131f] border border-[#192237]">
                  <span className="text-[9px] text-text-muted block">Min Stress</span>
                  <span className="text-xs font-bold text-text-primary">{diagnostics.min.toFixed(1)} psi</span>
                </div>
                <div className="p-1.5 rounded bg-[#0d131f] border border-[#192237]">
                  <span className="text-[9px] text-text-muted block">Mean Stress</span>
                  <span className="text-xs font-bold text-accent-cyan">{diagnostics.mean.toFixed(1)} psi</span>
                </div>
                <div className="p-1.5 rounded bg-[#0d131f] border border-[#192237]">
                  <span className="text-[9px] text-text-muted block">Max Stress</span>
                  <span className="text-xs font-bold text-rose-400">{diagnostics.max.toFixed(1)} psi</span>
                </div>
              </div>

              {stressDescription && (
                <div className="text-[10px] text-text-dim truncate font-mono">
                  Source: {stressDescription}
                </div>
              )}
            </div>
          ) : diagnostics && !diagnostics.isValid ? (
            <Alert variant="danger" title="Matrix Validation Error">
              {diagnostics.errorMessage || 'Invalid 64×64 field matrix.'}
            </Alert>
          ) : null}
        </div>
      </Card>
    </div>
  );
};
