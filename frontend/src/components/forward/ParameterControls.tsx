import React from 'react';
import { Sliders, RotateCcw, Zap } from 'lucide-react';

interface ParameterControlsProps {
  plateLength: number;
  plateHeight: number;
  defectX: number;
  defectY: number;
  defectDiameter: number;
  appliedTraction: number;
  framework: 'heterogeneous' | 'dnn' | 'pca';
  onChangeX: (val: number) => void;
  onChangeY: (val: number) => void;
  onChangeDiameter: (val: number) => void;
  onChangeTraction: (val: number) => void;
  onChangeFramework: (val: 'heterogeneous' | 'dnn' | 'pca') => void;
  onResetDefaults: () => void;
  disabled?: boolean;
}

export const ParameterControls: React.FC<ParameterControlsProps> = ({
  plateLength,
  plateHeight,
  defectX,
  defectY,
  defectDiameter,
  appliedTraction,
  framework,
  onChangeX,
  onChangeY,
  onChangeDiameter,
  onChangeTraction,
  onChangeFramework,
  onResetDefaults,
  disabled = false,
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Parameter Header & Reset */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <Sliders size={14} style={{ color: 'var(--color-accent)' }} />
          <span>Physical Parameter Inputs</span>
        </div>
        <button
          type="button"
          onClick={onResetDefaults}
          disabled={disabled}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-dim)',
            fontSize: '0.7rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
          title="Reset to canonical benchmark position (x=300, y=700, d=20)"
        >
          <RotateCcw size={12} />
          <span>Reset</span>
        </button>
      </div>

      {/* Defect X coordinate */}
      <div className="eng-input-group">
        <div className="eng-label">
          <span>Defect Center X</span>
          <span className="font-mono">{defectX.toFixed(1)} mm</span>
        </div>
        <input
          type="number"
          className="eng-input"
          min={5}
          max={plateLength - 5}
          step={5}
          value={defectX}
          disabled={disabled}
          onChange={(e) => onChangeX(parseFloat(e.target.value) || 0)}
        />
        <input
          type="range"
          className="eng-slider"
          min={10}
          max={plateLength - 10}
          step={1}
          value={defectX}
          disabled={disabled}
          onChange={(e) => onChangeX(parseFloat(e.target.value))}
        />
      </div>

      {/* Defect Y coordinate */}
      <div className="eng-input-group">
        <div className="eng-label">
          <span>Defect Center Y</span>
          <span className="font-mono">{defectY.toFixed(1)} mm</span>
        </div>
        <input
          type="number"
          className="eng-input"
          min={5}
          max={plateHeight - 5}
          step={5}
          value={defectY}
          disabled={disabled}
          onChange={(e) => onChangeY(parseFloat(e.target.value) || 0)}
        />
        <input
          type="range"
          className="eng-slider"
          min={10}
          max={plateHeight - 10}
          step={1}
          value={defectY}
          disabled={disabled}
          onChange={(e) => onChangeY(parseFloat(e.target.value))}
        />
      </div>

      {/* Defect Diameter */}
      <div className="eng-input-group">
        <div className="eng-label">
          <span>Defect Diameter (d)</span>
          <span className="font-mono">{defectDiameter.toFixed(1)} mm</span>
        </div>
        <input
          type="number"
          className="eng-input"
          min={5}
          max={50}
          step={1}
          value={defectDiameter}
          disabled={disabled}
          onChange={(e) => onChangeDiameter(parseFloat(e.target.value) || 0)}
        />
        <input
          type="range"
          className="eng-slider"
          min={5}
          max={50}
          step={1}
          value={defectDiameter}
          disabled={disabled}
          onChange={(e) => onChangeDiameter(parseFloat(e.target.value))}
        />
      </div>

      {/* Applied Traction Load */}
      <div className="eng-input-group">
        <div className="eng-label">
          <span>Applied Traction (σ₀)</span>
          <span className="font-mono">{appliedTraction.toFixed(1)} psi</span>
        </div>
        <input
          type="number"
          className="eng-input"
          min={10}
          max={500}
          step={10}
          value={appliedTraction}
          disabled={disabled}
          onChange={(e) => onChangeTraction(parseFloat(e.target.value) || 100)}
        />
        <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '2px' }}>
          Standard training baseline: 100.0 psi (linear superposition applied for other values)
        </div>
      </div>

      {/* Stacking Framework Selector */}
      <div className="eng-input-group">
        <div className="eng-label">
          <span>Stacking Meta-Framework</span>
          <span style={{ color: 'var(--color-accent)' }}>Stage 3B</span>
        </div>
        <select
          className="eng-input"
          value={framework}
          disabled={disabled}
          onChange={(e) => onChangeFramework(e.target.value as any)}
          style={{ cursor: 'pointer' }}
        >
          <option value="heterogeneous">Heterogeneous Ensemble (Recommended — Confirmatory R²=0.9634)</option>
          <option value="dnn">DNN Ridge Stacking (4 Neural Nets)</option>
          <option value="pca">PCA Component Stacking (Dimension Reduced)</option>
        </select>
      </div>

      {/* Quick Verification Presets */}
      <div style={{ marginTop: '4px' }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Zap size={11} style={{ color: 'var(--color-accent)' }} />
          <span>Quick Verification Presets:</span>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="eng-btn eng-btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.7rem' }}
            onClick={() => { onChangeX(300); onChangeY(700); onChangeDiameter(20); }}
          >
            Canonical (300, 700)
          </button>
          <button
            type="button"
            className="eng-btn eng-btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.7rem' }}
            onClick={() => { onChangeX(445); onChangeY(plateHeight / 2.0); onChangeDiameter(10); }}
          >
            Near-Hole Warning
          </button>
          <button
            type="button"
            className="eng-btn eng-btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.7rem' }}
            onClick={() => { onChangeX(15); onChangeY(700); onChangeDiameter(10); }}
          >
            Near-Edge Warning
          </button>
          <button
            type="button"
            className="eng-btn eng-btn-secondary"
            style={{ padding: '4px 8px', fontSize: '0.7rem' }}
            onClick={() => { onChangeX(plateLength / 2.0); onChangeY(plateHeight / 2.0); onChangeDiameter(20); }}
          >
            Collision Test
          </button>
        </div>
      </div>
    </div>
  );
};

