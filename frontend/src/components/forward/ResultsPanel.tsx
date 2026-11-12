import React, { useState } from 'react';
import { Award, Cpu, Clock, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { ForwardPredictionResponse } from '../../api/types';
import { Badge } from '../common/Badge';

interface ResultsPanelProps {
  prediction: ForwardPredictionResponse | null;
  isLoading: boolean;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  prediction,
  isLoading,
}) => {
  const [showBaseModels, setShowBaseModels] = useState(false);

  if (isLoading) {
    return (
      <div
        style={{
          padding: '40px 20px',
          textAlign: 'center',
          background: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div
          style={{
            width: '36px',
            height: '36px',
            border: '3px solid rgba(56, 189, 248, 0.2)',
            borderTopColor: 'var(--color-accent)',
            borderRadius: '50%',
            margin: '0 auto 14px auto',
            animation: 'spin 1s linear infinite',
          }}
        />
        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-main)' }}>
          Executing Stage 3B Forward Ensemble...
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          Querying 7 heterogeneous base estimators + Ridge meta-learner
        </div>
        <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div
        style={{
          padding: '30px 20px',
          textAlign: 'center',
          background: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px dashed var(--border-subtle)',
          color: 'var(--text-muted)',
          fontSize: '0.85rem',
        }}
      >
        <Cpu size={24} style={{ color: 'var(--text-dim)', marginBottom: '8px' }} />
        <div>No prediction generated yet.</div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>
          Configure plate parameters and click &quot;Run Forward Surrogate Analysis&quot;.
        </div>
      </div>
    );
  }

  const {
    predicted_max_stress_psi,
    predicted_Kt_element,
    ensemble_uncertainty_psi,
    applied_traction_psi,
    base_model_predictions_Kt,
    model_id,
    model_version,
    execution_time_ms,
    prediction_id,
    warnings,
    input_summary,
  } = prediction;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Header bar with badges */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Award size={16} style={{ color: 'var(--color-accent)' }} />
          <span style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Surrogate Output
          </span>
        </div>

        <div style={{ display: 'flex', gap: '6px' }}>
          <Badge variant="surrogate">
            ML_SURROGATE
          </Badge>
          <Badge variant="neutral">
            {model_id} ({model_version})
          </Badge>
        </div>
      </div>

      {/* Main KPI metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
        {/* Maximum Von Mises Stress Card */}
        <div
          style={{
            background: 'linear-gradient(180deg, rgba(56, 189, 248, 0.08) 0%, rgba(10, 13, 20, 0.6) 100%)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '8px',
            padding: '14px',
          }}
        >
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Predicted Max Von Mises Stress
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', margin: '4px 0' }}>
            <span className="font-mono" style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--color-accent)' }}>
              {predicted_max_stress_psi.toFixed(2)}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              psi
            </span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
            Uncertainty: ±{ensemble_uncertainty_psi.toFixed(2)} psi (dispersion)
          </div>
        </div>

        {/* Stress Concentration Factor Card */}
        <div
          style={{
            background: 'linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(10, 13, 20, 0.6) 100%)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: '8px',
            padding: '14px',
          }}
        >
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Stress Concentration Factor (Kt)
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', margin: '4px 0' }}>
            <span className="font-mono" style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--color-success)' }}>
              {predicted_Kt_element.toFixed(4)}
            </span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
            σ_nom = {applied_traction_psi.toFixed(1)} psi (Tension)
          </div>
        </div>
      </div>

      {/* Warnings & Advisories if returned */}
      {warnings && warnings.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {warnings.map((warn, i) => (
            <div
              key={i}
              style={{
                background: 'rgba(245, 158, 11, 0.1)',
                borderLeft: '3px solid var(--color-warning)',
                padding: '8px 12px',
                borderRadius: '4px',
                fontSize: '0.75rem',
                color: '#fde68a',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <AlertTriangle size={14} style={{ flexShrink: 0 }} />
              <span>{warn}</span>
            </div>
          ))}
        </div>
      )}

      {/* Primary Technical Disclaimers */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          lineHeight: 1.4,
        }}
      >
        <strong style={{ color: 'var(--color-accent)' }}>Surrogate Scope Notice:</strong> This prediction is generated rapidly by an ML surrogate ensemble trained on 51,625 linear-elastic FEM simulations. It predicts peak element-centroid stress and does not replace numerical mesh convergence or full 2D stress tensor post-processing.
      </div>

      {/* Technical Breakdown Accordion (Base Estimator Predictions) */}
      <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '10px' }}>
        <button
          type="button"
          onClick={() => setShowBaseModels(!showBaseModels)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            padding: '4px 0',
          }}
        >
          <span>Base Model Ensemble Predictions ({Object.keys(base_model_predictions_Kt || {}).length} estimators)</span>
          {showBaseModels ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {showBaseModels && base_model_predictions_Kt && (
          <div
            style={{
              marginTop: '8px',
              background: 'var(--bg-secondary)',
              borderRadius: '6px',
              padding: '8px 12px',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <table style={{ width: '100%', fontSize: '0.7rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-dim)', textAlign: 'left', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '4px 0' }}>Estimator</th>
                  <th style={{ padding: '4px 0', textAlign: 'right' }}>Predicted Kt</th>
                  <th style={{ padding: '4px 0', textAlign: 'right' }}>Equiv Stress</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {Object.entries(base_model_predictions_Kt).map(([name, val]) => (
                  <tr key={name} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '4px 0', color: 'var(--text-main)' }}>{name}</td>
                    <td style={{ padding: '4px 0', textAlign: 'right', color: 'var(--color-accent)' }}>{val.toFixed(4)}</td>
                    <td style={{ padding: '4px 0', textAlign: 'right', color: 'var(--text-muted)' }}>{(val * applied_traction_psi).toFixed(1)} psi</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Input Summary and Traceability Footnote */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.65rem',
          color: 'var(--text-dim)',
          fontFamily: 'var(--font-mono)',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Clock size={11} />
            <span>Latency: {execution_time_ms.toFixed(1)} ms</span>
          </div>
          <span>•</span>
          <span>AR: {input_summary?.aspect_ratio ?? 1.0}</span>
        </div>
        <div>
          ID: {prediction_id}
        </div>
      </div>
    </div>
  );
};
