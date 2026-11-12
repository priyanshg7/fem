import React from 'react';
import { Ruler, ShieldAlert, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import { GeometryValidationResponse } from '../../api/types';
import { Badge } from '../common/Badge';

interface ClearancePanelProps {
  validation?: GeometryValidationResponse | null;
  isValidating?: boolean;
}

export const ClearancePanel: React.FC<ClearancePanelProps> = ({
  validation,
  isValidating = false,
}) => {
  if (!validation) {
    return (
      <div style={{ padding: '12px', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
        Awaiting geometry validation...
      </div>
    );
  }

  const { status, clearances, issues, can_proceed_to_inference } = validation;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Header status bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <Ruler size={14} style={{ color: 'var(--color-accent)' }} />
          <span>Physical Clearances & Gating</span>
          {isValidating && <span style={{ fontSize: '0.7rem', color: 'var(--color-accent)' }}>(validating...)</span>}
        </div>

        <div>
          {status === 'VALID' && (
            <Badge variant="success" icon={<CheckCircle size={12} />}>
              Valid Solid Body
            </Badge>
          )}
          {status === 'VALID_WITH_WARNINGS' && (
            <Badge variant="warning" icon={<AlertTriangle size={12} />}>
              Advisory Warning
            </Badge>
          )}
          {status === 'INVALID' && (
            <Badge variant="danger" icon={<AlertCircle size={12} />}>
              Blocked (Collision/Penetration)
            </Badge>
          )}
        </div>
      </div>

      {/* Clearances Grid */}
      {clearances && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '8px',
            background: 'var(--bg-secondary)',
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
              Ligament Clearance (c_lig)
            </div>
            <div
              className="font-mono"
              style={{
                fontSize: '0.95rem',
                fontWeight: 700,
                color:
                  clearances.hole_to_hole_clearance_mm <= 0
                    ? 'var(--color-danger)'
                    : clearances.hole_to_hole_clearance_mm < clearances.min_allowed_margin_mm
                    ? 'var(--color-warning)'
                    : 'var(--color-accent)',
              }}
            >
              {clearances.hole_to_hole_clearance_mm.toFixed(2)} mm
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>
              Margin limit: {clearances.min_allowed_margin_mm.toFixed(1)} mm
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>
              Min Boundary Clearance
            </div>
            <div
              className="font-mono"
              style={{
                fontSize: '0.95rem',
                fontWeight: 700,
                color:
                  clearances.min_boundary_clearance_mm <= 0
                    ? 'var(--color-danger)'
                    : clearances.min_boundary_clearance_mm < clearances.min_allowed_margin_mm
                    ? 'var(--color-warning)'
                    : 'var(--color-success)',
              }}
            >
              {clearances.min_boundary_clearance_mm.toFixed(2)} mm
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>
              Edge: L:{clearances.clearance_to_left_edge_mm.toFixed(0)} R:{clearances.clearance_to_right_edge_mm.toFixed(0)} B:{clearances.clearance_to_bottom_edge_mm.toFixed(0)} T:{clearances.clearance_to_top_edge_mm.toFixed(0)}
            </div>
          </div>
        </div>
      )}

      {/* Validation Issues & Alerts */}
      {issues.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {issues.map((issue, idx) => (
            <div
              key={idx}
              style={{
                fontSize: '0.75rem',
                padding: '6px 10px',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
                background:
                  issue.severity === 'ERROR'
                    ? 'rgba(244, 63, 94, 0.1)'
                    : 'rgba(245, 158, 11, 0.1)',
                borderLeft: `3px solid ${
                  issue.severity === 'ERROR' ? 'var(--color-danger)' : 'var(--color-warning)'
                }`,
                color: issue.severity === 'ERROR' ? '#fecdd3' : '#fde68a',
              }}
            >
              <ShieldAlert size={14} style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <span className="font-mono" style={{ fontWeight: 600 }}>
                  [{issue.code}]:{' '}
                </span>
                <span>{issue.message}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Gating Status summary */}
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
        <span>Backend Gating Check:</span>
        <span style={{ fontWeight: 600, color: can_proceed_to_inference ? 'var(--color-success)' : 'var(--color-danger)' }}>
          {can_proceed_to_inference ? 'Inference Authorized' : 'Prediction Blocked'}
        </span>
      </div>
    </div>
  );
};
