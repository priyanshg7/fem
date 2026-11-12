import React from 'react';
import { Activity, ShieldCheck, AlertCircle, Cpu } from 'lucide-react';
import { Badge } from '../common/Badge';

interface HeaderProps {
  backendConnected: boolean;
  backendReady: boolean;
  activeModelId?: string;
  activeModelVersion?: string;
}

export const Header: React.FC<HeaderProps> = ({
  backendConnected,
  backendReady,
  activeModelId = 'forward_ensemble_stage3b',
  activeModelVersion = '3.2.0-frozen',
}) => {
  return (
    <header
      style={{
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #0284c7, #38bdf8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#030712',
              fontWeight: 800,
              fontSize: '1.1rem',
              letterSpacing: '-0.05em',
            }}
          >
            SΣ
          </div>
          <div>
            <div style={{ fontSize: '1rem', fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>SimuStruct Digital Twin</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 500 }}>v1.0.0</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Static Analysis of a Plate with Defect Using Machine Learning
            </div>
          </div>
        </div>

        <Badge variant="surrogate" icon={<Cpu size={12} />}>
          Surrogate Active: {activeModelId} ({activeModelVersion})
        </Badge>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
          {backendConnected ? (
            backendReady ? (
              <Badge variant="success" icon={<ShieldCheck size={12} />}>
                API Ready (FastAPI :8000)
              </Badge>
            ) : (
              <Badge variant="warning" icon={<Activity size={12} />}>
                Backend Connected (Models Loading)
              </Badge>
            )
          ) : (
            <Badge variant="danger" icon={<AlertCircle size={12} />}>
              Backend Disconnected
            </Badge>
          )}
        </div>
      </div>
    </header>
  );
};

