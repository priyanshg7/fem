import React from 'react';
import { Database, Shield } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer
      style={{
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border-subtle)',
        padding: '12px 24px',
        marginTop: 'auto',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Shield size={14} style={{ color: 'var(--color-accent)' }} />
          <span>Linear Elastic Mechanics (E = 200 GPa, ν = 0.30, t = 1.0 mm)</span>
        </div>
        <span>•</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Database size={14} style={{ color: 'var(--color-success)' }} />
          <span>FEM Training Corpus: 51,625 simulations across 5 aspect ratios</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontFamily: 'var(--font-mono)' }}>
        <span>API Contract: <strong style={{ color: 'var(--text-main)' }}>v1.0.0-frozen</strong></span>
        <span>•</span>
        <span style={{ color: 'var(--color-accent)' }}>SimuStruct Digital Twin Platform</span>
      </div>
    </footer>
  );
};

