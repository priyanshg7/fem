import React from 'react';
import { Layers, Crosshair, History } from 'lucide-react';

export type WorkspaceType = 'forward' | 'inverse' | 'history';

interface NavigationProps {
  currentWorkspace: WorkspaceType;
  onSelectWorkspace: (workspace: WorkspaceType) => void;
  caseCount?: number;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentWorkspace,
  onSelectWorkspace,
  caseCount = 0,
}) => {
  return (
    <nav
      style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}
    >
      {/* 1. Forward Stress Analysis */}
      <button
        type="button"
        onClick={() => onSelectWorkspace('forward')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px 18px',
          background: 'none',
          border: 'none',
          borderBottom:
            currentWorkspace === 'forward' ? '2px solid var(--color-accent)' : '2px solid transparent',
          color: currentWorkspace === 'forward' ? 'var(--color-accent)' : 'var(--text-muted)',
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.15s ease',
        }}
      >
        <Layers size={16} />
        <span>Forward Stress Analysis (Stage 3B)</span>
      </button>

      {/* 2. Inverse Defect Inspection (Stage 5C Active) */}
      <button
        type="button"
        onClick={() => onSelectWorkspace('inverse')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px 18px',
          background: 'none',
          border: 'none',
          borderBottom:
            currentWorkspace === 'inverse' ? '2px solid var(--color-accent)' : '2px solid transparent',
          color: currentWorkspace === 'inverse' ? 'var(--color-accent)' : 'var(--text-muted)',
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.15s ease',
        }}
      >
        <Crosshair size={16} />
        <span>Inverse Defect Inspection (Model D5)</span>
        <span
          style={{
            fontSize: '0.65rem',
            padding: '2px 6px',
            borderRadius: '4px',
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#10b981',
            fontWeight: 700,
          }}
        >
          ACTIVE
        </span>
      </button>

      {/* 3. Case History & Reports (Stage 5D Active) */}
      <button
        type="button"
        onClick={() => onSelectWorkspace('history')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px 18px',
          background: 'none',
          border: 'none',
          borderBottom:
            currentWorkspace === 'history' ? '2px solid var(--color-accent)' : '2px solid transparent',
          color: currentWorkspace === 'history' ? 'var(--color-accent)' : 'var(--text-muted)',
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.15s ease',
        }}
      >
        <History size={16} />
        <span>Case History &amp; Reports</span>
        {caseCount > 0 && (
          <span
            style={{
              fontSize: '0.65rem',
              padding: '1px 6px',
              borderRadius: '999px',
              background: 'rgba(0, 180, 216, 0.2)',
              color: 'var(--color-accent)',
              fontWeight: 700,
            }}
          >
            {caseCount}
          </span>
        )}
      </button>
    </nav>
  );
};
