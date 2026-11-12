import React from 'react';
import { RefreshCw } from 'lucide-react';

interface StaleNoticeProps {
  predictionId: string;
  onRerun: () => void;
  canRerun: boolean;
  isLoading: boolean;
}

export const StaleNotice: React.FC<StaleNoticeProps> = ({
  predictionId,
  onRerun,
  canRerun,
  isLoading,
}) => {
  return (
    <div
      style={{
        background: 'rgba(245, 158, 11, 0.12)',
        border: '1px solid rgba(245, 158, 11, 0.4)',
        borderRadius: '6px',
        padding: '10px 14px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '0.75rem',
        color: '#fef3c7',
        marginBottom: '12px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <RefreshCw size={16} style={{ color: 'var(--color-warning)', flexShrink: 0 }} />
        <div>
          <strong>Parameters Changed:</strong> Displayed prediction (<code className="font-mono">{predictionId}</code>) is stale and corresponds to earlier inputs.
        </div>
      </div>

      <button
        type="button"
        disabled={!canRerun || isLoading}
        onClick={onRerun}
        className="eng-btn eng-btn-primary"
        style={{
          padding: '4px 10px',
          fontSize: '0.75rem',
          height: 'auto',
          whiteSpace: 'nowrap',
        }}
      >
        Re-run Analysis
      </button>
    </div>
  );
};

