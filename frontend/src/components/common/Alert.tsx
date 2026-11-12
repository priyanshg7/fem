import React from 'react';
import { AlertTriangle, AlertCircle, Info, CheckCircle2 } from 'lucide-react';

export interface AlertProps {
  type?: 'warning' | 'danger' | 'info' | 'success';
  variant?: 'warning' | 'danger' | 'info' | 'success';
  code?: string;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export const Alert: React.FC<AlertProps> = ({
  type,
  variant,
  code,
  title,
  children,
  className = '',
}) => {
  const effectiveType = variant || type || 'warning';
  const displayTitle = title || (code ? `[${code}]` : undefined);

  const getIcon = () => {
    switch (effectiveType) {
      case 'danger':
        return <AlertCircle size={18} style={{ color: 'var(--color-danger)', flexShrink: 0 }} />;
      case 'warning':
        return <AlertTriangle size={18} style={{ color: 'var(--color-warning)', flexShrink: 0 }} />;
      case 'success':
        return <CheckCircle2 size={18} style={{ color: 'var(--color-success)', flexShrink: 0 }} />;
      default:
        return <Info size={18} style={{ color: 'var(--color-accent)', flexShrink: 0 }} />;
    }
  };

  return (
    <div className={`eng-alert eng-alert-${effectiveType} ${className}`}>
      {getIcon()}
      <div style={{ flex: 1 }}>
        {displayTitle && (
          <div
            style={{
              fontWeight: 600,
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              marginBottom: 2,
            }}
          >
            {displayTitle}
          </div>
        )}
        <div>{children}</div>
      </div>
    </div>
  );
};
