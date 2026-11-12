import React from 'react';

export type BadgeVariant =
  | 'surrogate'
  | 'success'
  | 'warning'
  | 'danger'
  | 'neutral'
  | 'cyan'
  | 'green'
  | 'amber'
  | 'red';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
  icon?: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  className = '',
  icon,
}) => {
  const normalizedVariant =
    variant === 'cyan'
      ? 'surrogate'
      : variant === 'green'
      ? 'success'
      : variant === 'amber'
      ? 'warning'
      : variant === 'red'
      ? 'danger'
      : variant;

  return (
    <span className={`badge badge-${normalizedVariant} ${className}`}>
      {icon && <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>}
      {children}
    </span>
  );
};
