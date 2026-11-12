import React from 'react';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  loading = false,
  disabled = false,
  icon,
  className = '',
  ...props
}) => {
  const isBusy = isLoading || loading;
  const sizeClasses =
    size === 'sm' ? 'px-2.5 py-1 text-xs' : size === 'lg' ? 'px-5 py-3 text-base' : 'px-4 py-2 text-sm';
  const variantClass = variant === 'ghost' ? 'eng-btn-secondary opacity-80' : `eng-btn-${variant}`;

  return (
    <button
      className={`eng-btn ${variantClass} ${sizeClasses} ${className}`}
      disabled={disabled || isBusy}
      {...props}
    >
      {isBusy ? (
        <>
          <Loader2 size={16} className="animate-spin" />
          <span>Processing...</span>
        </>
      ) : (
        <>
          {icon && <span>{icon}</span>}
          <span>{children}</span>
        </>
      )}
    </button>
  );
};
