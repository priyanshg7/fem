import React from 'react';

interface CardProps {
  title?: string;
  icon?: React.ReactNode;
  headerAction?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  icon,
  headerAction,
  children,
  className = '',
}) => {
  return (
    <div className={`eng-card ${className}`}>
      {title && (
        <div className="eng-card-header">
          <div className="eng-card-title">
            {icon && <span>{icon}</span>}
            <span>{title}</span>
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

