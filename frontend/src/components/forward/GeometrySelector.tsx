import React from 'react';

export interface CanonicalGeometry {
  id: string;
  name: string;
  length_mm: number;
  height_mm: number;
  aspect_ratio: number;
  canonical_dc_mm: number;
}

export const CANONICAL_GEOMETRIES: CanonicalGeometry[] = [
  { id: '1000x1000', name: 'Square Plate', length_mm: 1000.0, height_mm: 1000.0, aspect_ratio: 1.000, canonical_dc_mm: 83.333 },
  { id: '1000x900', name: 'Rectangular (AR 1.11)', length_mm: 1000.0, height_mm: 900.0, aspect_ratio: 1.111, canonical_dc_mm: 75.000 },
  { id: '1000x800', name: 'Rectangular (AR 1.25)', length_mm: 1000.0, height_mm: 800.0, aspect_ratio: 1.250, canonical_dc_mm: 66.667 },
  { id: '1000x700', name: 'Rectangular (AR 1.43)', length_mm: 1000.0, height_mm: 700.0, aspect_ratio: 1.429, canonical_dc_mm: 58.333 },
  { id: '1000x600', name: 'Rectangular (AR 1.67)', length_mm: 1000.0, height_mm: 600.0, aspect_ratio: 1.667, canonical_dc_mm: 50.000 },
];

interface GeometrySelectorProps {
  selectedHeight: number;
  onSelectGeometry: (geom: CanonicalGeometry) => void;
  disabled?: boolean;
}

export const GeometrySelector: React.FC<GeometrySelectorProps> = ({
  selectedHeight,
  onSelectGeometry,
  disabled = false,
}) => {
  return (
    <div className="eng-input-group">
      <div className="eng-label">
        <span>Plate Geometry Family (Canonical 5-Family Corpus)</span>
        <span style={{ color: 'var(--color-accent)' }}>L = 1000 mm</span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '8px',
          marginTop: '6px',
        }}
      >
        {CANONICAL_GEOMETRIES.map((geom) => {
          const isSelected = Math.abs(geom.height_mm - selectedHeight) < 1.0;
          return (
            <button
              key={geom.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectGeometry(geom)}
              style={{
                background: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'var(--bg-secondary)',
                border: isSelected ? '1px solid var(--color-accent)' : '1px solid var(--border-subtle)',
                borderRadius: '6px',
                padding: '10px 8px',
                textAlign: 'center',
                cursor: disabled ? 'not-allowed' : 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div
                style={{
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  color: isSelected ? 'var(--color-accent)' : 'var(--text-main)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                1000 × {geom.height_mm}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                α = {geom.aspect_ratio.toFixed(2)} | D_c = {geom.canonical_dc_mm.toFixed(1)} mm
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

