import React, { useRef, useState, useCallback } from 'react';
import { Crosshair, Move, AlertTriangle } from 'lucide-react';
import { Badge } from '../common/Badge';

interface PlateViewerProps {
  plateLength: number;
  plateHeight: number;
  centralHoleDiameter: number;
  defectDiameter: number;
  defectX: number;
  defectY: number;
  onChangeDefectPosition: (x: number, y: number) => void;
  isValid: boolean;
  hasCollision: boolean;
  ligamentClearance?: number;
  minBoundaryClearance?: number;
  readOnly?: boolean;
}

export const PlateViewer: React.FC<PlateViewerProps> = ({
  plateLength,
  plateHeight,
  centralHoleDiameter,
  defectDiameter,
  defectX,
  defectY,
  onChangeDefectPosition,
  isValid,
  hasCollision,
  ligamentClearance,
  minBoundaryClearance,
  readOnly = false,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // SVG viewport padding for dimension markers
  const margin = 50;
  const viewWidth = plateLength + margin * 2;
  const viewHeight = plateHeight + margin * 2;

  // Screen to Plate Coordinate Conversion
  const handlePointerCoord = useCallback(
    (clientX: number, clientY: number) => {
      if (!svgRef.current || readOnly) return;
      const rect = svgRef.current.getBoundingClientRect();
      const scaleX = viewWidth / rect.width;
      const scaleY = viewHeight / rect.height;

      const svgX = (clientX - rect.left) * scaleX;
      const svgY = (clientY - rect.top) * scaleY;

      // Map SVG coords (margin to margin + length) to physical mm
      // Note: SVG Y is down, Solid mechanics Y is up from bottom edge!
      const physX = Math.round(Math.max(5, Math.min(plateLength - 5, svgX - margin)));
      const physY = Math.round(Math.max(5, Math.min(plateHeight - 5, plateHeight - (svgY - margin))));

      onChangeDefectPosition(physX, physY);
    },
    [plateLength, plateHeight, viewWidth, viewHeight, readOnly, onChangeDefectPosition]
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    if (readOnly) return;
    setIsDragging(true);
    handlePointerCoord(e.clientX, e.clientY);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || readOnly) return;
    handlePointerCoord(e.clientX, e.clientY);
  };

  const handleMouseUp = () => setIsDragging(false);

  // Coordinates inside SVG system (with Y flipped so y=0 is bottom)
  const plateSvgX = margin;
  const plateSvgY = margin;
  const centerHoleX = margin + plateLength / 2.0;
  const centerHoleY = margin + plateHeight / 2.0;

  const defectSvgX = margin + defectX;
  const defectSvgY = margin + (plateHeight - defectY); // flipped for standard mechanics convention

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <Crosshair size={14} style={{ color: 'var(--color-accent)' }} />
          <span>Interactive Geometry Visualizer</span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>(Drag or click to reposition defect)</span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {hasCollision ? (
            <Badge variant="danger" icon={<AlertTriangle size={12} />}>
              COLLISION DETECTED
            </Badge>
          ) : isValid ? (
            <Badge variant="success">
              GEOMETRY VALID
            </Badge>
          ) : (
            <Badge variant="warning">
              ADVISORY WARNING
            </Badge>
          )}
          <Badge variant="neutral">
            Scale: {plateLength} × {plateHeight} mm
          </Badge>
        </div>
      </div>

      <div
        className="plate-canvas-container"
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${viewWidth} ${viewHeight}`}
          className="plate-svg"
          style={{ width: '100%', maxHeight: '520px', userSelect: 'none' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
        >
          <defs>
            {/* Grid pattern */}
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
            </pattern>
            {/* Defect Glow */}
            <filter id="defectGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Grid */}
          <rect width={viewWidth} height={viewHeight} fill="#070b14" />
          <rect
            x={plateSvgX}
            y={plateSvgY}
            width={plateLength}
            height={plateHeight}
            fill="#0c1222"
            stroke="var(--border-strong)"
            strokeWidth="2"
          />
          <rect
            x={plateSvgX}
            y={plateSvgY}
            width={plateLength}
            height={plateHeight}
            fill="url(#grid)"
          />

          {/* Tensile Traction Arrows (Uniaxial loading along X-axis: tension on left & right boundaries) */}
          {[-1, 1].map((dir, idx) => (
            <g key={idx}>
              {[0.2, 0.4, 0.6, 0.8].map((frac, aIdx) => {
                const arrowY = plateSvgY + plateHeight * frac;
                const startX = dir === 1 ? plateSvgX + plateLength : plateSvgX;
                const endX = dir === 1 ? startX + 22 : startX - 22;
                return (
                  <g key={aIdx} stroke="rgba(56, 189, 248, 0.4)" strokeWidth="1.5">
                    <line x1={startX} y1={arrowY} x2={endX} y2={arrowY} />
                    <polygon
                      points={
                        dir === 1
                          ? `${endX},${arrowY} ${endX - 5},${arrowY - 3} ${endX - 5},${arrowY + 3}`
                          : `${endX},${arrowY} ${endX + 5},${arrowY - 3} ${endX + 5},${arrowY + 3}`
                      }
                      fill="rgba(56, 189, 248, 0.5)"
                    />
                  </g>
                );
              })}
            </g>
          ))}

          {/* Canonical Central Circular Hole (Dc = H/12) */}
          <circle
            cx={centerHoleX}
            cy={centerHoleY}
            r={centralHoleDiameter / 2.0}
            fill="#070b14"
            stroke="var(--color-hole)"
            strokeWidth="2.5"
            strokeDasharray="4 2"
          />
          {/* Central Hole Center Cross */}
          <line
            x1={centerHoleX - 6}
            y1={centerHoleY}
            x2={centerHoleX + 6}
            y2={centerHoleY}
            stroke="var(--color-hole)"
            strokeWidth="1.5"
          />
          <line
            x1={centerHoleX}
            y1={centerHoleY - 6}
            x2={centerHoleX}
            y2={centerHoleY + 6}
            stroke="var(--color-hole)"
            strokeWidth="1.5"
          />

          {/* Ligament Line between Central Hole & Defect */}
          <line
            x1={centerHoleX}
            y1={centerHoleY}
            x2={defectSvgX}
            y2={defectSvgY}
            stroke={hasCollision ? 'var(--color-danger)' : 'rgba(245, 158, 11, 0.4)'}
            strokeWidth="1.5"
            strokeDasharray="3 3"
          />

          {/* Defect Hole (User positioned) */}
          <circle
            cx={defectSvgX}
            cy={defectSvgY}
            r={defectDiameter / 2.0}
            fill={hasCollision ? 'rgba(244, 63, 94, 0.4)' : '#070b14'}
            stroke={hasCollision ? 'var(--color-danger)' : 'var(--color-defect)'}
            strokeWidth="3"
            filter="url(#defectGlow)"
          />
          {/* Defect Position Marker Reticle */}
          <circle
            cx={defectSvgX}
            cy={defectSvgY}
            r="3"
            fill={hasCollision ? 'var(--color-danger)' : 'var(--color-defect)'}
          />

          {/* Dimension Lines: Length L */}
          <line
            x1={plateSvgX}
            y1={plateSvgY + plateHeight + 18}
            x2={plateSvgX + plateLength}
            y2={plateSvgY + plateHeight + 18}
            stroke="var(--text-dim)"
            strokeWidth="1"
          />
          <text
            x={plateSvgX + plateLength / 2.0}
            y={plateSvgY + plateHeight + 34}
            fill="var(--text-muted)"
            fontSize="12"
            fontFamily="var(--font-mono)"
            textAnchor="middle"
          >
            L = {plateLength} mm
          </text>

          {/* Dimension Lines: Height H */}
          <line
            x1={plateSvgX - 18}
            y1={plateSvgY}
            x2={plateSvgX - 18}
            y2={plateSvgY + plateHeight}
            stroke="var(--text-dim)"
            strokeWidth="1"
          />
          <text
            x={plateSvgX - 26}
            y={plateSvgY + plateHeight / 2.0}
            fill="var(--text-muted)"
            fontSize="12"
            fontFamily="var(--font-mono)"
            textAnchor="middle"
            transform={`rotate(-90 ${plateSvgX - 26} ${plateSvgY + plateHeight / 2.0})`}
          >
            H = {plateHeight} mm
          </text>

          {/* Physical Coordinate Label readout */}
          <g transform={`translate(${defectSvgX + defectDiameter / 2.0 + 8}, ${defectSvgY - 8})`}>
            <rect
              x="-4"
              y="-14"
              width="150"
              height="28"
              fill="rgba(15, 23, 42, 0.9)"
              stroke="var(--border-subtle)"
              rx="4"
            />
            <text
              x="4"
              y="-2"
              fill="var(--text-main)"
              fontSize="10"
              fontFamily="var(--font-mono)"
            >
              ({defectX}, {defectY}) d={defectDiameter}mm
            </text>
            <text
              x="4"
              y="9"
              fill="var(--color-accent)"
              fontSize="8.5"
              fontFamily="var(--font-mono)"
            >
              c_lig: {ligamentClearance !== undefined ? `${ligamentClearance.toFixed(1)}mm` : '--'} | c_bnd: {minBoundaryClearance !== undefined ? `${minBoundaryClearance.toFixed(1)}mm` : '--'}
            </text>
          </g>
        </svg>

        {/* Floating helper guide overlay */}
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            right: '12px',
            background: 'rgba(10, 13, 20, 0.85)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '6px 10px',
            fontSize: '0.7rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backdropFilter: 'blur(4px)',
          }}
        >
          <Move size={12} style={{ color: 'var(--color-accent)' }} />
          <span>Click plate or drag defect to set coordinates</span>
        </div>
      </div>
    </div>
  );
};
