import React, { useRef, useEffect, useState, useMemo } from 'react';
import { UncertaintyEllipse, ReferenceGroundTruth } from '../../api/types';
import { getViridisRgb } from '../../utils/colormap';
import { Eye, Layers, Crosshair } from 'lucide-react';

interface HeatmapViewerProps {
  plateLengthMm: number;
  plateHeightMm: number;
  heatmapGrid?: number[][] | null;
  stressGrid?: number[][] | null;
  predictedX?: number;
  predictedY?: number;
  uncertaintyRadiusMm?: number;
  uncertaintyEllipse?: UncertaintyEllipse;
  referenceGroundTruth?: ReferenceGroundTruth;
  abstentionStatus?: string;
  isStale?: boolean;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  plateLengthMm,
  plateHeightMm,
  heatmapGrid,
  stressGrid,
  predictedX,
  predictedY,
  uncertaintyRadiusMm,
  uncertaintyEllipse: _uncertaintyEllipse,
  referenceGroundTruth,
  abstentionStatus,
  isStale,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [selectedMode, setSelectedMode] = useState<'auto' | 'heatmap' | 'stress'>('auto');
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number; val: number | null } | null>(null);

  // Determine active view mode
  const hasHeatmap = Boolean(heatmapGrid && heatmapGrid.length === 64);
  const hasStress = Boolean(stressGrid && stressGrid.length === 64);

  const activeMode: 'heatmap' | 'stress' =
    selectedMode === 'auto'
      ? hasHeatmap
        ? 'heatmap'
        : 'stress'
      : selectedMode === 'heatmap' && hasHeatmap
      ? 'heatmap'
      : 'stress';

  // Canonical central hole diameter Dc = H / 12
  const centralHoleDiameter = plateHeightMm / 12;

  // Viewport setup (responsive scaling)
  const margin = 48;
  const targetWidth = 620;
  const scale = targetWidth / plateLengthMm;
  const plateWidthSvg = plateLengthMm * scale;
  const plateHeightSvg = plateHeightMm * scale;
  const totalSvgWidth = plateWidthSvg + margin * 2;
  const totalSvgHeight = plateHeightSvg + margin * 2;

  // Calculate stress / heatmap statistics
  const stats = useMemo(() => {
    const target = activeMode === 'heatmap' ? heatmapGrid : stressGrid;
    if (!target || target.length !== 64) return { min: 0, max: 100, mean: 50 };

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let r = 0; r < 64; r++) {
      for (let c = 0; c < 64; c++) {
        const v = target[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
      }
    }
    return {
      min: Math.round(min * 10) / 10,
      max: Math.round(max * 10) / 10,
      mean: Math.round((sum / 4096) * 10) / 10,
    };
  }, [activeMode, heatmapGrid, stressGrid]);

  // Render 64x64 Matrix onto Canvas with vertical flip (row 0 = physical y=0, row 63 = physical y=H)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const targetGrid = activeMode === 'heatmap' ? heatmapGrid : stressGrid;
    if (!targetGrid || targetGrid.length !== 64) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(64, 64);
    const data = imgData.data;

    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < 64; r++) {
      for (let c = 0; c < 64; c++) {
        const v = targetGrid[r][c];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }

    const range = maxVal - minVal > 1e-5 ? maxVal - minVal : 1;

    for (let py = 0; py < 64; py++) {
      // Screen row py=0 is top (physical y = H, row index 63)
      const dataRow = 63 - py;
      for (let px = 0; px < 64; px++) {
        const val = targetGrid[dataRow][px];
        const normalized =
          activeMode === 'heatmap'
            ? Math.max(0, Math.min(1, maxVal > 0 ? val / maxVal : 0))
            : Math.max(0, Math.min(1, (val - minVal) / range));

        const [r, g, b] = getViridisRgb(normalized);

        const idx = (py * 64 + px) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = activeMode === 'heatmap' ? Math.round(180 + normalized * 75) : 240;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [activeMode, heatmapGrid, stressGrid]);

  // Coordinate transforms (physical mm -> SVG canvas)
  const toSvgX = (xMm: number) => margin + xMm * scale;
  const toSvgY = (yMm: number) => margin + (plateHeightMm - yMm) * scale;

  const predSvgX = predictedX !== undefined ? toSvgX(predictedX) : null;
  const predSvgY = predictedY !== undefined ? toSvgY(predictedY) : null;

  const trueSvgX =
    referenceGroundTruth?.defect_center_x_mm !== undefined
      ? toSvgX(referenceGroundTruth.defect_center_x_mm)
      : null;
  const trueSvgY =
    referenceGroundTruth?.defect_center_y_mm !== undefined
      ? toSvgY(referenceGroundTruth.defect_center_y_mm)
      : null;

  const centralHoleX = toSvgX(plateLengthMm / 2);
  const centralHoleY = toSvgY(plateHeightMm / 2);
  const centralHoleRadiusSvg = (centralHoleDiameter / 2) * scale;

  // Handle Mouse Hover over plate for physical coordinate inspection
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    // Convert client SVG coordinates to plate coordinates
    const scaleFactor = totalSvgWidth / rect.width;
    const svgX = clientX * scaleFactor;
    const svgY = clientY * scaleFactor;

    if (
      svgX >= margin &&
      svgX <= margin + plateWidthSvg &&
      svgY >= margin &&
      svgY <= margin + plateHeightSvg
    ) {
      const xMm = Math.round(((svgX - margin) / scale) * 10) / 10;
      const yMm = Math.round(((margin + plateHeightSvg - svgY) / scale) * 10) / 10;

      // Lookup cell value in 64x64 grid
      const col = Math.min(63, Math.max(0, Math.floor((xMm / plateLengthMm) * 64)));
      const row = Math.min(63, Math.max(0, Math.floor((yMm / plateHeightMm) * 64)));

      const target = activeMode === 'heatmap' ? heatmapGrid : stressGrid;
      const val = target && target[row] ? target[row][col] : null;

      setHoverCoord({ x: xMm, y: yMm, val });
    } else {
      setHoverCoord(null);
    }
  };

  const handleMouseLeave = () => setHoverCoord(null);

  return (
    <div className="bg-[#0b101b] border border-[#1e293b] rounded-xl p-4 shadow-2xl flex flex-col items-center">
      {/* Top Header & Viewport Toolbar */}
      <div className="w-full flex flex-wrap items-center justify-between pb-3 border-b border-[#1e293b] gap-2">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-accent-cyan" />
          <span className="font-semibold text-text-primary text-sm tracking-wide">
            Digital Twin CAD / Spatial Viewport
          </span>
        </div>

        {/* Mode Switcher Pill */}
        <div className="flex items-center gap-2">
          <div className="bg-[#070b14] p-0.5 rounded-lg border border-[#1e293b] flex items-center">
            <button
              type="button"
              onClick={() => setSelectedMode('heatmap')}
              disabled={!hasHeatmap}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeMode === 'heatmap'
                  ? 'bg-accent-cyan text-[#070b14] font-bold shadow-sm'
                  : 'text-text-muted hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>U-Net Heatmap P(x,y)</span>
            </button>
            <button
              type="button"
              onClick={() => setSelectedMode('stress')}
              disabled={!hasStress}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeMode === 'stress'
                  ? 'bg-accent-cyan text-[#070b14] font-bold shadow-sm'
                  : 'text-text-muted hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Stress Field σ_vm</span>
            </button>
          </div>

          {/* Status Badges */}
          {abstentionStatus && (
            <span
              className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-wide border ${
                abstentionStatus === 'ACCEPTED'
                  ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                  : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
              }`}
            >
              {abstentionStatus}
            </span>
          )}

          {isStale && (
            <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-mono text-[10px]">
              STALE
            </span>
          )}
        </div>
      </div>

      {/* Main Interactive CAD SVG Viewport */}
      <div className="relative mt-3 flex justify-center items-center overflow-x-auto w-full bg-[#070a12] p-2 rounded-lg border border-[#151e30]">
        <svg
          width={totalSvgWidth}
          height={totalSvgHeight}
          className="select-none font-mono"
          style={{ maxWidth: '100%', height: 'auto' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            {/* Reticle Glow Filter */}
            <filter id="reticleGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#38bdf8" floodOpacity="0.9" />
            </filter>
            <filter id="truthGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#10b981" floodOpacity="0.9" />
            </filter>
            {/* Grid Pattern */}
            <pattern id="cadGrid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#141c2c" strokeWidth="0.5" />
            </pattern>
          </defs>

          {/* Background Grid */}
          <rect width={totalSvgWidth} height={totalSvgHeight} fill="url(#cadGrid)" />

          {/* Applied Traction Arrows (Top: Upward, Bottom: Downward) */}
          {Array.from({ length: 9 }).map((_, i) => {
            const arrowX = margin + (plateWidthSvg / 10) * (i + 1);
            return (
              <g key={`arrow-${i}`}>
                {/* Top Arrow */}
                <line x1={arrowX} y1={margin - 2} x2={arrowX} y2={margin - 16} stroke="#0284c7" strokeWidth="1.5" />
                <polygon points={`${arrowX - 3},${margin - 13} ${arrowX + 3},${margin - 13} ${arrowX},${margin - 18}`} fill="#38bdf8" />
                {/* Bottom Arrow */}
                <line x1={arrowX} y1={margin + plateHeightSvg + 2} x2={arrowX} y2={margin + plateHeightSvg + 16} stroke="#0284c7" strokeWidth="1.5" />
                <polygon points={`${arrowX - 3},${margin + plateHeightSvg + 13} ${arrowX + 3},${margin + plateHeightSvg + 13} ${arrowX},${margin + plateHeightSvg + 18}`} fill="#38bdf8" />
              </g>
            );
          })}

          {/* Plate Outline Background */}
          <rect
            x={margin}
            y={margin}
            width={plateWidthSvg}
            height={plateHeightSvg}
            fill="#090e18"
            stroke="#222f4c"
            strokeWidth="2"
            rx="2"
          />

          {/* Canvas Image Layer (Stress Contour or U-Net Heatmap) */}
          {(hasHeatmap || hasStress) && (
            <foreignObject x={margin} y={margin} width={plateWidthSvg} height={plateHeightSvg}>
              <canvas
                ref={canvasRef}
                width={64}
                height={64}
                style={{
                  width: '100%',
                  height: '100%',
                  imageRendering: 'pixelated',
                  display: 'block',
                }}
              />
            </foreignObject>
          )}

          {/* CAD Centerlines */}
          <line
            x1={margin}
            y1={centralHoleY}
            x2={margin + plateWidthSvg}
            y2={centralHoleY}
            stroke="#1e293b"
            strokeWidth="1"
            strokeDasharray="6 3 2 3"
          />
          <line
            x1={centralHoleX}
            y1={margin}
            x2={centralHoleX}
            y2={margin + plateHeightSvg}
            stroke="#1e293b"
            strokeWidth="1"
            strokeDasharray="6 3 2 3"
          />

          {/* Central Hole Cutout */}
          <circle
            cx={centralHoleX}
            cy={centralHoleY}
            r={centralHoleRadiusSvg}
            fill="#070a12"
            stroke="#38bdf8"
            strokeWidth="2.5"
            strokeDasharray="4 2"
          />
          <text
            x={centralHoleX}
            y={centralHoleY + 3.5}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize="9"
            fontWeight="bold"
          >
            Dc = {centralHoleDiameter.toFixed(1)}
          </text>

          {/* Dimension Lines */}
          {/* Top: Length L */}
          <line x1={margin} y1={margin - 24} x2={margin + plateWidthSvg} y2={margin - 24} stroke="#475569" strokeWidth="1" />
          <line x1={margin} y1={margin - 28} x2={margin} y2={margin - 20} stroke="#475569" strokeWidth="1" />
          <line x1={margin + plateWidthSvg} y1={margin - 28} x2={margin + plateWidthSvg} y2={margin - 20} stroke="#475569" strokeWidth="1" />
          <text x={margin + plateWidthSvg / 2} y={margin - 28} textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="bold">
            L = {plateLengthMm} mm
          </text>

          {/* Left: Height H */}
          <line x1={margin - 24} y1={margin} x2={margin - 24} y2={margin + plateHeightSvg} stroke="#475569" strokeWidth="1" />
          <line x1={margin - 28} y1={margin} x2={margin - 20} y2={margin} stroke="#475569" strokeWidth="1" />
          <line x1={margin - 28} y1={margin + plateHeightSvg} x2={margin - 20} y2={margin + plateHeightSvg} stroke="#475569" strokeWidth="1" />
          <text
            x={margin - 28}
            y={margin + plateHeightSvg / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="10"
            fontWeight="bold"
            transform={`rotate(-90, ${margin - 28}, ${margin + plateHeightSvg / 2})`}
          >
            H = {plateHeightMm} mm
          </text>

          {/* Connecting Line Between Truth and Prediction if both exist */}
          {predSvgX !== null && predSvgY !== null && trueSvgX !== null && trueSvgY !== null && (
            <line
              x1={trueSvgX}
              y1={trueSvgY}
              x2={predSvgX}
              y2={predSvgY}
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeDasharray="3 3"
            />
          )}

          {/* Reference Ground Truth Marker (Emerald) */}
          {trueSvgX !== null && trueSvgY !== null && (
            <g>
              <circle
                cx={trueSvgX}
                cy={trueSvgY}
                r="7"
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="2 2"
                filter="url(#truthGlow)"
              />
              <circle cx={trueSvgX} cy={trueSvgY} r="2" fill="#10b981" />
              {/* Truth Label */}
              <text
                x={trueSvgX + 10}
                y={trueSvgY + 4}
                fill="#10b981"
                fontSize="9"
                fontWeight="bold"
              >
                Truth ({referenceGroundTruth?.defect_center_x_mm?.toFixed(1)}, {referenceGroundTruth?.defect_center_y_mm?.toFixed(1)})
              </text>
            </g>
          )}

          {/* Predicted Defect Reticle & Dispersion Ellipse (Cyan) */}
          {predSvgX !== null && predSvgY !== null && (
            <g>
              {/* Uncertainty Dispersion Circle */}
              {uncertaintyRadiusMm && (
                <circle
                  cx={predSvgX}
                  cy={predSvgY}
                  r={uncertaintyRadiusMm * scale}
                  fill="rgba(56, 189, 248, 0.08)"
                  stroke="#38bdf8"
                  strokeWidth="1.2"
                  strokeDasharray="4 2"
                />
              )}

              {/* Reticle Crosshair */}
              <circle
                cx={predSvgX}
                cy={predSvgY}
                r="10"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
                filter="url(#reticleGlow)"
              />
              <line x1={predSvgX - 16} y1={predSvgY} x2={predSvgX + 16} y2={predSvgY} stroke="#38bdf8" strokeWidth="1.5" />
              <line x1={predSvgX} y1={predSvgY - 16} x2={predSvgX} y2={predSvgY + 16} stroke="#38bdf8" strokeWidth="1.5" />
              <circle cx={predSvgX} cy={predSvgY} r="2.5" fill="#ffffff" />

              {/* Coordinate Callout Tag */}
              <rect
                x={predSvgX - 45}
                y={predSvgY - 28}
                width="90"
                height="16"
                fill="#070b14"
                stroke="#38bdf8"
                strokeWidth="1"
                rx="3"
              />
              <text
                x={predSvgX}
                y={predSvgY - 17}
                textAnchor="middle"
                fill="#38bdf8"
                fontSize="9"
                fontWeight="bold"
              >
                Pred: ({predictedX?.toFixed(1)}, {predictedY?.toFixed(1)})
              </text>
            </g>
          )}
        </svg>

        {/* Floating Mouse Inspection HUD */}
        {hoverCoord && (
          <div className="absolute bottom-4 left-4 bg-[#070b14]/90 border border-accent-cyan/50 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] font-mono text-text-primary shadow-lg flex items-center gap-2">
            <span className="text-accent-cyan font-bold">X: {hoverCoord.x} mm</span>
            <span className="text-accent-cyan font-bold">Y: {hoverCoord.y} mm</span>
            {hoverCoord.val !== null && (
              <span className="text-emerald-400 font-bold border-l border-[#1e293b] pl-2">
                {activeMode === 'heatmap'
                  ? `P: ${(hoverCoord.val).toFixed(4)}`
                  : `σ: ${hoverCoord.val.toFixed(1)} psi`}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Bottom Colorbar Legend */}
      <div className="w-full mt-3 flex items-center justify-between text-xs font-mono text-text-muted">
        <span className="text-[11px]">
          {activeMode === 'heatmap' ? '0.0 (Zero Prob)' : `${stats.min} psi (Min)`}
        </span>

        {/* Colormap Gradient Bar */}
        <div className="flex-1 mx-4">
          <div
            className="h-3 rounded-full border border-[#1e293b]"
            style={{
              background:
                'linear-gradient(to right, #440154, #482878, #3e4a89, #31688e, #26828e, #1f9e89, #35b779, #6ece58, #b5de2b, #fde725)',
            }}
          />
          <div className="text-[10px] text-center text-text-dim mt-0.5">
            Viridis Colormap ({activeMode === 'heatmap' ? 'Softmax Heatmap Density' : 'Von Mises Stress Distribution'})
          </div>
        </div>

        <span className="text-[11px] font-bold text-accent-cyan">
          {activeMode === 'heatmap' ? '1.0 (Peak Density)' : `${stats.max} psi (Max)`}
        </span>
      </div>
    </div>
  );
};
