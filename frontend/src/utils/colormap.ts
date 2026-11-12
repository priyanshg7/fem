/**
 * Scientific Colormap Utility (Viridis)
 * Provides perceptually uniform colors for stress distributions and probability heatmaps.
 */

// 16 key control points for Viridis colormap: [t, r, g, b]
const VIRIDIS_STOPS: [number, number, number, number][] = [
  [0.0, 68, 1, 84],
  [0.067, 71, 19, 101],
  [0.133, 72, 36, 117],
  [0.2, 70, 52, 128],
  [0.267, 65, 68, 135],
  [0.333, 57, 86, 140],
  [0.4, 49, 104, 142],
  [0.467, 42, 120, 142],
  [0.533, 35, 138, 141],
  [0.6, 31, 158, 137],
  [0.667, 38, 177, 128],
  [0.733, 62, 197, 112],
  [0.8, 100, 214, 92],
  [0.867, 149, 227, 67],
  [0.933, 203, 237, 48],
  [1.0, 253, 231, 37],
];

export function getViridisColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = VIRIDIS_STOPS[i];
    const [t1, r1, g1, b1] = VIRIDIS_STOPS[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = (clamped - t0) / (t1 - t0);
      const r = Math.round(r0 + f * (r1 - r0));
      const g = Math.round(g0 + f * (g1 - g0));
      const b = Math.round(b0 + f * (b1 - b0));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return 'rgb(253, 231, 37)';
}

export function getViridisRgb(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = VIRIDIS_STOPS[i];
    const [t1, r1, g1, b1] = VIRIDIS_STOPS[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = (clamped - t0) / (t1 - t0);
      const r = Math.round(r0 + f * (r1 - r0));
      const g = Math.round(g0 + f * (g1 - g0));
      const b = Math.round(b0 + f * (b1 - b0));
      return [r, g, b];
    }
  }
  return [253, 231, 37];
}
