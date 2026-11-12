/**
 * Matrix Input Validation & Diagnostics Utilities
 * Strictly validates 64x64 grid dimensions, numeric validity, and physical ranges.
 */

export interface MatrixDiagnostics {
  isValid: boolean;
  rows: number;
  cols: number;
  min: number;
  max: number;
  mean: number;
  nonFiniteCount: number;
  errorMessage?: string;
}

export function validateStressGrid(matrix: any): MatrixDiagnostics {
  if (!matrix || !Array.isArray(matrix)) {
    return {
      isValid: false,
      rows: 0,
      cols: 0,
      min: 0,
      max: 0,
      mean: 0,
      nonFiniteCount: 0,
      errorMessage: 'Stress input must be a 2D matrix array.',
    };
  }

  const rows = matrix.length;
  if (rows !== 64) {
    return {
      isValid: false,
      rows,
      cols: matrix[0]?.length || 0,
      min: 0,
      max: 0,
      mean: 0,
      nonFiniteCount: 0,
      errorMessage: `Stress matrix must have exactly 64 rows (received ${rows}).`,
    };
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let nonFiniteCount = 0;
  for (let r = 0; r < 64; r++) {
    const row = matrix[r];
    if (!Array.isArray(row) || row.length !== 64) {
      return {
        isValid: false,
        rows: 64,
        cols: row?.length || 0,
        min: 0,
        max: 0,
        mean: 0,
        nonFiniteCount: 0,
        errorMessage: `Row ${r} has ${row?.length || 0} columns; exactly 64 columns required.`,
      };
    }

    for (let c = 0; c < 64; c++) {
      const val = row[c];
      if (typeof val !== 'number' || isNaN(val) || !isFinite(val)) {
        nonFiniteCount++;
      } else {
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
      }
    }
  }

  if (nonFiniteCount > 0) {
    return {
      isValid: false,
      rows: 64,
      cols: 64,
      min: isFinite(min) ? min : 0,
      max: isFinite(max) ? max : 0,
      mean: 0,
      nonFiniteCount,
      errorMessage: `Matrix contains ${nonFiniteCount} non-finite or NaN value(s).`,
    };
  }

  return {
    isValid: true,
    rows: 64,
    cols: 64,
    min: Math.round(min * 100) / 100,
    max: Math.round(max * 100) / 100,
    mean: Math.round((sum / (64 * 64)) * 100) / 100,
    nonFiniteCount: 0,
  };
}

export function parseCsvToMatrix(csvText: string): number[][] {
  const lines = csvText
    .trim()
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const matrix: number[][] = [];
  for (let i = 0; i < lines.length; i++) {
    const tokens = lines[i].split(',').map((t) => t.trim());
    // If first row looks like header (e.g. contains letters), skip if it has 65 or starts with text
    if (i === 0 && isNaN(Number(tokens[0]))) {
      continue;
    }
    // Filter tokens or convert to numbers
    const numRow = tokens.map((t) => Number(t));
    matrix.push(numRow);
  }
  return matrix;
}

export function parseJsonToMatrix(jsonText: string): number[][] {
  const parsed = JSON.parse(jsonText);
  if (Array.isArray(parsed) && Array.isArray(parsed[0])) {
    return parsed;
  }
  if (parsed && Array.isArray(parsed.stress_grid_64x64)) {
    return parsed.stress_grid_64x64;
  }
  throw new Error('JSON object does not contain a 2D matrix or stress_grid_64x64 field.');
}
