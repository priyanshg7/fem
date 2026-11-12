/**
 * Stage 5C & 5D Test Suite:
 * - Matrix Validation & CSV/JSON Parsing
 * - Colormap RGB Generation
 * - Case Record Hashing & Formatting
 * - Live Backend Inverse Prediction against FastAPI (Model D5 U-Net)
 */
import assert from 'node:assert/strict';
import { validateStressGrid, parseCsvToMatrix, parseJsonToMatrix } from '../src/utils/matrixValidator';
import { getViridisRgb } from '../src/utils/colormap';
import { computeContentHash, generateCaseId } from '../src/services/caseHistory';
import sampleStressData from '../src/data/sample_stress_field.json';

const BASE_URL = 'http://127.0.0.1:8000/api/v1';

async function postJson(endpoint: string, body: any) {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return { status: res.status, ok: res.ok, data };
}

export function testMatrixValidator() {
  console.log('  [TEST] Matrix Diagnostics & Dimensional Validation');

  // 1. Valid 64x64 matrix
  const validGrid = Array.from({ length: 64 }, () =>
    Array.from({ length: 64 }, () => 100.0)
  );
  const diagValid = validateStressGrid(validGrid);
  assert.equal(diagValid.isValid, true);
  assert.equal(diagValid.rows, 64);
  assert.equal(diagValid.cols, 64);
  assert.equal(diagValid.min, 100.0);
  assert.equal(diagValid.max, 100.0);

  // 2. Reject non-64 rows
  const smallGrid = Array.from({ length: 32 }, () =>
    Array.from({ length: 64 }, () => 100.0)
  );
  const diagSmall = validateStressGrid(smallGrid);
  assert.equal(diagSmall.isValid, false);
  assert(diagSmall.errorMessage?.includes('64 rows'));

  // 3. Reject non-finite values (NaN / Inf)
  const nanGrid = Array.from({ length: 64 }, () =>
    Array.from({ length: 64 }, () => 100.0)
  );
  nanGrid[10][20] = NaN;
  const diagNan = validateStressGrid(nanGrid);
  assert.equal(diagNan.isValid, false);
  assert.equal(diagNan.nonFiniteCount, 1);
  assert(diagNan.errorMessage?.includes('non-finite'));

  // 4. CSV parsing
  const csvLines = Array.from({ length: 64 }, () =>
    Array.from({ length: 64 }, () => '105.5').join(',')
  ).join('\n');
  const parsedCsv = parseCsvToMatrix(csvLines);
  assert.equal(parsedCsv.length, 64);
  assert.equal(parsedCsv[0].length, 64);
  assert.equal(parsedCsv[0][0], 105.5);

  // 5. JSON parsing
  const jsonStr = JSON.stringify({ stress_grid_64x64: validGrid });
  const parsedJson = parseJsonToMatrix(jsonStr);
  assert.equal(parsedJson.length, 64);
  assert.equal(parsedJson[0][0], 100.0);
}

export function testColormapAndHashing() {
  console.log('  [TEST] Colormap & Content Hashing');

  // Viridis rgb bounds
  const c0 = getViridisRgb(0.0);
  assert.equal(c0[0], 68);
  assert.equal(c0[1], 1);
  assert.equal(c0[2], 84);

  const c1 = getViridisRgb(1.0);
  assert.equal(c1[0], 253);
  assert.equal(c1[1], 231);
  assert.equal(c1[2], 37);

  // Case ID generation format
  const caseIdFwd = generateCaseId('FORWARD');
  assert(caseIdFwd.startsWith('CASE-FWD-'));
  const caseIdInv = generateCaseId('INVERSE');
  assert(caseIdInv.startsWith('CASE-INV-'));
}

export async function testLiveBackendInversePrediction() {
  console.log('  [TEST] Live Backend Inverse Localization (Stage 3D Model D5)');

  // 1. Confirmatory Specimen (SPEC-I02 1000x700 demo field)
  const req = {
    plate_length_mm: sampleStressData.plate_length_mm,
    plate_height_mm: sampleStressData.plate_height_mm,
    applied_traction_psi: sampleStressData.applied_traction_psi,
    stress_grid_64x64: sampleStressData.stress_grid_64x64,
    return_heatmap: true,
  };

  const res = await postJson('/predict/inverse', req);
  assert.equal(res.status, 200, `Inverse prediction failed: ${JSON.stringify(res.data)}`);
  const data = res.data;

  // Provenance & Identity
  assert.equal(data.success, true);
  assert.equal(data.provenance, 'ML_SURROGATE');
  assert.equal(data.model_id, 'inverse_unet_d5_stage3d');
  assert.equal(data.input_mode, 'FULL_FIELD_64x64');
  assert(data.prediction_id.startsWith('inv_'));

  // Numerical sanity: SPEC-I02 ground truth is (507.8, 244.4)
  assert(Math.abs(data.predicted_defect_x_mm - 507.75) < 1.0);
  assert(Math.abs(data.predicted_defect_y_mm - 244.44) < 1.0);

  // Dispersion & Abstention
  assert(data.uncertainty_radius_mm > 0 && data.uncertainty_radius_mm < 5.0);
  assert.equal(data.abstention.status, 'ACCEPTED');
  assert.equal(data.abstention.abstain, false);

  // Heatmap presence & dimensions
  assert(Array.isArray(data.heatmap_grid_64x64));
  assert.equal(data.heatmap_grid_64x64.length, 64);
  assert.equal(data.heatmap_grid_64x64[0].length, 64);

  // Anisotropic Ellipse
  assert(data.uncertainty_ellipse.semi_major_axis_mm > 0);
  assert(data.uncertainty_ellipse.semi_minor_axis_mm >= 0);

  // Latency (< 2500ms initial/warm)
  assert(data.execution_time_ms < 2500.0);
}

export async function testLiveBackendAbstentionAndError() {
  console.log('  [TEST] Live Backend Abstention Trigger & Error Handlers');

  // 1. Ambiguous synthetic uniform field (should trigger FLAGGED_FOR_REVIEW)
  const uniformGrid = Array.from({ length: 64 }, () =>
    Array.from({ length: 64 }, () => 100.0)
  );
  const ambigReq = {
    plate_length_mm: 1000.0,
    plate_height_mm: 1000.0,
    applied_traction_psi: 100.0,
    stress_grid_64x64: uniformGrid,
    return_heatmap: false,
    confidence_abstention_threshold_mm: 9.07,
  };
  const ambigRes = await postJson('/predict/inverse', ambigReq);
  assert.equal(ambigRes.status, 200);
  assert.equal(ambigRes.data.abstention.abstain, true);
  assert.equal(ambigRes.data.abstention.status, 'FLAGGED_FOR_REVIEW');
  assert(ambigRes.data.uncertainty_radius_mm > 9.07);
  assert.equal(ambigRes.data.heatmap_grid_64x64, null); // return_heatmap was false

  // 2. Malformed Grid Shape (400 Bad Request)
  const badReq = {
    plate_length_mm: 1000.0,
    plate_height_mm: 1000.0,
    stress_grid_64x64: [[100.0]], // 1x1 invalid shape
  };
  const badRes = await postJson('/predict/inverse', badReq);
  assert.equal(badRes.status, 400);
  const errorCode = badRes.data.error?.code || badRes.data.code;
  assert.equal(errorCode, 'INVALID_STRESS_GRID_SHAPE');
}
