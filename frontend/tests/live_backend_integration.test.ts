/**
 * Live Backend Integration Tests
 * Executes actual HTTP requests against the FastAPI backend service (:8000/api/v1)
 * verifying end-to-end integration between frontend client and Python surrogate models.
 */
import assert from 'node:assert/strict';

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

async function getJson(endpoint: string) {
  const res = await fetch(`${BASE_URL}${endpoint}`);
  const data = await res.json();
  return { status: res.status, ok: res.ok, data };
}

export async function testLiveHealthAndReadiness() {
  console.log('  [TEST] Live Backend Health and Readiness Probes');
  const health = await getJson('/health');
  assert.equal(health.status, 200, `Expected 200 OK from /health, got ${health.status}`);
  assert.equal(health.data.status, 'healthy');
  assert.equal(health.data.ready, true);
  assert.equal(health.data.active_forward_model, 'forward_ensemble_stage3b');

  const readiness = await getJson('/readiness');
  assert.equal(readiness.status, 200);
  assert.equal(readiness.data.ready, true);
  assert.equal(readiness.data.all_models_loaded, true);
}

export async function testLiveModelRegistry() {
  console.log('  [TEST] Live Model Registry Metadata Reflection');
  const models = await getJson('/models');
  assert.equal(models.status, 200);
  const modelList = Array.isArray(models.data) ? models.data : models.data.models;
  assert(Array.isArray(modelList), 'models list should be an array');
  assert(modelList.some((m: any) => m.model_id === 'forward_ensemble_stage3b'));

  const detail = await getJson('/models/forward_ensemble_stage3b');
  assert.equal(detail.status, 200);
  assert.equal(detail.data.model_id, 'forward_ensemble_stage3b');
  assert.equal(detail.data.category, 'forward');
  assert.equal(detail.data.version, '3.2.0-frozen');
  assert(detail.data.operating_domain.aspect_ratios_supported.includes(1.0));
}

export async function testLiveGeometryValidation() {
  console.log('  [TEST] Live Geometry Validation Workflow');
  // 1. Valid Canonical Specimen
  const validReq = {
    plate_length_mm: 1000.0,
    plate_height_mm: 1000.0,
    central_hole_diameter_mm: 83.333,
    defect_diameter_mm: 20.0,
    defect_center_x_mm: 300.0,
    defect_center_y_mm: 700.0,
  };
  const validRes = await postJson('/validate/geometry', validReq);
  assert.equal(validRes.status, 200);
  assert.equal(validRes.data.is_valid, true);
  assert.equal(validRes.data.can_proceed_to_inference, true);
  assert.equal(validRes.data.status, 'VALID');
  assert(validRes.data.clearances.hole_to_hole_clearance_mm > 0);

  // 2. Collision with Central Hole
  const collisionReq = {
    plate_length_mm: 1000.0,
    plate_height_mm: 1000.0,
    central_hole_diameter_mm: 83.333,
    defect_diameter_mm: 20.0,
    defect_center_x_mm: 500.0,
    defect_center_y_mm: 500.0,
  };
  const collisionRes = await postJson('/validate/geometry', collisionReq);
  assert.equal(collisionRes.status, 200);
  assert.equal(collisionRes.data.is_valid, false);
  assert.equal(collisionRes.data.can_proceed_to_inference, false);
  assert.equal(collisionRes.data.status, 'INVALID');
  assert(collisionRes.data.issues.some((i: any) => i.code === 'CENTRAL_HOLE_COLLISION'));
}

export async function testLiveForwardPrediction() {
  console.log('  [TEST] Live Forward Surrogate Prediction (Stage 3B Ensemble)');
  const predReq = {
    plate_length_mm: 1000.0,
    plate_height_mm: 1000.0,
    central_hole_diameter_mm: 83.333,
    defect_diameter_mm: 20.0,
    defect_center_x_mm: 300.0,
    defect_center_y_mm: 700.0,
    applied_traction_psi: 100.0,
    framework: 'heterogeneous',
  };

  const predRes = await postJson('/predict/forward', predReq);
  assert.equal(predRes.status, 200, `Prediction failed: ${JSON.stringify(predRes.data)}`);
  const data = predRes.data;

  // Provenance & Identity
  assert.equal(data.success, true);
  assert.equal(data.provenance, 'ML_SURROGATE');
  assert.equal(data.model_id, 'forward_ensemble_stage3b');
  assert(data.prediction_id.startsWith('fwd_'));

  // Numerical sanity
  assert(data.predicted_max_stress_psi > 250.0 && data.predicted_max_stress_psi < 350.0);
  assert(Math.abs(data.predicted_Kt_element - data.predicted_max_stress_psi / 100.0) < 0.001);
  assert(data.ensemble_uncertainty_psi > 0);

  // Estimators breakdown
  const baseModels = data.base_model_predictions_Kt;
  assert(baseModels.ANN_512 !== undefined);
  assert(baseModels.DeepMLP_D2 !== undefined);
  assert(baseModels.ResMLP !== undefined);
  assert(baseModels.RandomForest !== undefined);
  assert(baseModels.ExtraTrees !== undefined);
  assert(baseModels.HistGradientBoosting !== undefined);

  // Performance latency check (< 2500ms initial, < 150ms warm)
  assert(data.execution_time_ms < 2500.0, `Initial latency exceeded target: ${data.execution_time_ms} ms`);

  // Warm call latency verification
  const warmRes = await postJson('/predict/forward', predReq);
  assert.equal(warmRes.status, 200);
  assert(warmRes.data.execution_time_ms < 150.0, `Warm latency exceeded target: ${warmRes.data.execution_time_ms} ms`);
}
