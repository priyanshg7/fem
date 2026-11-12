/**
 * Master Test Runner for Frontend Engineering Dashboard.
 * Executes Unit, Integration, Geometry, Live Forward Surrogate,
 * Live D5 Inverse Surrogate, and Reporting Tests.
 */
import { testCanonicalGeometryFamilies, testCoordinateClamping } from './geometry_rules.test';
import { testApiErrorClass, testRequestSchemaSanity } from './api_client.test';
import {
  testLiveHealthAndReadiness,
  testLiveModelRegistry,
  testLiveGeometryValidation,
  testLiveForwardPrediction,
} from './live_backend_integration.test';
import {
  testMatrixValidator,
  testColormapAndHashing,
  testLiveBackendInversePrediction,
  testLiveBackendAbstentionAndError,
} from './inverse_and_reporting.test';

async function main() {
  console.log('====================================================');
  console.log('SIMUSTRUCT REACT ENGINEERING DASHBOARD TEST SUITE');
  console.log('====================================================');

  const tests = [
    // Stage 5A Foundation Tests
    { name: 'Canonical Geometry Families', fn: testCanonicalGeometryFamilies },
    { name: 'Coordinate Clamping Bounds', fn: testCoordinateClamping },
    { name: 'ApiError Serialization', fn: testApiErrorClass },
    { name: 'Request Payload Typing', fn: testRequestSchemaSanity },
    { name: 'Live Backend Health & Readiness', fn: testLiveHealthAndReadiness },
    { name: 'Live Model Registry Reflection', fn: testLiveModelRegistry },
    // Stage 5B Forward Tests
    { name: 'Live Geometry Validation', fn: testLiveGeometryValidation },
    { name: 'Live Forward Surrogate Prediction', fn: testLiveForwardPrediction },
    // Stage 5C Inverse & 5D Reporting Tests
    { name: 'Matrix Diagnostics & Validation', fn: testMatrixValidator },
    { name: 'Colormap & Content Hashing', fn: testColormapAndHashing },
    { name: 'Live Backend Inverse Localization (D5)', fn: testLiveBackendInversePrediction },
    { name: 'Live Backend Abstention & Error Handling', fn: testLiveBackendAbstentionAndError },
  ];

  let passed = 0;
  let failed = 0;

  for (const t of tests) {
    try {
      const t0 = performance.now();
      await t.fn();
      const elapsed = (performance.now() - t0).toFixed(1);
      console.log(`[PASS] ${t.name} (${elapsed} ms)`);
      passed++;
    } catch (err: any) {
      console.error(`[FAIL] ${t.name}:`, err.message || err);
      failed++;
    }
  }

  console.log('\n====================================================');
  console.log(`FRONTEND TEST SUMMARY: ${passed} PASSED, ${failed} FAILED (TOTAL ${tests.length})`);
  console.log('====================================================');

  if (failed > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

main();
