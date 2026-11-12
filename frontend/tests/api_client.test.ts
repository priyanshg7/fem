/**
 * Unit tests for Centralized API Client and Error Handling.
 */
import assert from 'node:assert/strict';
import { ApiError } from '../src/api/client';

export function testApiErrorClass() {
  console.log('  [TEST] ApiError Class Construction & Serialization');
  const err = new ApiError(400, 'Physical collision detected', 'CENTRAL_HOLE_COLLISION', [
    { code: 'CENTRAL_HOLE_COLLISION', field: 'hole_to_hole_clearance_mm' },
  ]);

  assert.equal(err.status, 400);
  assert.equal(err.message, 'Physical collision detected');
  assert.equal(err.code, 'CENTRAL_HOLE_COLLISION');
  assert.equal(err.details?.length, 1);
  assert.equal(err.details[0].field, 'hole_to_hole_clearance_mm');
  assert.equal(err.name, 'ApiError');
}

export function testRequestSchemaSanity() {
  console.log('  [TEST] Forward Prediction Request Payload Typing');
  const payload = {
    plate_length_mm: 1000.0,
    plate_height_mm: 800.0,
    central_hole_diameter_mm: 66.667,
    defect_diameter_mm: 15.0,
    defect_center_x_mm: 250.0,
    defect_center_y_mm: 350.0,
    applied_traction_psi: 100.0,
    framework: 'heterogeneous' as const,
  };

  assert.equal(typeof payload.plate_length_mm, 'number');
  assert.equal(typeof payload.defect_center_x_mm, 'number');
  assert.equal(payload.applied_traction_psi, 100.0);
  assert.equal(payload.framework, 'heterogeneous');
}

