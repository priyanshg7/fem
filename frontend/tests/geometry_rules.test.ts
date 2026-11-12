/**
 * Unit tests for Frontend Geometry Rules and Canonical Plate Families.
 */
import assert from 'node:assert/strict';
import { CANONICAL_GEOMETRIES } from '../src/components/forward/GeometrySelector';

export function testCanonicalGeometryFamilies() {
  console.log('  [TEST] Canonical Geometry Families (5 Plates)');
  assert.equal(CANONICAL_GEOMETRIES.length, 5, 'Must define exactly 5 canonical families');

  // Verify L = 1000 mm constant across all
  for (const geom of CANONICAL_GEOMETRIES) {
    assert.equal(geom.length_mm, 1000.0, `Plate length must be 1000 mm (got ${geom.length_mm})`);
  }

  // Verify Dc = H / 12 rule
  const expectedHeights = [1000.0, 900.0, 800.0, 700.0, 600.0];
  for (let i = 0; i < CANONICAL_GEOMETRIES.length; i++) {
    const geom = CANONICAL_GEOMETRIES[i];
    assert.equal(geom.height_mm, expectedHeights[i]);
    const expectedDc = geom.height_mm / 12.0;
    assert(
      Math.abs(geom.canonical_dc_mm - expectedDc) < 0.001,
      `Dc must equal H / 12 (expected ${expectedDc}, got ${geom.canonical_dc_mm})`
    );
    const expectedAr = 1000.0 / geom.height_mm;
    assert(
      Math.abs(geom.aspect_ratio - expectedAr) < 0.002,
      `Aspect ratio must equal 1000 / H (expected ${expectedAr}, got ${geom.aspect_ratio})`
    );
  }
}

export function testCoordinateClamping() {
  console.log('  [TEST] Screen to Physical Coordinate Mapping Bounds');
  const L = 1000.0;
  const H = 800.0;
  
  // Boundary clamping tests
  const clampX = (val: number) => Math.max(5, Math.min(L - 5, val));
  const clampY = (val: number) => Math.max(5, Math.min(H - 5, val));

  assert.equal(clampX(-50), 5, 'Negative X must clamp to min 5 mm');
  assert.equal(clampX(1500), 995, 'Excess X must clamp to L - 5 mm');
  assert.equal(clampY(-10), 5, 'Negative Y must clamp to min 5 mm');
  assert.equal(clampY(950), 795, 'Excess Y must clamp to H - 5 mm');
  assert.equal(clampX(500), 500, 'Inside X must remain unchanged');
  assert.equal(clampY(400), 400, 'Inside Y must remain unchanged');
}

