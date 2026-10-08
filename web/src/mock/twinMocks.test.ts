import { describe, expect, it } from 'vitest';
import type { TwinParameters } from '../lib/api';
import { planetSceneRadius, semiMajorAxisScene, transitDepth } from '../lib/twinGeometry';
import wasp39b from './wasp39b-twin.json';
import wasp121b from './wasp121b-twin.json';

const REQUIRED_PARAMS = [
  'planet_radius',
  'stellar_radius',
  'stellar_teff',
  'semi_major_axis_stellar_radii',
  'inclination',
  'eccentricity',
  'scale_height',
  'cloud_top_pressure',
];

const SOURCES = new Set(['measured', 'inferred', 'derived', 'assumed']);
const MOLECULES = ['H2O', 'CO2', 'CO', 'CH4', 'SO2'];

/**
 * Headless render-path verification for both demo planets (Phase 8 §14):
 * each mock twin dataset must satisfy the API contract AND map to sane
 * scene values through the same geometry functions TwinView uses.
 */
describe.each([
  ['WASP-39 b', wasp39b as TwinParameters],
  ['WASP-121 b', wasp121b as TwinParameters],
])('twin mock dataset: %s', (_name, twin) => {
  it('declares every parameter the scene needs, with valid source tags', () => {
    const names = new Set(twin.parameters.map((p) => p.name));
    for (const required of REQUIRED_PARAMS) {
      expect(names.has(required), `missing twin parameter ${required}`).toBe(true);
    }
    for (const p of twin.parameters) {
      expect(SOURCES.has(p.source), `${p.name} has invalid source ${p.source}`).toBe(true);
      if (p.ci_68) expect(p.ci_68[1]).toBeGreaterThanOrEqual(p.ci_68[0]);
    }
  });

  it('covers all five V1 molecules with fractions in [0, 1]', () => {
    const names = twin.molecules.map((m) => m.molecule).sort();
    expect(names).toEqual([...MOLECULES].sort());
    let total = 0;
    for (const m of twin.molecules) {
      expect(m.contribution_fraction).toBeGreaterThanOrEqual(0);
      expect(m.contribution_fraction).toBeLessThanOrEqual(1);
      total += m.contribution_fraction;
    }
    // Fractions are per-molecule in-band shares (bands overlap), so the
    // sum may exceed 1 but must stay positive and bounded.
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThan(MOLECULES.length);
  });

  it('maps to finite, sane scene geometry', () => {
    const byName = new Map(twin.parameters.map((p) => [p.name, p.value]));
    const rpRs = 0.14; // representative hot-Jupiter ratio; scene scales linearly
    const planetR = planetSceneRadius(rpRs);
    const aScene = semiMajorAxisScene(byName.get('semi_major_axis_stellar_radii') ?? null);
    expect(planetR).toBeGreaterThan(0);
    expect(Number.isFinite(planetR)).toBe(true);
    expect(aScene).not.toBeNull();
    expect(aScene as number).toBeGreaterThan(planetR * 5);
    const depth = transitDepth(rpRs);
    expect(depth).toBeGreaterThan(0);
    expect(depth).toBeLessThan(0.1);
  });
});
