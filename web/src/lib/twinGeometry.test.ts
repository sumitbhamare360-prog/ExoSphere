import { describe, expect, it } from 'vitest';
import {
  ATMOSPHERE_SHELL_SCALE_HEIGHTS,
  STAR_SCENE_RADIUS,
  VISUAL_EXAGGERATION,
  cloudVisual,
  isTransiting,
  orbitPath,
  orbitPoint,
  planetSceneRadius,
  scaleHeightM,
  semiMajorAxisScene,
  shellThicknessScene,
  solveKepler,
  starColor,
  transitDepth,
} from './twinGeometry';

describe('scaleHeightM', () => {
  it('matches H = kB*T/(mu*mH*g) for WASP-39 b-like values', () => {
    // T=1170 K, mu=2.35 g/mol, g=4.3 m/s^2 -> ~955 km
    const h = scaleHeightM(1170, 2.35, 4.3);
    expect(h).toBeCloseTo(955000, -3);
  });

  it('rejects non-positive inputs', () => {
    expect(() => scaleHeightM(0, 2.3, 10)).toThrow();
    expect(() => scaleHeightM(1000, -1, 10)).toThrow();
    expect(() => scaleHeightM(1000, 2.3, 0)).toThrow();
  });
});

describe('planetSceneRadius / semiMajorAxisScene', () => {
  it('scales Rp/Rs from the fixed stellar scene radius', () => {
    expect(planetSceneRadius(0.14)).toBeCloseTo(0.14 * STAR_SCENE_RADIUS);
  });

  it('rejects non-positive Rp/Rs', () => {
    expect(() => planetSceneRadius(0)).toThrow();
  });

  it('returns null semi-major axis when unknown', () => {
    expect(semiMajorAxisScene(null)).toBeNull();
    expect(semiMajorAxisScene(-3)).toBeNull();
    expect(semiMajorAxisScene(11.4)).toBeCloseTo(11.4 * STAR_SCENE_RADIUS);
  });
});

describe('transitDepth', () => {
  it('is (Rp/Rs)^2', () => {
    expect(transitDepth(0.146)).toBeCloseTo(0.146 ** 2);
  });
});

describe('starColor', () => {
  it('returns channel values in [0, 1] and reddens cool stars', () => {
    for (const teff of [1000, 3000, 5400, 5778, 10000, 40000]) {
      const [r, g, b] = starColor(teff);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(1);
    }
    const cool = starColor(3000);
    const hot = starColor(10000);
    expect(cool[0]).toBeGreaterThan(cool[2]);
    expect(hot[2]).toBeGreaterThan(hot[0]);
  });

  it('clamps out-of-range Teff instead of producing NaN', () => {
    for (const c of starColor(-500)) expect(c).not.toBeNaN();
    for (const c of starColor(1e9)) expect(c).not.toBeNaN();
  });
});

describe('solveKepler', () => {
  it('solves circular orbits exactly', () => {
    expect(solveKepler(1.234, 0)).toBeCloseTo(1.234, 10);
  });

  it('satisfies M = E - e*sin(E) for eccentric orbits', () => {
    for (const [m, e] of [[0.5, 0.3], [2.1, 0.6], [5.0, 0.8]] as Array<[number, number]>) {
      const eAnom = solveKepler(m, e);
      expect(eAnom - e * Math.sin(eAnom)).toBeCloseTo(m, 8);
    }
  });
});

describe('orbitPoint', () => {
  const a = 114; // 11.4 R* in scene units

  it('places phase 0 at transit (maximum +z, on the midplane)', () => {
    const p = orbitPoint(0, a, 0, 90);
    expect(p.z).toBeCloseTo(a, 6);
    expect(p.x).toBeCloseTo(0, 6);
    expect(p.y).toBeCloseTo(0, 6);
  });

  it('places phase 0.5 at secondary eclipse (maximum -z)', () => {
    const p = orbitPoint(0.5, a, 0, 90);
    expect(p.z).toBeCloseTo(-a, 6);
  });

  it('wraps phases into [0, 1)', () => {
    const p1 = orbitPoint(1.25, a, 0, 90);
    const p2 = orbitPoint(0.25, a, 0, 90);
    expect(p1.x).toBeCloseTo(p2.x, 10);
    expect(p1.z).toBeCloseTo(p2.z, 10);
  });

  it('keeps an inclined orbit off the midplane at transit', () => {
    const p = orbitPoint(0, a, 0, 80);
    expect(Math.abs(p.y)).toBeGreaterThan(0);
  });
});

describe('orbitPath', () => {
  it('returns a closed loop', () => {
    const path = orbitPath(100, 0, 90, 64);
    expect(path).toHaveLength(65);
    expect(path[0].x).toBeCloseTo(path[64].x, 10);
    expect(path[0].z).toBeCloseTo(path[64].z, 10);
  });
});

describe('isTransiting', () => {
  it('is true at transit center for edge-on orbits', () => {
    const p = orbitPoint(0, 114, 0, 90);
    expect(isTransiting(p, STAR_SCENE_RADIUS, 1.5)).toBe(true);
  });

  it('is false at secondary eclipse (behind the star)', () => {
    const p = orbitPoint(0.5, 114, 0, 90);
    expect(isTransiting(p, STAR_SCENE_RADIUS, 1.5)).toBe(false);
  });
});

describe('cloudVisual', () => {
  it('puts deep (clear) decks at the surface and high decks at the top', () => {
    expect(cloudVisual(2, true).altitudeFraction).toBe(0);
    expect(cloudVisual(-6, true).altitudeFraction).toBe(1);
  });

  it('clamps out-of-range pressures and raises opacity with altitude', () => {
    const deep = cloudVisual(10, true);
    const high = cloudVisual(-10, true);
    expect(deep.altitudeFraction).toBe(0);
    expect(high.altitudeFraction).toBe(1);
    expect(high.opacity).toBeGreaterThan(deep.opacity);
  });

  it('passes the constrained flag through', () => {
    expect(cloudVisual(-1, false).constrained).toBe(false);
    expect(cloudVisual(-1, true).constrained).toBe(true);
  });
});

describe('shellThicknessScene', () => {
  it('is 4 scale heights exaggerated', () => {
    const rpScene = 1.4;
    const rpM = 1.4 * 7.1492e7;
    const h = 955000;
    expect(shellThicknessScene(rpScene, h, rpM)).toBeCloseTo(
      rpScene * ((ATMOSPHERE_SHELL_SCALE_HEIGHTS * h) / rpM) * VISUAL_EXAGGERATION
    );
  });
});
