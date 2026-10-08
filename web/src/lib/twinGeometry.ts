/**
 * Pure scene-geometry mappings for the scientific digital twin.
 *
 * All functions here are deterministic and unit-tested
 * (`twinGeometry.test.ts`). They map API twin parameters (SI + catalog
 * units) onto three.js scene units. Visual choices are documented below
 * and labelled in the UI; nothing here invents science.
 *
 * Scene convention (three.js, y-up):
 * - The star sits at the origin. Observer (default/transit camera) looks
 *   from +z toward the origin.
 * - Lengths are in scene units where the stellar radius is STAR_SCENE_RADIUS.
 * - The orbit lies in the x-z plane for an edge-on orbit (i = 90 deg);
 *   lower inclinations tilt it about the x-axis.
 */

export const MOLECULE_COLORS: Record<string, string> = {
  H2O: '#3b82f6',
  CO2: '#ef4444',
  CO: '#f97316',
  CH4: '#8b5cf6',
  SO2: '#f59e0b',
};

/** Star radius in scene units. Everything else scales from Rp/Rs and a/Rs. */
export const STAR_SCENE_RADIUS = 10;

/**
 * Atmosphere shell thickness in scale heights. The shell outer edge sits
 * 4 pressure scale heights above the reference radius (e^-4 ~ 2% residual
 * pressure, a conventional "top of the modelled atmosphere").
 */
export const ATMOSPHERE_SHELL_SCALE_HEIGHTS = 4;

/**
 * Visual exaggeration applied ONLY to the atmosphere shell thickness
 * (and cloud altitude within it) so a ~0.1%-of-radius layer is visible.
 * The planet/star/orbit geometry itself is NOT exaggerated. Shown in the UI
 * next to the "not to scale" label.
 */
export const VISUAL_EXAGGERATION = 25;

/** Tidal locking is an assumption for hot Jupiters, not a measurement. */
export const TIDAL_LOCK_NOTE =
  'Assumed tidally locked (synchronous rotation): dayside permanently faces the star.';

/** Width of the 68% cloud-credible-interval (dex) below which the cloud is drawn solid. */
export const CLOUD_CONSTRAINED_CI_WIDTH_DEX = 2.0;

/**
 * Pressure scale height [m]: H = k_B * T / (mu * m_H * g).
 * Must match `twin.scale_height_m` on the backend.
 */
export function scaleHeightM(tK: number, mmwGmol: number, gravityMS2: number): number {
  const kB = 1.380649e-23;
  const mH = 1.6735575e-27;
  if (tK <= 0 || mmwGmol <= 0 || gravityMS2 <= 0) {
    throw new Error('scaleHeightM requires positive T, mmw and gravity');
  }
  return (kB * tK) / (mmwGmol * mH * gravityMS2);
}

/** Planet radius in scene units from the Rp/Rs ratio. */
export function planetSceneRadius(rpRs: number): number {
  if (rpRs <= 0) throw new Error('Rp/Rs must be > 0');
  return rpRs * STAR_SCENE_RADIUS;
}

/** Semi-major axis in scene units from a/Rs. Null when unknown. */
export function semiMajorAxisScene(aRs: number | null): number | null {
  if (aRs === null || !(aRs > 0)) return null;
  return aRs * STAR_SCENE_RADIUS;
}

/** Transit depth (Rp/Rs)^2 from the radius ratio. */
export function transitDepth(rpRs: number): number {
  return rpRs * rpRs;
}

/**
 * Approximate blackbody color as [r, g, b] in 0..1 for a stellar Teff in K.
 * Tanner Helland's approximation (public-domain algorithm), valid ~1000-40000 K;
 * inputs are clamped to that range. Used for the star sphere tint only.
 */
export function starColor(teffK: number): [number, number, number] {
  const t = Math.min(40000, Math.max(1000, teffK)) / 100;
  let r: number;
  let g: number;
  let b: number;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    b = 255;
  }
  const clamp = (v: number) => Math.min(255, Math.max(0, v)) / 255;
  return [clamp(r), clamp(g), clamp(b)];
}

/**
 * Solve Kepler's equation M = E - e*sin(E) by Newton iteration.
 * M in radians. Returns eccentric anomaly E in radians.
 */
export function solveKepler(meanAnomaly: number, e: number): number {
  const ecc = Math.min(Math.max(e, 0), 0.9);
  let eAnom = meanAnomaly;
  for (let i = 0; i < 12; i++) {
    eAnom -= (eAnom - ecc * Math.sin(eAnom) - meanAnomaly) / (1 - ecc * Math.cos(eAnom));
  }
  return eAnom;
}

export interface OrbitPoint {
  x: number;
  y: number;
  z: number;
}

/**
 * Planet position in scene units at orbital phase in [0, 1).
 * Phase 0 = inferior conjunction (transit center): the planet is at maximum
 * +z (between star and the default camera) on the midplane.
 */
export function orbitPoint(
  phase01: number,
  semiMajorScene: number,
  eccentricity: number,
  inclinationDeg: number
): OrbitPoint {
  const phase = ((phase01 % 1) + 1) % 1;
  const meanAnomaly = 2 * Math.PI * phase;
  const eccAnom = solveKepler(meanAnomaly, eccentricity);
  const e = Math.min(Math.max(eccentricity, 0), 0.9);
  const cosE = Math.cos(eccAnom);
  const sinE = Math.sin(eccAnom);
  // Position in the orbital plane, periapsis along +x, transit at +z.
  // Rotate by +90 deg so phase 0 sits at +z instead of +x.
  const xOrb = semiMajorScene * (cosE - e);
  const zOrb = semiMajorScene * Math.sqrt(Math.max(0, 1 - e * e)) * sinE;
  const theta = Math.PI / 2;
  const x = xOrb * Math.cos(theta) - zOrb * Math.sin(theta);
  const z0 = xOrb * Math.sin(theta) + zOrb * Math.cos(theta);
  // Tilt about the x-axis for inclination: i=90 -> orbit in x-z plane.
  const tilt = ((90 - inclinationDeg) * Math.PI) / 180;
  const y = -Math.sin(tilt) * z0;
  const z = Math.cos(tilt) * z0;
  return { x, y, z };
}

/** Sampled orbit path for the orbit line. */
export function orbitPath(
  semiMajorScene: number,
  eccentricity: number,
  inclinationDeg: number,
  segments = 128
): OrbitPoint[] {
  const points: OrbitPoint[] = [];
  for (let i = 0; i <= segments; i++) {
    points.push(orbitPoint(i / segments, semiMajorScene, eccentricity, inclinationDeg));
  }
  return points;
}

/**
 * Whether the planet is currently transiting (disc overlap on the sky plane).
 * Projected separation < Rs + Rp in scene units.
 */
export function isTransiting(
  pos: OrbitPoint,
  starRadiusScene: number,
  planetRadiusScene: number
): boolean {
  const sep = Math.hypot(pos.x, pos.y);
  return pos.z > 0 && sep < starRadiusScene + planetRadiusScene;
}

export interface CloudVisual {
  /** Fraction of the displayed shell thickness at which the cloud deck sits (0 = surface). */
  altitudeFraction: number;
  /** Shell opacity 0..1. */
  opacity: number;
  /** False when the retrieval left the cloud-top pressure unconstrained. */
  constrained: boolean;
}

/**
 * Map cloud-top pressure to a cloud-deck visual.
 *
 * Mapping (documented choice): altitudeFraction = (2 - logP) / 8, so a deep
 * deck (logP = 2, effectively clear) sits at the surface and a high deck
 * (logP = -6) at the top of the displayed shell. Opacity rises the higher
 * (lower-pressure) the deck: opacity = 0.15 + 0.55 * altitudeFraction.
 * `constrained` is passed through from the API (CI width < 2 dex);
 * unconstrained decks render hatched and labelled uncertain.
 */
export function cloudVisual(logPCloud: number, constrained: boolean): CloudVisual {
  const altitudeFraction = Math.min(1, Math.max(0, (2 - logPCloud) / 8));
  return {
    altitudeFraction,
    opacity: 0.15 + 0.55 * altitudeFraction,
    constrained,
  };
}

/** Displayed atmosphere shell thickness in scene units (exaggerated). */
export function shellThicknessScene(planetRadiusScene: number, scaleHeightM: number, planetRadiusM: number): number {
  const frac = (ATMOSPHERE_SHELL_SCALE_HEIGHTS * scaleHeightM) / planetRadiusM;
  return planetRadiusScene * frac * VISUAL_EXAGGERATION;
}
