import { useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { orbitPath, orbitPoint, type OrbitPoint } from '../../lib/twinGeometry';
import { PlanetBody } from './PlanetBody';
import { AtmosphereShell } from './AtmosphereShell';
import { CloudLayer } from './CloudLayer';

export interface OrbitSystemProps {
  semiMajorScene: number;
  eccentricity: number;
  inclinationDeg: number;
  planetRadiusScene: number;
  shellThicknessScene: number;
  /** Uniform atmosphere tint (molecule highlight). */
  atmosphereColor: string;
  /** Uniform atmosphere opacity (scaled by modelled contribution). */
  atmosphereOpacity: number;
  cloudDeckAltitude: number;
  cloudOpacity: number;
  cloudConstrained: boolean;
  /** Orbital phase in [0, 1); animated externally when playing. */
  phase: number;
  onPhaseChange?: (phase: number) => void;
  playing: boolean;
  /** Orbits per second at speed 1. */
  speed: number;
}

function pointsToLineGeometry(points: OrbitPoint[]): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(points.length * 3);
  points.forEach((p, i) => {
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
  });
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return geometry;
}

export function OrbitSystem({
  semiMajorScene,
  eccentricity,
  inclinationDeg,
  planetRadiusScene,
  shellThicknessScene,
  atmosphereColor,
  atmosphereOpacity,
  cloudDeckAltitude,
  cloudOpacity,
  cloudConstrained,
  phase,
  onPhaseChange,
  playing,
  speed,
}: OrbitSystemProps) {
  const lineGeometry = useMemo(
    () => pointsToLineGeometry(orbitPath(semiMajorScene, eccentricity, inclinationDeg)),
    [semiMajorScene, eccentricity, inclinationDeg]
  );

  // Advance the orbital phase on each frame while playing.
  useFrame((_, delta) => {
    if (!playing || !onPhaseChange) return;
    const next = (phase + delta * 0.05 * speed) % 1;
    onPhaseChange(next);
  });

  const pos = orbitPoint(phase, semiMajorScene, eccentricity, inclinationDeg);

  return (
    <group>
      {/* Orbit path */}
      <lineLoop geometry={lineGeometry}>
        <lineBasicMaterial color="#9ca3af" transparent opacity={0.6} />
      </lineLoop>
      {/* Planet stack (planet + uniform atmosphere shell + cloud deck) */}
      <group position={[pos.x, pos.y, pos.z]}>
        <PlanetBody radius={planetRadiusScene} />
        <AtmosphereShell
          planetRadius={planetRadiusScene}
          shellThickness={shellThicknessScene}
          color={atmosphereColor}
          opacity={atmosphereOpacity}
        />
        <CloudLayer
          planetRadius={planetRadiusScene}
          deckAltitude={cloudDeckAltitude}
          opacity={cloudOpacity}
          constrained={cloudConstrained}
        />
      </group>
    </group>
  );
}
