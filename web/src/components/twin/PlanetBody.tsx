import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * Neutral planet shading: a procedurally generated vertical gradient
 * (canvas texture, uniform in longitude). No continents, oceans, mountains,
 * photo textures, or any geographic features — AGENTS.md rule 4.
 */
export function makeGradientTexture(base: string, top: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 4;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  const gradient = ctx.createLinearGradient(0, 0, 0, 256);
  gradient.addColorStop(0, top);
  gradient.addColorStop(0.5, base);
  gradient.addColorStop(1, top);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 4, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

interface PlanetBodyProps {
  radius: number;
}

/** Matte planet sphere. Lit by the scene lights, so the star-facing side is bright. */
export function PlanetBody({ radius }: PlanetBodyProps) {
  const texture = useMemo(() => makeGradientTexture('#5b6b82', '#8fa0b5'), []);
  return (
    <mesh>
      <sphereGeometry args={[radius, 64, 64]} />
      <meshStandardMaterial map={texture} roughness={1} metalness={0} />
    </mesh>
  );
}
