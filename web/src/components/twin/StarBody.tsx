import * as THREE from 'three';
import { useMemo } from 'react';

/** Procedural radial glow sprite (generated, not a photo). */
function makeGlowTexture(): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255, 244, 214, 1)');
  gradient.addColorStop(0.35, 'rgba(255, 236, 190, 0.55)');
  gradient.addColorStop(1, 'rgba(255, 236, 190, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

interface StarBodyProps {
  radius: number;
  /** sRGB tint from the blackbody approximation. */
  color: [number, number, number];
}

/** Host star: emissive sphere sized by stellar radius, plus a soft glow. */
export function StarBody({ radius, color }: StarBodyProps) {
  const glow = useMemo(() => makeGlowTexture(), []);
  const tint = useMemo(() => new THREE.Color(color[0], color[1], color[2]), [color]);
  return (
    <group>
      <mesh>
        <sphereGeometry args={[radius, 64, 64]} />
        <meshBasicMaterial color={tint} />
      </mesh>
      <sprite scale={[radius * 4.2, radius * 4.2, 1]}>
        <spriteMaterial map={glow} transparent opacity={0.85} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
    </group>
  );
}
