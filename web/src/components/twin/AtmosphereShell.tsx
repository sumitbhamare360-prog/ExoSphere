import * as THREE from 'three';

interface AtmosphereShellProps {
  planetRadius: number;
  shellThickness: number;
  /** Uniform tint applied to the whole shell (molecule highlight). */
  color: string;
  /** Uniform opacity 0..1 (scaled by modelled contribution). */
  opacity: number;
}

/**
 * Translucent atmosphere shell of uniform color and opacity.
 * The glow is identical at every longitude/latitude: it represents bulk
 * atmospheric presence in the model, NEVER a geographic molecule map.
 */
export function AtmosphereShell({ planetRadius, shellThickness, color, opacity }: AtmosphereShellProps) {
  return (
    <mesh>
      <sphereGeometry args={[planetRadius + shellThickness, 64, 64]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} side={THREE.FrontSide} />
    </mesh>
  );
}
