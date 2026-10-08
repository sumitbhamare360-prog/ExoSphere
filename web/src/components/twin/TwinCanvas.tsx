import { Suspense, useEffect, useMemo, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { StarBody } from './StarBody';
import { OrbitSystem } from './OrbitSystem';

export type CameraPreset = 'orbit' | 'transit';

interface TwinCanvasProps {
  starRadiusScene: number;
  starColor: [number, number, number];
  semiMajorScene: number;
  eccentricity: number;
  inclinationDeg: number;
  planetRadiusScene: number;
  shellThicknessScene: number;
  atmosphereColor: string;
  atmosphereOpacity: number;
  cloudDeckAltitude: number;
  cloudOpacity: number;
  cloudConstrained: boolean;
  phase: number;
  onPhaseChange: (phase: number) => void;
  playing: boolean;
  speed: number;
  cameraPreset: CameraPreset;
}

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

/** Moves the camera when the preset changes. */
function CameraRig({
  preset,
  semiMajorScene,
  starRadiusScene,
}: {
  preset: CameraPreset;
  semiMajorScene: number;
  starRadiusScene: number;
}) {
  const { camera, controls } = useThree((s) => ({ camera: s.camera, controls: s.controls }));
  useEffect(() => {
    if (preset === 'transit') {
      // Observer's line of sight: on +z, outside the orbit, looking at the star.
      camera.position.set(0, semiMajorScene * 0.06, semiMajorScene + starRadiusScene * 6);
    } else {
      const d = semiMajorScene * 1.6;
      camera.position.set(d * 0.8, d * 0.65, d * 0.9);
    }
    camera.lookAt(0, 0, 0);
    const orbit = controls as unknown as { target?: THREE.Vector3; update?: () => void } | null;
    if (orbit?.target) {
      orbit.target.set(0, 0, 0);
      orbit.update?.();
    }
  }, [preset, semiMajorScene, starRadiusScene, camera, controls]);
  return null;
}

function Scene(props: TwinCanvasProps) {
  return (
    <>
      <CameraRig
        preset={props.cameraPreset}
        semiMajorScene={props.semiMajorScene}
        starRadiusScene={props.starRadiusScene}
      />
      {/* Starlight: a point light at the star lights the planet's dayside and
          leaves a visible day/night terminator. Faint ambient keeps the night
          side barely visible. */}
      <pointLight position={[0, 0, 0]} intensity={2500} decay={0} color="#fff5e0" />
      <ambientLight intensity={0.12} />
      <StarBody radius={props.starRadiusScene} color={props.starColor} />
      <OrbitSystem
        semiMajorScene={props.semiMajorScene}
        eccentricity={props.eccentricity}
        inclinationDeg={props.inclinationDeg}
        planetRadiusScene={props.planetRadiusScene}
        shellThicknessScene={props.shellThicknessScene}
        atmosphereColor={props.atmosphereColor}
        atmosphereOpacity={props.atmosphereOpacity}
        cloudDeckAltitude={props.cloudDeckAltitude}
        cloudOpacity={props.cloudOpacity}
        cloudConstrained={props.cloudConstrained}
        phase={props.phase}
        onPhaseChange={props.onPhaseChange}
        playing={props.playing}
        speed={props.speed}
      />
      <OrbitControls makeDefault enableDamping={false} minDistance={props.starRadiusScene} maxDistance={props.semiMajorScene * 8} />
    </>
  );
}

export function TwinCanvas(props: TwinCanvasProps) {
  const [ok] = useState(webglAvailable);
  const dpr = useMemo(() => (typeof window !== 'undefined' ? Math.min(window.devicePixelRatio, 2) : 1), []);
  if (!ok) {
    return (
      <div className="flex h-full items-center justify-center bg-gray-900 text-sm text-gray-300" role="alert">
        WebGL is unavailable in this browser, so the 3D twin cannot render.
      </div>
    );
  }
  return (
    <Canvas
      dpr={dpr}
      gl={{ antialias: true, alpha: false }}
      camera={{ fov: 45, near: 0.1, far: 100000, position: [60, 45, 60] }}
      style={{ background: '#05070d' }}
    >
      <color attach="background" args={['#05070d']} />
      <Suspense fallback={null}>
        <Scene {...props} />
      </Suspense>
    </Canvas>
  );
}

export default TwinCanvas;
