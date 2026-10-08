import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type TwinParameters } from '../../lib/api';
import { useExoStore } from '../../store/useExoStore';
import {
  MOLECULE_COLORS,
  STAR_SCENE_RADIUS,
  VISUAL_EXAGGERATION,
  ATMOSPHERE_SHELL_SCALE_HEIGHTS,
  TIDAL_LOCK_NOTE,
  cloudVisual,
  isTransiting,
  orbitPoint,
  planetSceneRadius,
  semiMajorAxisScene,
  shellThicknessScene,
  starColor,
  transitDepth,
} from '../../lib/twinGeometry';
import { TwinPanel } from './TwinPanel';
import type { CameraPreset } from './TwinCanvas';
import wasp39bMock from '../../mock/wasp39b-twin.json';
import wasp121bMock from '../../mock/wasp121b-twin.json';

const TwinCanvas = lazy(() => import('./TwinCanvas'));

export type TwinMockKey = 'wasp39b' | 'wasp121b';

const MOCKS: Record<TwinMockKey, TwinParameters> = {
  wasp39b: wasp39bMock as TwinParameters,
  wasp121b: wasp121bMock as TwinParameters,
};

function paramByName(twin: TwinParameters, name: string) {
  const found = twin.parameters.find((p) => p.name === name);
  if (!found) throw new Error(`twin parameter ${name} missing`);
  return found;
}

/** Minimal scientific-report actions for a real (non-mock) analysis. */
function ReportButtons({ analysisId }: { analysisId: string }) {
  const [state, setState] = useState<'idle' | 'building' | 'ready' | 'error'>('idle');
  const [message, setMessage] = useState<string | null>(null);

  const generate = async () => {
    setState('building');
    setMessage(null);
    try {
      await api.generateReport(analysisId, 'html');
      setState('ready');
    } catch (error) {
      setState('error');
      setMessage(error instanceof Error ? error.message : 'Report build failed');
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Scientific report">
      <button type="button" className="btn-secondary" onClick={generate} disabled={state === 'building'}>
        {state === 'building' ? 'Generating report…' : 'Generate report'}
      </button>
      {(state === 'ready') && (
        <a className="btn-secondary" href={api.reportDownloadUrl(analysisId, 'html')} download>
          Download
        </a>
      )}
      {state === 'error' && message && (
        <span className="text-sm text-red-600" role="alert">{message}</span>
      )}
    </div>
  );
}

interface TwinViewProps {
  /** Real analysis id (fetched from the API). */
  analysisId?: string | null;
  /** Mock dataset key (UI development without a backend). */
  mockKey?: TwinMockKey | null;
}

export function TwinView({ analysisId = null, mockKey = null }: TwinViewProps) {
  const selectedMolecule = useExoStore((s) => s.selectedMolecule);
  const setSelectedMolecule = useExoStore((s) => s.setSelectedMolecule);
  const setMoleculeBands = useExoStore((s) => s.setMoleculeBands);

  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [phase, setPhase] = useState(0);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('orbit');

  const useMock = mockKey !== null || import.meta.env.VITE_USE_MOCK === 'true';
  const mockTwin: TwinParameters | null = mockKey ? MOCKS[mockKey] : MOCKS.wasp39b;

  const twinQuery = useQuery({
    queryKey: analysisId ? api.keys.twin(analysisId) : ['analyses', 'twin', 'none'],
    queryFn: () => api.getTwin(analysisId ?? ''),
    enabled: !useMock && analysisId !== null,
  });

  const bandsQuery = useQuery({
    queryKey: api.keys.moleculeBands,
    queryFn: api.getMoleculeBands,
    enabled: !useMock,
    staleTime: Infinity,
  });

  // Share band windows with the spectrum viewer through the store.
  useEffect(() => {
    if (useMock) {
      setMoleculeBands({
        H2O: [{ start: 0.93, end: 0.98 }, { start: 1.1, end: 1.2 }, { start: 1.3, end: 1.5 }, { start: 1.8, end: 2.0 }, { start: 2.6, end: 3.1 }],
        CO2: [{ start: 2.0, end: 2.1 }, { start: 2.7, end: 2.8 }, { start: 4.2, end: 4.4 }],
        CO: [{ start: 2.3, end: 2.4 }, { start: 4.5, end: 4.9 }],
        CH4: [{ start: 1.6, end: 1.8 }, { start: 2.2, end: 2.4 }, { start: 3.2, end: 3.5 }],
        SO2: [{ start: 3.9, end: 4.2 }],
      });
      return;
    }
    const bands = bandsQuery.data?.molecules;
    if (!bands) return;
    const mapped: Record<string, Array<{ start: number; end: number }>> = {};
    for (const [mol, wins] of Object.entries(bands)) {
      mapped[mol] = wins.map(([start, end]) => ({ start, end }));
    }
    setMoleculeBands(mapped);
  }, [useMock, bandsQuery.data, setMoleculeBands]);

  const twin: TwinParameters | null = useMock ? mockTwin : (twinQuery.data ?? null);

  const scene = useMemo(() => {
    if (!twin) return null;
    const rp = paramByName(twin, 'planet_radius').value ?? 1.0;
    const rs = paramByName(twin, 'stellar_radius').value ?? 1.0;
    const teff = paramByName(twin, 'stellar_teff').value ?? 5778;
    const aRs = paramByName(twin, 'semi_major_axis_stellar_radii').value;
    const incl = paramByName(twin, 'inclination').value ?? 90;
    const eccP = paramByName(twin, 'eccentricity').value;
    const ecc = eccP ?? 0;
    const hKm = paramByName(twin, 'scale_height').value ?? 100;
    const cloud = paramByName(twin, 'cloud_top_pressure');
    const cloudLogP = cloud.value ?? 2.0;
    const cloudConstrained = cloud.constrained ?? false;

    const rpRs = (rp * 7.1492e7) / (rs * 6.957e8);
    const planetR = planetSceneRadius(rpRs);
    const aScene = semiMajorAxisScene(aRs) ?? planetR * 12;
    const shell = shellThicknessScene(planetR, hKm * 1000, rp * 7.1492e7);
    const cloudVis = cloudVisual(cloudLogP, cloudConstrained);
    const mol = twin.molecules.find((m) => m.molecule === selectedMolecule);
    const contrib = mol ? mol.contribution_fraction : 0;
    return {
      rpRs,
      planetR,
      aScene,
      incl,
      ecc,
      shell,
      cloudVis,
      cloudLogP,
      starColor: starColor(teff),
      atmosphereColor: selectedMolecule ? (MOLECULE_COLORS[selectedMolecule] ?? '#93c5fd') : '#9fc3e8',
      atmosphereOpacity: selectedMolecule ? 0.08 + 0.6 * contrib : 0.07,
      depth: transitDepth(rpRs),
      contrib,
    };
  }, [twin, selectedMolecule]);

  // Mini light curve synced to the orbital phase (flat-bottom approximation).
  const lightCurve = useMemo(() => {
    if (!scene) return [];
    const pts: Array<{ phase: number; flux: number }> = [];
    for (let i = 0; i <= 64; i++) {
      const p = i / 64;
      const pos = orbitPoint(p, scene.aScene, scene.ecc, scene.incl);
      const sep = Math.hypot(pos.x, pos.y);
      const inTransit = pos.z > 0 && sep < STAR_SCENE_RADIUS + scene.planetR;
      pts.push({ phase: p, flux: inTransit ? 1 - scene.depth : 1 });
    }
    return pts;
  }, [scene]);

  if (!useMock && twinQuery.isLoading) {
    return <p className="text-sm text-gray-500">Loading digital twin…</p>;
  }
  if (!useMock && twinQuery.isError) {
    return (
      <p className="text-sm text-red-600" role="alert">
        Failed to load twin parameters:{' '}
        {twinQuery.error instanceof Error ? twinQuery.error.message : 'unknown error'}
      </p>
    );
  }
  if (!twin || !scene) {
    return <p className="text-sm text-gray-500">No twin data available.</p>;
  }

  const curvePath =
    lightCurve.length > 0
      ? `M ${lightCurve.map((pt) => `${(pt.phase * 100).toFixed(1)},${(18 - pt.flux * 14).toFixed(1)}`).join(' L ')}`
      : '';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 space-y-3">
        <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Twin controls">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setPlaying((p) => !p)}
            aria-pressed={playing}
          >
            {playing ? 'Pause orbit' : 'Play orbit'}
          </button>
          <label className="flex items-center gap-2 text-sm text-gray-700">
            Speed
            <input
              type="range"
              min={0.1}
              max={4}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              aria-label="Orbit animation speed"
            />
            <span className="font-mono text-xs">{speed.toFixed(1)}×</span>
          </label>
          <div className="flex gap-1" role="group" aria-label="Camera preset">
            {(['orbit', 'transit'] as const).map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => setCameraPreset(preset)}
                aria-pressed={cameraPreset === preset}
                className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                  cameraPreset === preset
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                {preset === 'orbit' ? 'Orbit view' : 'Transit view'}
              </button>
            ))}
          </div>
          <span className="ml-auto font-mono text-xs text-gray-500" aria-live="polite">
            phase {phase.toFixed(3)}
            {isTransiting(
              orbitPoint(phase, scene.aScene, scene.ecc, scene.incl),
              STAR_SCENE_RADIUS,
              scene.planetR
            )
              ? ' • in transit'
              : ''}
          </span>
        </div>

        <div className="rounded-xl overflow-hidden border border-gray-800" style={{ height: 480 }}>
          <Suspense fallback={<p className="p-4 text-sm text-gray-400">Loading 3D view…</p>}>
            <TwinCanvas
              starRadiusScene={STAR_SCENE_RADIUS}
              starColor={scene.starColor}
              semiMajorScene={scene.aScene}
              eccentricity={scene.ecc}
              inclinationDeg={scene.incl}
              planetRadiusScene={scene.planetR}
              shellThicknessScene={scene.shell}
              atmosphereColor={scene.atmosphereColor}
              atmosphereOpacity={scene.atmosphereOpacity}
              cloudDeckAltitude={scene.cloudVis.altitudeFraction * scene.shell}
              cloudOpacity={scene.cloudVis.opacity}
              cloudConstrained={scene.cloudVis.constrained}
              phase={phase}
              onPhaseChange={setPhase}
              playing={playing}
              speed={speed}
              cameraPreset={cameraPreset}
            />
          </Suspense>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-3">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-sm font-semibold text-gray-900">Light curve (model indicator)</h3>
            <span className="font-mono text-xs text-gray-500">depth {(scene.depth * 1e6).toFixed(0)} ppm</span>
          </div>
          <svg viewBox="0 0 100 20" className="w-full h-16" role="img" aria-label="Model light curve synced to orbital phase">
            <path d={curvePath} fill="none" stroke="#2563eb" strokeWidth="0.8" />
            <line
              x1={phase * 100}
              y1="0"
              x2={phase * 100}
              y2="20"
              stroke="#ef4444"
              strokeWidth="0.6"
            />
          </svg>
          <p className="text-xs text-gray-500">Flat-bottom approximation from (Rp/Rs)²; synced to the orbit animation.</p>
        </div>

        <p className="text-xs text-gray-500">
          Atmosphere shell spans {ATMOSPHERE_SHELL_SCALE_HEIGHTS} scale heights, visually exaggerated ×
          {VISUAL_EXAGGERATION} (not to scale). {TIDAL_LOCK_NOTE}
        </p>
      </div>

      <div className="lg:col-span-1">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">{twin.planet_name} — digital twin</h2>
          {useMock && (
            <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-xs font-medium">
              mock data
            </span>
          )}
        </div>
        {!useMock && analysisId && (
          <div className="mb-3">
            <ReportButtons analysisId={analysisId} />
          </div>
        )}
        <TwinPanel
          parameters={twin.parameters}
          molecules={twin.molecules}
          selectedMolecule={selectedMolecule}
          onSelectMolecule={setSelectedMolecule}
          moleculeColors={MOLECULE_COLORS}
        />
      </div>
    </div>
  );
}

export default TwinView;
