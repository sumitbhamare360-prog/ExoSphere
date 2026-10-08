import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import { PlanetSearchForm } from '../components/PlanetSearchForm';

type Preset = 'fast' | 'standard';

const PRESETS: Record<Preset, { n_live: number; dlogz: number; label: string }> = {
  fast: { n_live: 100, dlogz: 0.5, label: 'Fast (n_live=100, dlogz=0.5)' },
  standard: { n_live: 500, dlogz: 0.01, label: 'Standard (n_live=500, dlogz=0.01)' },
};

export function AnalyzePage() {
  const queryClient = useQueryClient();
  const [planetName, setPlanetName] = useState<string | null>(null);
  const [observationId, setObservationId] = useState<string | null>(null);
  const [preset, setPreset] = useState<Preset>('fast');
  const [seed, setSeed] = useState(42);
  const [runDetection, setRunDetection] = useState(true);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  const planetQuery = useQuery({
    queryKey: planetName ? api.keys.planet(planetName) : ['planets', 'detail', ''],
    queryFn: () => api.getPlanet(planetName ?? ''),
    enabled: planetName !== null,
  });

  const createMutation = useMutation({
    mutationFn: api.createAnalysis,
    onSuccess: (data) => {
      setAnalysisId(data.analysis_id);
      void queryClient.invalidateQueries({ queryKey: ['analyses'] });
    },
  });

  const statusQuery = useQuery({
    queryKey: analysisId ? api.keys.analysisStatus(analysisId) : ['analyses', 'status', ''],
    queryFn: () => api.getAnalysisStatus(analysisId ?? ''),
    enabled: analysisId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as { status?: string } | undefined;
      return data && (data.status === 'completed' || data.status === 'failed') ? false : 2000;
    },
  });

  const handleRun = () => {
    if (!planetName) return;
    createMutation.mutate({
      planet_name: planetName,
      observation_id: observationId ?? undefined,
      n_live: PRESETS[preset].n_live,
      dlogz: PRESETS[preset].dlogz,
      seed,
      run_ml: true,
      run_detection: runDetection,
      run_quality: true,
      run_preprocess: true,
      fixed_params: {},
    });
  };

  const status = statusQuery.data;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analyze</h1>
        <p className="text-gray-600 mt-1">Create a new atmospheric retrieval analysis</p>
      </div>

      <div className="card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">1. Find a planet</h2>
        <PlanetSearchForm onSelect={(name) => { setPlanetName(name); setObservationId(null); setAnalysisId(null); }} />
        {planetName && (
          <p className="mt-3 text-sm text-gray-700">
            Selected planet: <strong>{planetName}</strong>
          </p>
        )}
      </div>

      {planetQuery.isLoading && <p className="text-sm text-gray-500">Loading planet details…</p>}
      {planetQuery.isError && (
        <p className="text-sm text-red-600" role="alert">
          Failed to load planet: {planetQuery.error instanceof Error ? planetQuery.error.message : 'unknown error'}
        </p>
      )}

      {planetQuery.data && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-1">2. Choose a data source</h2>
          <p className="text-sm text-gray-500 mb-4">
            {planetQuery.data.observations.length} observation(s) available for {planetQuery.data.name}.
          </p>
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {planetQuery.data.observations.map((obs) => (
              <button
                key={obs.observation_id}
                type="button"
                onClick={() => setObservationId(obs.observation_id)}
                className={`w-full text-left p-4 rounded-lg border transition-colors ${
                  observationId === obs.observation_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:bg-gray-50'
                }`}
                aria-pressed={observationId === obs.observation_id}
              >
                <div className="font-medium text-gray-900">{obs.instrument}</div>
                <div className="text-sm text-gray-500">
                  {obs.observation_id} • {obs.num_points} points • {obs.reference}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {planetName && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">3. Options and run</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            <div>
              <label htmlFor="preset" className="label">Retrieval settings</label>
              <select
                id="preset"
                className="input"
                value={preset}
                onChange={(e) => setPreset(e.target.value as Preset)}
              >
                <option value="fast">{PRESETS.fast.label}</option>
                <option value="standard">{PRESETS.standard.label}</option>
              </select>
            </div>
            <div>
              <label htmlFor="seed" className="label">Seed</label>
              <input
                id="seed"
                type="number"
                className="input"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </div>
            <div className="flex items-end pb-2">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={runDetection}
                  onChange={(e) => setRunDetection(e.target.checked)}
                  className="rounded border-gray-300"
                />
                Run detection analysis
              </label>
            </div>
          </div>
          <button className="btn-primary" onClick={handleRun} disabled={createMutation.isPending}>
            {createMutation.isPending ? 'Starting…' : 'Run analysis'}
          </button>
          {createMutation.isError && (
            <p className="mt-2 text-sm text-red-600" role="alert">
              Failed to start: {createMutation.error instanceof Error ? createMutation.error.message : 'unknown error'}
            </p>
          )}
        </div>
      )}

      {analysisId && (
        <div className="card p-6" aria-live="polite">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Job progress</h2>
          <p className="text-sm text-gray-600">
            Analysis <code>{analysisId}</code>
            {status ? ` — ${status.status} (${status.stage}, ${Math.round(status.progress * 100)}%)` : ' — starting…'}
          </p>
          {status?.status === 'failed' && (
            <p className="mt-2 text-sm text-red-600" role="alert">
              Failed{status.error_message ? `: ${status.error_message}` : ''}.
            </p>
          )}
          {status?.status === 'completed' && (
            <Link to={`/twin/${analysisId}`} className="mt-3 inline-block btn-primary">
              Open 3D digital twin
            </Link>
          )}
        </div>
      )}
    </div>
  );
}

export default AnalyzePage;
