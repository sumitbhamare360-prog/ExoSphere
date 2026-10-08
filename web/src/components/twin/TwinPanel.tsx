import type { TwinMoleculeContribution, TwinParameter } from '../../lib/api';

const SOURCE_STYLES: Record<string, string> = {
  measured: 'bg-blue-100 text-blue-800',
  inferred: 'bg-green-100 text-green-800',
  derived: 'bg-purple-100 text-purple-800',
  assumed: 'bg-amber-100 text-amber-800',
};

function SourceBadge({ source }: { source: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${SOURCE_STYLES[source] ?? 'bg-gray-100 text-gray-800'}`}
      title={`Source: ${source}`}
    >
      {source}
    </span>
  );
}

interface TwinPanelProps {
  parameters: TwinParameter[];
  molecules: TwinMoleculeContribution[];
  selectedMolecule: string | null;
  onSelectMolecule: (molecule: string | null) => void;
  moleculeColors: Record<string, string>;
}

function formatValue(param: TwinParameter): string {
  if (param.value === null || param.value === undefined) return '—';
  const v = param.value;
  const abs = Math.abs(v);
  const num = abs !== 0 && (abs >= 10000 || abs < 0.001) ? v.toExponential(2) : `${Math.round(v * 1000) / 1000}`;
  let text = `${num}${param.unit ? ` ${param.unit}` : ''}`;
  if (param.ci_68) {
    text += ` [${param.ci_68[0]}, ${param.ci_68[1]}]`;
  }
  return text;
}

export function TwinPanel({ parameters, molecules, selectedMolecule, onSelectMolecule, moleculeColors }: TwinPanelProps) {
  return (
    <div className="flex flex-col gap-4">
      <section aria-label="Twin parameters">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Parameters</h3>
        <ul className="divide-y divide-gray-100 rounded-lg border border-gray-200 bg-white">
          {parameters.map((p) => (
            <li key={p.name} className="px-3 py-2" title={p.note ?? undefined}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-gray-700">{p.label}</span>
                <SourceBadge source={p.source} />
              </div>
              <div className="text-sm font-mono text-gray-900">{formatValue(p)}</div>
            </li>
          ))}
        </ul>
      </section>

      <section aria-label="Modelled molecular contributions">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Molecules (model-derived)</h3>
        <ul className="space-y-2">
          {molecules.map((m) => {
            const selected = selectedMolecule === m.molecule;
            return (
              <li key={m.molecule}>
                <button
                  type="button"
                  onClick={() => onSelectMolecule(selected ? null : m.molecule)}
                  aria-pressed={selected}
                  className={`w-full text-left rounded-lg border p-3 transition-colors ${
                    selected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:bg-gray-50'
                  }`}
                  title={m.note}
                >
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-2 text-sm font-medium text-gray-900">
                      <span
                        className="inline-block h-3 w-3 rounded-full"
                        style={{ backgroundColor: moleculeColors[m.molecule] ?? '#888' }}
                        aria-hidden="true"
                      />
                      {m.molecule}
                    </span>
                    <span className="text-sm font-mono text-gray-900">
                      {(m.contribution_fraction * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-gray-200" aria-hidden="true">
                    <div
                      className="h-1.5 rounded-full"
                      style={{
                        width: `${Math.round(m.contribution_fraction * 100)}%`,
                        backgroundColor: moleculeColors[m.molecule] ?? '#888',
                      }}
                    />
                  </div>
                  <p className="mt-1 text-xs text-gray-500">
                    Modelled contribution to the spectral signal: {(m.contribution_fraction * 100).toFixed(1)}%
                    (model-derived)
                    {!m.in_band && ' • band outside spectrum coverage'}
                  </p>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <section aria-label="Legend" className="rounded-lg border border-gray-200 bg-white p-3">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Legend</h3>
        <ul className="space-y-1 text-xs text-gray-600">
          <li><span className="font-medium text-blue-800">measured</span> — catalog / data value</li>
          <li><span className="font-medium text-green-800">inferred</span> — retrieval posterior median (+ 68% CI)</li>
          <li><span className="font-medium text-purple-800">derived</span> — computed from other values</li>
          <li><span className="font-medium text-amber-800">assumed</span> — default used; data missing</li>
        </ul>
        <p className="mt-2 text-xs text-gray-500">
          Scientific visualization from measured and model-derived parameters. Not a photograph. No surface,
          cloud-map, or molecule-map claims.
        </p>
      </section>
    </div>
  );
}

export default TwinPanel;
