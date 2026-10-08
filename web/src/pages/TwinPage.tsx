import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { TwinView, type TwinMockKey } from '../components/twin/TwinView';

const DEMO_PLANETS: Array<{ key: TwinMockKey; name: string; note: string }> = [
  { key: 'wasp39b', name: 'WASP-39 b', note: 'Hot Saturn · benchmark target' },
  { key: 'wasp121b', name: 'WASP-121 b', note: 'Ultra-hot Jupiter · second demo planet' },
];

/**
 * Route wrapper for the 3D scientific digital twin.
 * - `/twin/:analysisId` renders the twin for a real backend analysis.
 * - `/twin` (or `/twin/demo`) renders the demo picker backed by mock twin
 *   datasets (clearly labelled "mock data" in the UI).
 */
export function TwinPage() {
  const { analysisId } = useParams<{ analysisId?: string }>();
  const [mockKey, setMockKey] = useState<TwinMockKey>('wasp39b');

  const isDemo = !analysisId || analysisId === 'demo';

  if (!isDemo) {
    return <TwinView analysisId={analysisId} />;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Demo planet picker">
        <span className="text-sm text-gray-600">Demo planet:</span>
        {DEMO_PLANETS.map((p) => (
          <button
            key={p.key}
            type="button"
            onClick={() => setMockKey(p.key)}
            aria-pressed={mockKey === p.key}
            title={p.note}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
              mockKey === p.key
                ? 'border-blue-500 bg-blue-50 text-blue-700'
                : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            {p.name}
          </button>
        ))}
        <Link to="/analyze" className="ml-auto text-sm text-blue-600 hover:text-blue-800">
          Run a real analysis →
        </Link>
      </div>
      <TwinView key={mockKey} mockKey={mockKey} />
    </div>
  );
}

export default TwinPage;
