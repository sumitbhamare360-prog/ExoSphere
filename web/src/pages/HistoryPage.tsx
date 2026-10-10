import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export function HistoryPage() {
  const historyQuery = useQuery({
    queryKey: api.keys.analyses(1, 20),
    queryFn: () => api.listAnalyses(1, 20),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">History</h1>
        <p className="text-gray-600 mt-1">Past analyses and their results.</p>
      </div>

      {historyQuery.isLoading && <p className="text-sm text-gray-500">Loading history…</p>}
      {historyQuery.isError && (
        <p className="text-sm text-red-600" role="alert">
          Failed to load history:{' '}
          {historyQuery.error instanceof Error ? historyQuery.error.message : 'unknown error'}
        </p>
      )}

      {historyQuery.data && historyQuery.data.items.length === 0 && (
        <div className="card p-6">
          <p className="text-sm text-gray-500">
            No analyses yet. Run an analysis from the Analyze page, then reopen it here.
          </p>
        </div>
      )}

      {historyQuery.data && historyQuery.data.items.length > 0 && (
        <div className="card divide-y divide-gray-100">
          {historyQuery.data.items.map((item) => (
            <div key={item.analysis_id} className="flex items-center justify-between gap-3 p-4">
              <div>
                <div className="font-medium text-gray-900">
                  {item.planet_name}{' '}
                  <span className="font-mono text-xs text-gray-500">{item.analysis_id}</span>
                </div>
                <div className="text-sm text-gray-500">
                  {item.status} ({item.stage}, {Math.round(item.progress * 100)}%)
                </div>
              </div>
              {item.status === 'completed' ? (
                <Link
                  to={`/twin/${item.analysis_id}`}
                  className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 hover:bg-gray-50"
                >
                  Open results
                </Link>
              ) : (
                <Link
                  to={`/twin/${item.analysis_id}`}
                  className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 hover:bg-gray-50"
                >
                  View status
                </Link>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
