import { useState } from 'react';
import type { FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export interface PlanetSearchHit {
  name: string;
  pl_rade?: number;
  pl_bmasse?: number;
  pl_orbper?: number;
  pl_eqt?: number;
  st_teff?: number;
  st_rad?: number;
}

interface PlanetSearchFormProps {
  onSelect?: (planetName: string) => void;
  autoSearch?: string | null;
}

export function PlanetSearchForm({ onSelect, autoSearch = null }: PlanetSearchFormProps) {
  const [query, setQuery] = useState(autoSearch ?? '');
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(autoSearch);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: submittedQuery ? api.keys.planets(submittedQuery) : ['planets', 'search', ''],
    queryFn: () => api.searchPlanets(submittedQuery ?? ''),
    enabled: submittedQuery !== null && submittedQuery.trim().length > 0,
  });

  const results: PlanetSearchHit[] = (data as PlanetSearchHit[] | undefined) ?? [];

  const handleSearch = (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSubmittedQuery(query.trim());
  };

  return (
    <form onSubmit={handleSearch} className="space-y-4" aria-label="Planet search">
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for a planet (e.g., WASP-39 b)..."
            className="input pr-10"
            aria-label="Planet name"
          />
          <button
            type="submit"
            className="absolute right-3 top-1/2 -translate-y-1/2 text-blue-600 hover:text-blue-800"
            aria-label="Search"
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="text-sm text-gray-400">…</span>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            )}
          </button>
        </div>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Searching the NASA Exoplanet Archive…</p>}
      {isError && (
        <p className="text-sm text-red-600" role="alert">
          Search failed: {error instanceof Error ? error.message : 'unknown error'}
        </p>
      )}

      {!isLoading && !isError && submittedQuery && results.length === 0 && (
        <p className="text-sm text-gray-500">No planets found for “{submittedQuery}”.</p>
      )}

      {results.length > 0 && (
        <div className="mt-4 space-y-2 max-h-60 overflow-y-auto">
          {results.map((planet) => (
            <button
              key={planet.name}
              type="button"
              className="w-full text-left p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              onClick={() => onSelect?.(planet.name)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">{planet.name}</h4>
                  <p className="text-sm text-gray-500 mt-1">
                    {planet.pl_rade != null ? `Rp: ${planet.pl_rade} R⊕` : 'Rp: —'}
                    {' • '}
                    {planet.pl_bmasse != null ? `Mp: ${planet.pl_bmasse} M⊕` : 'Mp: —'}
                    {' • '}
                    {planet.pl_eqt != null ? `Teq: ${planet.pl_eqt} K` : 'Teq: —'}
                  </p>
                </div>
                <svg
                  className="w-5 h-5 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </button>
          ))}
        </div>
      )}
    </form>
  );
}

export default PlanetSearchForm;
