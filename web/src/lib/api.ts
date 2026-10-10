const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  // Health
  health: () => request<{ status: string; version: string; database: string }>('/health'),

  // Planets
  searchPlanets: (query: string) => 
    request<Array<{
      name: string;
      pl_rade?: number;
      pl_bmasse?: number;
      pl_orbper?: number;
      pl_eqt?: number;
      st_teff?: number;
      st_rad?: number;
    }>>(`/planets/search?q=${encodeURIComponent(query)}`),

  getPlanet: (name: string) => 
    request<{
      name: string;
      pl_rade?: number;
      pl_bmasse?: number;
      pl_bmassprov?: string;
      pl_eqt?: number;
      pl_insol?: number;
      pl_orbper?: number;
      pl_orbsmax?: number;
      pl_orbeccen?: number;
      pl_orbincl?: number;
      pl_tranmid?: number;
      pl_ratror?: number;
      pl_ratdor?: number;
      st_rad?: number;
      st_teff?: number;
      st_mass?: number;
      st_logg?: number;
      st_met?: number;
      st_spectype?: string;
      surface_gravity_m_s2?: number;
      observations: Array<{
        observation_id: string;
        instrument: string;
        facility: string;
        spectrum_type: string;
        wavelength_min_um?: number;
        wavelength_max_um?: number;
        num_points: number;
        reference: string;
        bibcode: string;
      }>;
    }>(`/planets/${encodeURIComponent(name)}`),

  // Analyses
  createAnalysis: (data: {
    planet_name: string;
    observation_id?: string;
    n_live?: number;
    dlogz: number;
    max_iter?: number;
    maxcall?: number | null;
    seed: number;
    run_ml: boolean;
    run_detection: boolean;
    run_quality: boolean;
    run_preprocess: boolean;
    run_quality_check?: boolean;
    error_inflation?: number;
    error_inflation_free?: boolean;
    fixed_params: Record<string, any>;
  }) => request<{ analysis_id: string; status: string; message: string }>('/analyses', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  listAnalyses: (page = 1, pageSize = 20, status?: string) => 
    request<{
      items: Array<{
        analysis_id: string;
        planet_name: string;
        observation_id: string;
        status: string;
        stage: string;
        progress: number;
        created_at: string;
        completed_at: string | null;
      }>;
      total: number;
      page: number;
      page_size: number;
    }>(`/analyses?page=${page}&page_size=${pageSize}${status ? `&status=${status}` : ''}`),

  getAnalysis: (analysisId: string) => 
    request<{
      analysis_id: string;
      planet_name: string;
      observation_id: string;
      status: string;
      stage: string;
      progress: number;
      current_step: string;
      error_message: string | null;
      config: Record<string, any>;
      seed: number;
      n_live: number;
      dlogz: number;
      max_iter: number;
      created_at: string;
      started_at: string | null;
      completed_at: string | null;
      updated_at: string;
      provenance: Record<string, any> | null;
    }>(`/analyses/${analysisId}`),

  getAnalysisStatus: (analysisId: string) =>
    request<{
      analysis_id: string;
      status: string;
      stage: string;
      progress: number;
      current_step: string;
      error_message: string | null;
      created_at: string;
      started_at: string | null;
      completed_at: string | null;
      updated_at: string;
    }>(`/analyses/${analysisId}/status`),

    getSpectrum: (analysisId: string, cleaned: boolean = false) => 
      request<{
        wavelength_um: number[];
        transmission: number[];
        uncertainty: number[];
        wavelength_bin_edges_um: number[];
        quality_flags: string[];
        observation_id: string;
        target_id: string;
        instrument: string;
        provenance: Record<string, any> | null;
      }>(`/analyses/${analysisId}/spectrum?cleaned=${cleaned}`),

    getQuality: (analysisId: string) => 
      request<{
        analysis_id: string;
        overall_suitability: 'GOOD' | 'LIMITED' | 'POOR';
        median_snr: number | null;
        max_band_snr: number | null;
        wavelength_coverage_fraction: number;
        flagged_fraction: number;
        nan_fraction: number;
        outlier_count: number;
        outlier_fraction: number;
        median_uncertainty: number | null;
        uncertainty_non_positive_count: number;
        uncertainty_nan_count: number;
        uncertainty_huge_count: number;
        uncertainty_tiny_count: number;
        effective_resolving_power: number | null;
        molecule_ratings: Record<string, 'GOOD' | 'LIMITED' | 'POOR'>;
        molecule_details: Record<string, {
          molecule: string;
          rating: 'GOOD' | 'LIMITED' | 'POOR';
          coverage_fraction: number;
          n_points: number;
          snr: number | null;
        }>;
        thresholds: Record<string, number>;
        config_version: string;
        method: Record<string, string>;
      }>(`/analyses/${analysisId}/quality`),

    getMLResults: (analysisId: string) => 
      request<{
        analysis_id: string;
        scores: Record<string, number>;
        model_version: string;
        dataset_hash: string;
        model_config_hash: string;
        timestamp: string;
        input_coverage_fraction: number;
        input_wavelength_range: [number, number];
        grid_match: boolean;
        warnings: string[];
      }>(`/analyses/${analysisId}/ml`),

    getRetrieval: (analysisId: string) => 
      request<{
        analysis_id: string;
        logz: number;
        logz_err: number;
        best_fit: Record<string, number>;
        median: Record<string, number>;
        ci_68: Record<string, [number, number]>;
        ci_95: Record<string, [number, number]>;
        param_names: string[];
        n_samples: number;
        n_live: number;
        dlogz: number;
        runtime_s: number;
        sampler: string;
        seed: number;
        error_inflation: number;
      }>(`/analyses/${analysisId}/retrieval`),

    getPosterior: (analysisId: string) =>
      request<{
        analysis_id: string;
        samples: number[][];
        weights: number[];
        param_names: string[];
        logz: number;
        logz_err: number;
      }>(`/analyses/${analysisId}/posterior`),

    getDetection: (analysisId: string) => 
      request<{
        analysis_id: string;
        results: Array<{
          molecule: string;
          ln_bayes_factor: number;
          ln_bayes_factor_err: number | null;
          sigma_equivalent: number;
          status: 'detected' | 'tentative' | 'not_detected';
          upper_limit_log_vmr: number | null;
          details: Record<string, any>;
        }>;
        summary: Record<string, string>;
      }>(`/analyses/${analysisId}/detection`),

    getModel: (analysisId: string) => 
      request<{
        analysis_id: string;
        wavelength_um: number[];
        best_fit_depth: number[];
        median_depth: number[];
        ci_lo: number[];
        ci_hi: number[];
        observed_wavelength_um: number[] | null;
        observed_depth: number[] | null;
        observed_uncertainty: number[] | null;
      }>(`/analyses/${analysisId}/model`),

    getProvenance: (analysisId: string) =>
      request<{
        analysis_id: string;
        planet: string | null;
        observation_id: string | null;
        telescope: string | null;
        instrument: string | null;
        source_archive: string | null;
        input_data_version: string | null;
        input_data_hash: string | null;
        preprocessing_version: string | null;
        ml_model_version: string | null;
        retrieval_model_version: string | null;
        retrieval_parameters: Record<string, any>;
        timestamp: string;
        result_reference: string | null;
      }>(`/analyses/${analysisId}/provenance`),

    deleteAnalysis: (analysisId: string) => 
      request<{ message: string }>(`/analyses/${analysisId}`, { method: 'DELETE' }),

    getTwin: (analysisId: string) =>
      request<TwinParameters>(`/analyses/${analysisId}/twin`),

    generateReport: (analysisId: string, format: 'html' | 'pdf' = 'html') =>
      request<{
        analysis_id: string;
        format: string;
        file_path: string;
        file_hash: string;
        report_version: string;
        created_at: string | null;
      }>(`/analyses/${analysisId}/report`, {
        method: 'POST',
        body: JSON.stringify({ format }),
      }),

    reportDownloadUrl: (analysisId: string, format: 'html' | 'pdf' = 'html') =>
      `${API_BASE}/analyses/${analysisId}/report?format=${format}`,

    // Config
    getMoleculeBands: () => 
      request<{
        version: string;
        wavelength_range_um: [number, number];
        molecules: Record<string, Array<[number, number]>>;
        quality: {
          continuum_window_points: number;
          outlier_n_sigma: number;
          overall: Record<string, number>;
          molecule: Record<string, number>;
          uncertainty: Record<string, number>;
        };
      }>('/config/molecule-bands'),

    // Utility
    getMoleculeBandsConfig: () => 
      request<{
        molecules: Record<string, Array<[number, number]>>;
        wavelength_range_um: [number, number];
        quality: Record<string, any>;
        version: string;
      }>('/config/molecule-bands'),

  // Query keys for React Query
  keys: {
    health: ['health'],
    planets: (query: string) => ['planets', 'search', query],
    planet: (name: string) => ['planets', 'detail', name],
    analyses: (page: number, pageSize: number, status?: string) => ['analyses', 'list', page, pageSize, status],
    analysis: (id: string) => ['analyses', 'detail', id],
    analysisStatus: (id: string) => ['analyses', 'status', id],
    spectrum: (id: string, cleaned: boolean) => ['analyses', 'spectrum', id, cleaned],
    quality: (id: string) => ['analyses', 'quality', id],
    ml: (id: string) => ['analyses', 'ml', id],
    retrieval: (id: string) => ['analyses', 'retrieval', id],
    posterior: (id: string) => ['analyses', 'posterior', id],
    detection: (id: string) => ['analyses', 'detection', id],
    model: (id: string) => ['analyses', 'model', id],
    provenance: (id: string) => ['analyses', 'provenance', id],
    twin: (id: string) => ['analyses', 'twin', id],
    report: (id: string) => ['analyses', 'report', id],
    moleculeBands: ['config', 'moleculeBands'],
  },
};

export interface TwinParameter {
  name: string;
  label: string;
  value: number | null;
  unit: string;
  source: 'measured' | 'inferred' | 'derived' | 'assumed';
  ci_68: [number, number] | null;
  constrained?: boolean | null;
  note: string | null;
}

export interface TwinMoleculeContribution {
  molecule: string;
  contribution_fraction: number;
  in_band: boolean;
  note: string;
}

export interface TwinParameters {
  analysis_id: string;
  planet_name: string;
  parameters: TwinParameter[];
  molecules: TwinMoleculeContribution[];
  meta: Record<string, unknown>;
}

export type {
  // Types would be defined here based on the API responses
};