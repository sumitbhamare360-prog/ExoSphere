import { useQuery } from '@tanstack/react-query';
import { api } from '../../lib/api';
import { useExoStore } from '../../store/useExoStore';
import { SpectrumViewer } from '../spectrum/SpectrumViewer';
import { MoleculePanel, type MoleculeEvidence } from '../molecule/MoleculePanel';
import { MOLECULE_COLORS } from '../../lib/twinGeometry';

const MOLECULES = ['H2O', 'CO2', 'CO', 'CH4', 'SO2'];

/**
 * Results view for a real analysis: observed spectrum with the best-fit
 * model and molecule-band overlays, data-quality status, and the
 * per-molecule evidence panel (quality / ML / retrieval clearly separated).
 * Selecting a molecule here shares state with the 3D twin below.
 */
export function ResultsView({ analysisId }: { analysisId: string }) {
  const moleculeBands = useExoStore((s) => s.moleculeBands);
  const setSelectedMolecule = useExoStore((s) => s.setSelectedMolecule);

  const spectrumQuery = useQuery({
    queryKey: api.keys.spectrum(analysisId, true),
    queryFn: () => api.getSpectrum(analysisId, true),
  });
  const modelQuery = useQuery({
    queryKey: api.keys.model(analysisId),
    queryFn: () => api.getModel(analysisId),
    retry: false,
  });
  const qualityQuery = useQuery({
    queryKey: api.keys.quality(analysisId),
    queryFn: () => api.getQuality(analysisId),
    retry: false,
  });
  const mlQuery = useQuery({
    queryKey: api.keys.ml(analysisId),
    queryFn: () => api.getMLResults(analysisId),
    retry: false,
  });
  const detectionQuery = useQuery({
    queryKey: api.keys.detection(analysisId),
    queryFn: () => api.getDetection(analysisId),
    retry: false,
  });

  if (spectrumQuery.isLoading) {
    return <p className="text-sm text-gray-500">Loading results…</p>;
  }
  if (spectrumQuery.isError || !spectrumQuery.data) {
    return (
      <p className="text-sm text-red-600" role="alert">
        Failed to load spectrum:{' '}
        {spectrumQuery.error instanceof Error ? spectrumQuery.error.message : 'unknown error'}
      </p>
    );
  }

  const spectrum = spectrumQuery.data;
  const model = modelQuery.data;
  const quality = qualityQuery.data;
  const ml = mlQuery.data;
  const detectionByMolecule = new Map(
    (detectionQuery.data?.results ?? []).map((d) => [d.molecule, d])
  );

  const evidence: MoleculeEvidence[] = MOLECULES.map((molecule) => {
    const det = detectionByMolecule.get(molecule);
    const rating = quality?.molecule_ratings?.[molecule] as
      | 'GOOD'
      | 'LIMITED'
      | 'POOR'
      | undefined;
    return {
      molecule,
      qualityRating: rating ?? null,
      mlScore: ml?.scores?.[molecule] ?? null,
      mlModelVersion: ml?.model_version ?? null,
      lnBayes: det?.ln_bayes_factor ?? null,
      lnBayesErr: det?.ln_bayes_factor_err ?? null,
      sigmaEquivalent: det?.sigma_equivalent ?? null,
      detectionStatus: det?.status ?? null,
      upperLimitLogVmr: det?.upper_limit_log_vmr ?? null,
    };
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg font-semibold text-gray-900">Results</h2>
        {quality && (
          <span
            className={`px-2 py-0.5 rounded-full text-xs font-medium ${
              quality.overall_suitability === 'GOOD'
                ? 'bg-green-100 text-green-800'
                : quality.overall_suitability === 'LIMITED'
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
            }`}
          >
            Data quality: {quality.overall_suitability}
          </span>
        )}
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-3">
        <SpectrumViewer
          spectrum={{
            wavelength: spectrum.wavelength_um,
            transmission: spectrum.transmission,
            uncertainty: spectrum.uncertainty,
            wavelengthBinEdges: spectrum.wavelength_bin_edges_um,
            qualityFlags: spectrum.quality_flags,
          }}
          model={
            model && model.best_fit_depth.length > 0
              ? {
                  wavelength: model.wavelength_um,
                  depth: model.best_fit_depth,
                  credibleLo: model.ci_lo,
                  credibleHi: model.ci_hi,
                }
              : undefined
          }
          moleculeBands={moleculeBands}
          moleculeColors={MOLECULE_COLORS}
          onWavelengthSelect={(wavelength) => {
            const entry = Object.entries(moleculeBands).find(([, bands]) =>
              bands.some((b) => wavelength >= b.start && wavelength <= b.end)
            );
            setSelectedMolecule(entry ? entry[0] : null);
          }}
          height={380}
        />
      </div>

      <MoleculePanel evidence={evidence} />
    </div>
  );
}

export default ResultsView;
