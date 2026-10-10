export type QualityRating = 'GOOD' | 'LIMITED' | 'POOR';

export interface MoleculeEvidence {
  molecule: string;
  /** Data-quality rating: what the data can support (never a composition claim). */
  qualityRating: QualityRating | null;
  /** ML candidate score 0-1 (not an abundance, not a detection). */
  mlScore: number | null;
  mlModelVersion?: string | null;
  /** Retrieval nested-model evidence (authoritative). */
  lnBayes: number | null;
  lnBayesErr?: number | null;
  sigmaEquivalent?: number | null;
  detectionStatus?: 'detected' | 'tentative' | 'not_detected' | null;
  upperLimitLogVmr?: number | null;
}

const RATING_STYLES: Record<QualityRating, string> = {
  GOOD: 'bg-green-100 text-green-800',
  LIMITED: 'bg-yellow-100 text-yellow-800',
  POOR: 'bg-red-100 text-red-800',
};

function verdict(evidence: MoleculeEvidence): { text: string; className: string } {
  if (evidence.qualityRating === 'POOR') {
    return { text: 'Not constrained (POOR data quality)', className: 'not-constrained' };
  }
  if (evidence.lnBayes === null || evidence.lnBayes === undefined) {
    return { text: 'Not assessed (no model comparison stored)', className: 'not-constrained' };
  }
  if (evidence.lnBayes >= 5) {
    return { text: `Supported (ln B = ${evidence.lnBayes.toFixed(2)} ≥ 5)`, className: 'supported' };
  }
  if (evidence.lnBayes >= 3) {
    return {
      text: `Weakly supported (3 ≤ ln B = ${evidence.lnBayes.toFixed(2)} < 5)`,
      className: 'weak',
    };
  }
  return { text: `Not constrained (ln B = ${evidence.lnBayes.toFixed(2)} < 3)`, className: 'not-constrained' };
}

/**
 * Per-molecule evidence cards with three clearly separated sections:
 * (1) data quality, (2) ML candidate score, (3) retrieval evidence.
 * ML numbers are always labelled and never drive the verdict.
 */
export function MoleculePanel({ evidence }: { evidence: MoleculeEvidence[] }) {
  return (
    <div className="space-y-3" aria-label="Per-molecule evidence">
      {evidence.map((item) => {
        const assessment = verdict(item);
        return (
          <div key={item.molecule} className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-gray-900">{item.molecule}</h4>
              <span className={`text-xs font-medium ${assessment.className}`}>{assessment.text}</span>
            </div>

            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
              <div className="rounded bg-gray-50 p-2">
                <div className="font-medium text-gray-700 mb-1">1. Data quality</div>
                {item.qualityRating ? (
                  <span className={`inline-flex px-2 py-0.5 rounded-full font-medium ${RATING_STYLES[item.qualityRating]}`}>
                    {item.qualityRating}
                  </span>
                ) : (
                  <span className="text-gray-500">Not assessed</span>
                )}
              </div>

              <div className="rounded bg-gray-50 p-2">
                <div className="font-medium text-gray-700 mb-1">2. ML candidate score</div>
                {item.mlScore !== null && item.mlScore !== undefined ? (
                  <span className="font-mono text-gray-900">
                    {item.mlScore.toFixed(4)} (ML candidate score)
                  </span>
                ) : (
                  <span className="text-gray-500">Not run</span>
                )}
              </div>

              <div className="rounded bg-gray-50 p-2">
                <div className="font-medium text-gray-700 mb-1">3. Retrieval evidence</div>
                {item.lnBayes !== null && item.lnBayes !== undefined ? (
                  <span className="font-mono text-gray-900">
                    ln B = {item.lnBayes.toFixed(2)}
                    {item.lnBayesErr !== null && item.lnBayesErr !== undefined ? ` ± ${item.lnBayesErr.toFixed(2)}` : ''}
                    {item.upperLimitLogVmr !== null && item.upperLimitLogVmr !== undefined
                      ? `; 95% upper limit ${item.upperLimitLogVmr.toFixed(2)} dex`
                      : ''}
                  </span>
                ) : (
                  <span className="text-gray-500">Not run</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default MoleculePanel;
