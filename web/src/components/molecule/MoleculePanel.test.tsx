import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { MoleculePanel, type MoleculeEvidence } from './MoleculePanel';

const FULL: MoleculeEvidence[] = [
  {
    molecule: 'H2O',
    qualityRating: 'GOOD',
    mlScore: 0.9,
    mlModelVersion: 'CNN-v1',
    lnBayes: 12.5,
    lnBayesErr: 0.6,
    sigmaEquivalent: 5.0,
    detectionStatus: 'detected',
    upperLimitLogVmr: null,
  },
  {
    molecule: 'CH4',
    qualityRating: 'POOR',
    mlScore: 0.1,
    lnBayes: 9.9,
    detectionStatus: 'detected',
    upperLimitLogVmr: null,
  },
  {
    molecule: 'CO',
    qualityRating: 'GOOD',
    mlScore: null,
    lnBayes: null,
    detectionStatus: null,
    upperLimitLogVmr: null,
  },
];

describe('MoleculePanel', () => {
  it('renders three separated sections per molecule', () => {
    const { getAllByText } = render(<MoleculePanel evidence={FULL} />);
    expect(getAllByText('1. Data quality')).toHaveLength(3);
    expect(getAllByText('2. ML candidate score')).toHaveLength(3);
    expect(getAllByText('3. Retrieval evidence')).toHaveLength(3);
  });

  it('labels every ML number as a candidate score', () => {
    const { container } = render(<MoleculePanel evidence={FULL} />);
    const text = container.textContent ?? '';
    expect(text).toContain('0.9000 (ML candidate score)');
    expect(text).not.toMatch(/abundance/i);
  });

  it('lets data quality override strong evidence (POOR -> not constrained)', () => {
    const { getByText } = render(<MoleculePanel evidence={FULL} />);
    expect(
      getByText('Not constrained (POOR data quality)', { selector: 'span' })
    ).toBeTruthy();
  });

  it('marks missing stages as not run', () => {
    const { getAllByText } = render(<MoleculePanel evidence={FULL} />);
    // CO has no ML score and no retrieval evidence.
    expect(getAllByText('Not run').length).toBeGreaterThanOrEqual(2);
  });

  it('shows upper limits in dex when present', () => {
    const withLimit: MoleculeEvidence[] = [
      {
        molecule: 'CO',
        qualityRating: 'LIMITED',
        mlScore: 0.2,
        lnBayes: 0.5,
        detectionStatus: 'not_detected',
        upperLimitLogVmr: -5.5,
      },
    ];
    const { getByText } = render(<MoleculePanel evidence={withLimit} />);
    expect(getByText(/95% upper limit -5.50 dex/)).toBeTruthy();
  });
});
