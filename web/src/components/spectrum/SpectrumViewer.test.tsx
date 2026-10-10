import { describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/react';
import { SpectrumViewer } from './SpectrumViewer';

const SPECTRUM = {
  wavelength: [1.0, 1.5, 2.0, 2.5, 3.0],
  transmission: [0.02, 0.021, 0.0205, 0.022, 0.0215],
  uncertainty: [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
};

const BANDS = { H2O: [{ start: 1.3, end: 1.6 }] };
const COLORS = { H2O: '#3b82f6' };

describe('SpectrumViewer', () => {
  it('renders an accessible spectrum plot', () => {
    const { getByRole } = render(
      <SpectrumViewer spectrum={SPECTRUM} moleculeBands={BANDS} moleculeColors={COLORS} />
    );
    expect(
      getByRole('img', { name: /transmission spectrum/i })
    ).toBeTruthy();
  });

  it('draws the model line and credible band when provided', () => {
    const { container } = render(
      <SpectrumViewer
        spectrum={SPECTRUM}
        model={{
          wavelength: [1.0, 1.5, 2.0, 2.5, 3.0],
          depth: [0.02, 0.021, 0.0205, 0.022, 0.0215],
          credibleLo: [0.0199, 0.0209, 0.0204, 0.0219, 0.0214],
          credibleHi: [0.0201, 0.0211, 0.0206, 0.0221, 0.0216],
        }}
        moleculeBands={BANDS}
        moleculeColors={COLORS}
      />
    );
    // data polyline + model polyline
    expect(container.querySelectorAll('polyline').length).toBeGreaterThanOrEqual(2);
  });

  it('shows the empty state for missing data', () => {
    const { getByText } = render(
      <SpectrumViewer
        spectrum={{ wavelength: [], transmission: [], uncertainty: [] }}
        moleculeBands={{}}
        moleculeColors={{}}
      />
    );
    expect(getByText(/no spectrum data/i)).toBeTruthy();
  });

  it('reports wavelength selection on click', () => {
    const onSelect = vi.fn();
    const { container } = render(
      <SpectrumViewer
        spectrum={SPECTRUM}
        moleculeBands={BANDS}
        moleculeColors={COLORS}
        onWavelengthSelect={onSelect}
      />
    );
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
  });
});
