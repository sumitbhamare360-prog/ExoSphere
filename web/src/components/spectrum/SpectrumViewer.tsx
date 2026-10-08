import { useMemo, useState } from 'react';

interface SpectrumViewerProps {
  spectrum: {
    wavelength: number[];
    transmission: number[];
    uncertainty: number[];
    wavelengthBinEdges?: number[];
    qualityFlags?: string[];
  };
  model?: {
    wavelength: number[];
    depth: number[];
    credibleLo?: number[];
    credibleHi?: number[];
  };
  moleculeBands: Record<string, Array<{ start: number; end: number }>>;
  moleculeColors: Record<string, string>;
  onWavelengthSelect?: (wavelength: number) => void;
  onRegionSelect?: (region: { start: number; end: number }) => void;
  height?: number;
  showCredibleBand?: boolean;
  showModel?: boolean;
  showMoleculeBands?: boolean;
}

const DEFAULT_COLORS: Record<string, string> = {
  H2O: '#3b82f6',
  CO2: '#ef4444',
  CO: '#f97316',
  CH4: '#8b5cf6',
  SO2: '#f59e0b',
};

const MODEL_COLOR = '#6366f1';

/**
 * Minimal working spectrum plot (SVG).
 *
 * Renders observed points with error bars, molecule-band overlays, and an
 * optional model line. Interactive zoom and the model credible band are
 * intentionally not implemented here (tracked as Phase 7 follow-up work);
 * the corresponding props are accepted so callers compile, and
 * `showCredibleBand` is currently a no-op documented as such.
 */
export function SpectrumViewer({
  spectrum,
  model,
  moleculeBands,
  moleculeColors = {},
  onWavelengthSelect,
  height = 500,
  showModel = true,
  showMoleculeBands = true,
}: SpectrumViewerProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const { wavelength, transmission, uncertainty } = spectrum;
  const n = wavelength.length;

  const colors = { ...DEFAULT_COLORS, ...moleculeColors };

  const ranges = useMemo(() => {
    if (n === 0) return null;
    const xMin = Math.min(...wavelength);
    const xMax = Math.max(...wavelength);
    const allDepths = model && showModel ? [...transmission, ...model.depth] : transmission;
    const yMin = Math.min(...allDepths);
    const yMax = Math.max(...allDepths);
    const yPad = (yMax - yMin) * 0.1 || 1;
    return { xMin, xMax, yMin: yMin - yPad, yMax: yMax + yPad };
  }, [wavelength, transmission, model, showModel, n]);

  const bandOverlays = useMemo(() => {
    if (!ranges) return [];
    const overlays: Array<{ molecule: string; start: number; end: number; color: string }> = [];
    for (const [molecule, bands] of Object.entries(moleculeBands)) {
      for (const band of bands) {
        if (band.end >= ranges.xMin && band.start <= ranges.xMax) {
          overlays.push({
            molecule,
            start: Math.max(band.start, ranges.xMin),
            end: Math.min(band.end, ranges.xMax),
            color: colors[molecule] ?? '#888888',
          });
        }
      }
    }
    return overlays;
  }, [moleculeBands, ranges, colors]);

  if (!ranges || n === 0) {
    return <p className="text-sm text-gray-500">No spectrum data to display.</p>;
  }

  const xScale = (w: number) => ((w - ranges.xMin) / (ranges.xMax - ranges.xMin)) * 100;
  const yScale = (d: number) => 100 - ((d - ranges.yMin) / (ranges.yMax - ranges.yMin)) * 100;

  const spectrumPath = wavelength.map((w, i) => `${xScale(w).toFixed(2)},${yScale(transmission[i]).toFixed(2)}`).join(' ');
  const modelPath =
    model && showModel
      ? model.wavelength.map((w, i) => `${xScale(w).toFixed(2)},${yScale(model.depth[i]).toFixed(2)}`).join(' ')
      : null;

  const stride = Math.max(1, Math.floor(n / 200));
  const errorBars: number[] = [];
  for (let i = 0; i < n; i += stride) errorBars.push(i);

  const hover = hoverIndex !== null ? { i: hoverIndex, w: wavelength[hoverIndex], d: transmission[hoverIndex] } : null;

  return (
    <div className="w-full relative" style={{ height }}>
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="w-full h-full"
        role="img"
        aria-label="Observed transmission spectrum with molecule band overlays"
        onMouseMove={(e) => {
          const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const frac = (e.clientX - rect.left) / rect.width;
          const wGuess = ranges.xMin + frac * (ranges.xMax - ranges.xMin);
          let best = 0;
          let bestDist = Infinity;
          for (let i = 0; i < n; i++) {
            const dist = Math.abs(wavelength[i] - wGuess);
            if (dist < bestDist) {
              bestDist = dist;
              best = i;
            }
          }
          setHoverIndex(best);
        }}
        onMouseLeave={() => setHoverIndex(null)}
        onClick={() => {
          if (hoverIndex !== null) onWavelengthSelect?.(wavelength[hoverIndex]);
        }}
      >
        {showMoleculeBands &&
          bandOverlays.map((band, i) => (
            <g key={i}>
              <rect
                x={`${xScale(band.start)}%`}
                width={`${Math.max(xScale(band.end) - xScale(band.start), 0.3)}%`}
                y="0"
                height="100%"
                fill={band.color}
                fillOpacity={0.1}
              />
              <text
                x={`${(xScale(band.start) + xScale(band.end)) / 2}%`}
                y="95%"
                textAnchor="middle"
                fontSize="4"
                fill={band.color}
                fillOpacity="0.9"
              >
                {band.molecule}
              </text>
            </g>
          ))}

        <polyline points={spectrumPath} fill="none" stroke="#1f2937" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />

        {modelPath && (
          <polyline points={modelPath} fill="none" stroke={MODEL_COLOR} strokeWidth="1" strokeDasharray="5,5" vectorEffect="non-scaling-stroke" />
        )}

        {errorBars.map((i) => (
          <line
            key={i}
            x1={`${xScale(wavelength[i])}%`}
            y1={`${yScale(transmission[i] - uncertainty[i])}%`}
            x2={`${xScale(wavelength[i])}%`}
            y2={`${yScale(transmission[i] + uncertainty[i])}%`}
            stroke="#9ca3af"
            strokeWidth="0.5"
            strokeOpacity="0.6"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {hover && (
          <g>
            <line
              x1={`${xScale(hover.w)}%`}
              y1="0"
              x2={`${xScale(hover.w)}%`}
              y2="100%"
              stroke="#ef4444"
              strokeWidth="1"
              strokeDasharray="4,4"
              vectorEffect="non-scaling-stroke"
            />
            <circle cx={`${xScale(hover.w)}%`} cy={`${yScale(hover.d)}%`} r="3" fill="#ef4444" stroke="white" strokeWidth="1" />
          </g>
        )}
      </svg>

      {hover && (
        <div className="absolute pointer-events-none bg-white border border-gray-200 rounded-lg shadow-lg p-2 text-xs z-10 font-mono">
          <div>λ: {hover.w.toFixed(3)} µm</div>
          <div>Depth: {hover.d.toExponential(3)}</div>
        </div>
      )}
    </div>
  );
}

export default SpectrumViewer;
