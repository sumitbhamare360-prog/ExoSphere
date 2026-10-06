"""Bin-edge construction from spectral bin centers (Spectrum needs N+1 edges)."""

from __future__ import annotations

import numpy as np


def bin_edges_from_centers(centers, widths=None) -> list[float]:
    """Build ascending bin edges from bin centers.

    Inner edges are the midpoints of adjacent centers, which guarantees strict
    ordering for strictly ascending centers. Outer edges use half of the first /
    last bin width when ``widths`` is given and valid, otherwise half the nearest
    neighbor spacing.
    """
    c = np.asarray(centers, dtype=float)
    n = c.size
    if n == 0:
        raise ValueError("cannot build bin edges from an empty wavelength array")
    if n == 1:
        if widths is None:
            raise ValueError(
                "cannot infer bin edges for a single point without bin widths"
            )
        w = np.asarray(widths, dtype=float)
        if not (np.isfinite(w[0]) and w[0] > 0):
            raise ValueError("bin width must be > 0 for a single-point spectrum")
        return [float(c[0] - w[0] / 2), float(c[0] + w[0] / 2)]
    if not bool(np.all(np.diff(c) > 0)):
        raise ValueError("bin centers must be strictly ascending")

    edges = np.empty(n + 1, dtype=float)
    edges[1:n] = (c[:-1] + c[1:]) / 2.0

    use_widths = widths is not None
    if use_widths:
        w = np.asarray(widths, dtype=float)
        use_widths = bool(
            w.shape == c.shape
            and np.all(np.isfinite(w))
            and w[0] > 0
            and w[-1] > 0
        )

    first_gap = (c[1] - c[0]) / 2.0
    if use_widths:
        outer_first = c[0] - w[0] / 2.0
        edges[0] = outer_first if outer_first < edges[1] else c[0] - first_gap
    else:
        edges[0] = c[0] - first_gap

    last_gap = (c[n - 1] - c[n - 2]) / 2.0
    if use_widths:
        outer_last = c[n - 1] + w[n - 1] / 2.0
        edges[n] = outer_last if outer_last > edges[n - 1] else c[n - 1] + last_gap
    else:
        edges[n] = c[n - 1] + last_gap

    return [float(value) for value in edges]
