"""Preprocessing: clean a spectrum before quality assessment / retrieval.

`clean()` never mutates its input; it returns a new :class:`Spectrum` plus a
:class:`PreprocessLog` recording every removed point (original input index +
reason), the parameters used, and the preprocessing version that is written
into the output spectrum's provenance.

Removal order (each step sees the survivors of the previous step):
    1. ``nan``             - non-finite wavelength/depth
    2. ``flagged``         - non-empty quality flag
    3. ``bad_uncertainty`` - non-finite, <= 0, huge (> max_frac_of_depth x the
                             median depth) or implausibly tiny sigma
    4. ``outlier``         - robust MAD spike clip: each interior point is
                             compared with the linear prediction from its two
                             direct neighbours, z = r / sigma_eff, and flagged
                             when |z - median(z)| > clip_n_sigma x 1.4826 x
                             MAD(z); smooth spectral features survive, isolated
                             spikes do not (endpoints and points across
                             wavelength gaps are never flagged).
Optional inverse-variance rebinning groups the surviving points onto a regular
grid; empty bins are dropped, never filled.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.quality.assess import is_flagged, load_quality_config, mad_outlier_mask

PREPROCESS_VERSION = "preprocess-1.0.0"

RebinMode = Literal["resolution", "count"]


class CleanOptions(BaseModel):
    """Options for :func:`clean`. Defaults are the documented behaviour."""

    drop_nan: bool = True
    drop_flagged: bool = True
    drop_bad_uncertainty: bool = True
    sigma_clip: bool = True
    clip_n_sigma: float = 5.0
    rebin: RebinMode | None = None
    target_resolution_um: float | None = None  # constant d-lambda grid ('resolution')
    bin_count: int | None = None  # equal-width grid ('count')

    def parameters_dict(self) -> dict[str, Any]:
        return self.model_dump()


class RemovedPoint(BaseModel):
    index: int  # index in the INPUT spectrum
    reason: str


class RebinGroup(BaseModel):
    output_index: int
    input_indices: list[int]  # input-spectrum indices merged into this bin


class RebinLog(BaseModel):
    mode: RebinMode
    n_input: int
    n_output: int
    grid_edges: list[float] = Field(default_factory=list)
    empty_bin_centers: list[float] = Field(default_factory=list)
    groups: list[RebinGroup] = Field(default_factory=list)


class PreprocessLog(BaseModel):
    """Every removal/transform performed by :func:`clean` (JSON serializable)."""

    preprocessing_version: str = PREPROCESS_VERSION
    n_input: int
    n_output: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    removed: list[RemovedPoint] = Field(default_factory=list)
    rebin: RebinLog | None = None

    @property
    def removal_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.removed:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json()


def _drop(
    wavelength: np.ndarray,
    depth: np.ndarray,
    sigma: np.ndarray,
    flags: list[str],
    widths: np.ndarray,
    indices: np.ndarray,
    bad: np.ndarray,
    reason: str,
    removed: list[RemovedPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Record ``bad`` points (with their original input indices) and drop them."""
    for local in np.nonzero(bad)[0]:
        removed.append(RemovedPoint(index=int(indices[local]), reason=reason))
    keep = ~bad
    return (
        wavelength[keep],
        depth[keep],
        sigma[keep],
        [flag for flag, k in zip(flags, keep, strict=True) if k] if flags else [],
        widths[keep],
        indices[keep],
    )


def clean(
    spectrum: Spectrum, options: CleanOptions | None = None
) -> tuple[Spectrum, PreprocessLog]:
    """Clean a spectrum: drop bad points, MAD sigma-clip, optional rebin.

    Returns a NEW spectrum (the input is never mutated) and a PreprocessLog.
    Raises ValueError if every point would be removed.
    """
    opts = options if options is not None else CleanOptions()
    thresholds = load_quality_config().thresholds
    removed: list[RemovedPoint] = []

    all_edges = np.asarray(spectrum.wavelength_bin_edges, dtype=np.float64)
    wavelength = np.asarray(spectrum.wavelength, dtype=np.float64)
    depth = np.asarray(spectrum.transmission, dtype=np.float64)
    sigma = np.asarray(spectrum.uncertainty, dtype=np.float64)
    flags = list(spectrum.quality_flags)
    widths = all_edges[1:] - all_edges[:-1]
    indices = np.arange(wavelength.size, dtype=int)
    n_input = int(wavelength.size)

    # 1. NaN / non-finite values
    if opts.drop_nan:
        bad = ~np.isfinite(depth) | ~np.isfinite(wavelength)
        wavelength, depth, sigma, flags, widths, indices = _drop(
            wavelength, depth, sigma, flags, widths, indices, bad, "nan", removed
        )

    # 2. flagged points
    if opts.drop_flagged and flags:
        bad = np.array([is_flagged(flag) for flag in flags], dtype=bool)
        wavelength, depth, sigma, flags, widths, indices = _drop(
            wavelength, depth, sigma, flags, widths, indices, bad, "flagged", removed
        )

    # 3. uncertainty sanity
    if opts.drop_bad_uncertainty and depth.size:
        finite_depth = depth[np.isfinite(depth)]
        median_depth = float(np.median(finite_depth)) if finite_depth.size else None
        huge_limit = (
            float(thresholds["uncertainty_max_frac_of_depth"]) * abs(median_depth)
            if median_depth is not None and median_depth != 0
            else None
        )
        tiny_limit = float(thresholds["uncertainty_min_fractional"])
        bad = ~np.isfinite(sigma) | (sigma <= 0)
        if huge_limit is not None and huge_limit > 0:
            bad = bad | (sigma > huge_limit)
        bad = bad | ((sigma > 0) & (sigma < tiny_limit))
        wavelength, depth, sigma, flags, widths, indices = _drop(
            wavelength, depth, sigma, flags, widths, indices, bad, "bad_uncertainty", removed
        )

    # 4. robust MAD clip: point vs linear prediction from its two neighbours
    if opts.sigma_clip and depth.size:
        outlier = mad_outlier_mask(wavelength, depth, sigma, opts.clip_n_sigma)
        wavelength, depth, sigma, flags, widths, indices = _drop(
            wavelength, depth, sigma, flags, widths, indices, outlier, "outlier", removed
        )

    if wavelength.size == 0:
        raise ValueError("clean() removed every point; nothing left to return")

    out_edges = bin_edges_from_centers(wavelength, widths)

    rebin_log: RebinLog | None = None
    if opts.rebin is not None:
        wavelength, depth, sigma, flags, out_edges, rebin_log = _rebin(
            wavelength, depth, sigma, flags, indices, opts, removed
        )

    if wavelength.size == 0:
        raise ValueError("clean() removed every point; nothing left to return")

    new_provenance = spectrum.provenance.model_copy(
        update={"preprocessing_version": PREPROCESS_VERSION}
    )
    cleaned = Spectrum(
        wavelength=[float(v) for v in wavelength],
        transmission=[float(v) for v in depth],
        uncertainty=[float(v) for v in sigma],
        wavelength_bin_edges=[float(v) for v in out_edges],
        quality_flags=flags,
        observation_id=spectrum.observation_id,
        target_id=spectrum.target_id,
        instrument=spectrum.instrument,
        provenance=new_provenance,
    )
    log = PreprocessLog(
        preprocessing_version=PREPROCESS_VERSION,
        n_input=n_input,
        n_output=int(wavelength.size),
        parameters=opts.parameters_dict(),
        removed=removed,
        rebin=rebin_log,
    )
    return cleaned, log


def _rebin(
    wavelength: np.ndarray,
    depth: np.ndarray,
    sigma: np.ndarray,
    flags: list[str],
    indices: np.ndarray,
    opts: CleanOptions,
    removed: list[RemovedPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[float], RebinLog]:
    """Inverse-variance rebin onto a regular grid.

    ``new_depth = sum(w_i d_i)/sum(w_i)`` with ``w_i = 1/sigma_i^2`` and
    ``new_sigma = 1/sqrt(sum(w_i))`` - the correct error propagation for the
    weighted mean of independent Gaussian measurements.
    """
    if opts.rebin == "resolution":
        step = float(opts.target_resolution_um or 0.0)
        if step <= 0:
            raise ValueError("rebin='resolution' requires target_resolution_um > 0")
        start = float(np.floor(wavelength[0] / step) * step)
        stop = float(np.ceil(wavelength[-1] / step) * step)
        grid = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    else:
        count = int(opts.bin_count or 0)
        if count < 1:
            raise ValueError("rebin='count' requires bin_count >= 1")
        grid = np.linspace(float(wavelength[0]), float(wavelength[-1]), count + 1)

    usable = np.isfinite(depth) & np.isfinite(sigma) & (sigma > 0)
    for local in np.nonzero(~usable)[0]:
        removed.append(RemovedPoint(index=int(indices[local]), reason="unusable_for_rebin"))

    w_wave = wavelength[usable]
    w_depth = depth[usable]
    w_sigma = sigma[usable]
    w_indices = indices[usable]
    w_flags = [flag for flag, ok in zip(flags, usable, strict=True) if ok] if flags else []

    if w_wave.size == 0:
        raise ValueError("rebin: no points with usable weights")

    which = np.digitize(w_wave, grid[1:-1], right=False)
    n_bins = len(grid) - 1

    out_centers: list[float] = []
    out_depth: list[float] = []
    out_sigma: list[float] = []
    out_flags: list[str] = []
    out_input_indices: list[list[int]] = []
    empty_bin_centers: list[float] = []
    non_empty: list[int] = []

    for bin_index in range(n_bins):
        members = np.nonzero(which == bin_index)[0]
        if members.size == 0:
            empty_bin_centers.append(float((grid[bin_index] + grid[bin_index + 1]) / 2))
            continue
        weights = 1.0 / np.square(w_sigma[members])
        total_weight = float(np.sum(weights))
        non_empty.append(bin_index)
        out_centers.append(float((grid[bin_index] + grid[bin_index + 1]) / 2))
        out_depth.append(float(np.sum(weights * w_depth[members]) / total_weight))
        out_sigma.append(float(1.0 / np.sqrt(total_weight)))
        if w_flags:
            member_flags = [w_flags[i] for i in members]
            out_flags.append(next((str(flag) for flag in member_flags if str(flag)), ""))
        else:
            out_flags.append("")
        out_input_indices.append([int(w_indices[i]) for i in members])

    if not out_centers:
        raise ValueError("rebin produced no non-empty bins")

    # Edges: exact grid edges for adjacent retained bins, midpoints across gaps.
    out_edges: list[float] = []
    for position, bin_index in enumerate(non_empty):
        if position == 0:
            out_edges.append(float(grid[bin_index]))
        else:
            previous = non_empty[position - 1]
            if bin_index == previous + 1:
                out_edges.append(float(grid[bin_index]))
            else:
                mid = (out_centers[position - 1] + out_centers[position]) / 2
                out_edges.append(float(max(mid, float(grid[previous + 1]))))
    out_edges.append(float(grid[non_empty[-1] + 1]))

    if len(out_edges) != len(out_centers) + 1 or not bool(np.all(np.diff(out_edges) > 0)):
        out_edges = bin_edges_from_centers(
            np.asarray(out_centers),
            [float(grid[i + 1] - grid[i]) for i in non_empty],
        )

    rebin_log = RebinLog(
        mode=opts.rebin,
        n_input=int(wavelength.size),
        n_output=len(out_centers),
        grid_edges=[float(v) for v in grid],
        empty_bin_centers=empty_bin_centers,
        groups=[
            RebinGroup(output_index=i, input_indices=input_indices)
            for i, input_indices in enumerate(out_input_indices)
        ],
    )
    return (
        np.asarray(out_centers, dtype=np.float64),
        np.asarray(out_depth, dtype=np.float64),
        np.asarray(out_sigma, dtype=np.float64),
        out_flags,
        [float(v) for v in out_edges],
        rebin_log,
    )
