"""Data drift detection via KL-divergence between a feature's training-time baseline
distribution and its distribution over a recent window of live traffic, per the PRD's
`data_drift_metrics` schema and success metric ("data drift detection using KL-divergence")."""

import numpy as np

EPSILON = 1e-6
DEFAULT_BINS = 10


def histogram(values: list[float], *, bin_edges: list[float] | None = None, bins: int = DEFAULT_BINS) -> dict:
    """Returns {"bin_edges": [...], "density": [...]} - a normalized probability distribution
    over `bins` buckets, reusable as either a baseline (fit) or a comparison (reuse edges)."""

    arr = np.asarray(values, dtype=float)
    if bin_edges is None:
        counts, edges = np.histogram(arr, bins=bins)
    else:
        counts, edges = np.histogram(arr, bins=bin_edges)
    density = counts / max(counts.sum(), 1)
    return {"bin_edges": edges.tolist(), "density": density.tolist()}


def open_outer_edges(bin_edges: list[float]) -> list[float]:
    """Replaces a histogram's outermost edges with +-inf, so binning a new set of values
    against them can never silently drop out-of-range values into nowhere (np.histogram
    otherwise excludes anything outside [edges[0], edges[-1]], which would make a value that
    drifted completely outside the training-time range look like *no* drift instead of extreme
    drift)."""

    return [-np.inf, *bin_edges[1:-1], np.inf]


def kl_divergence(expected: dict, actual_values: list[float]) -> float:
    """KL(actual || expected): how surprised the expected (baseline/training) distribution
    would be by the actual (live) distribution. Both smoothed with epsilon to avoid div-by-zero
    on empty bins."""

    expected_density = np.asarray(expected["density"], dtype=float)
    actual = histogram(actual_values, bin_edges=open_outer_edges(expected["bin_edges"]))
    actual_density = np.asarray(actual["density"], dtype=float)

    p = actual_density + EPSILON
    q = expected_density + EPSILON
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))
