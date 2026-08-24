"""Shared congestion metrics, reused across Phase 3 validation, Phase 6
dataset stats, and Phase 7/8 evaluation.
"""

import numpy as np


def combined_congestion(ratio):
    """Per-GCell congestion combining horizontal/vertical channels.

    Matches OpenROAD's own GUI heatmap convention in its default
    all-directions mode (see notes/phase2-congestion-extraction.md):
    max(horizontal_ratio, vertical_ratio) per GCell.

    `ratio` is (ny, nx, 2) as produced by extract_congestion.py.
    Returns (ny, nx).
    """
    return np.maximum(ratio[:, :, 0], ratio[:, :, 1])


def fraction_above(ratio, threshold):
    """Fraction of GCells whose combined congestion exceeds `threshold`."""
    combined = combined_congestion(ratio)
    return float(np.count_nonzero(combined > threshold)) / combined.size


def _rank(x):
    """Rank-transform a 1D array (ties broken arbitrarily but stably -
    fine here since congestion ratios are floats, ties are ~never exact).
    Avoids adding scipy as a dependency for something this small.
    """
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(pred, target):
    """Spearman rank correlation between two 1D arrays."""
    pred_rank = _rank(np.asarray(pred).reshape(-1))
    target_rank = _rank(np.asarray(target).reshape(-1))
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])
