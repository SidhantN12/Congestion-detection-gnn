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


def _avg_rank(x):
    """Ranks 0..n-1 with tied values sharing their average rank (midranks).

    Real congestion maps are heavily tied (integer demand over a handful of
    capacity values: 163-199 distinct values over 7,225 GCells on
    dynamic_node, notes/phase14-real-corpus.md), so ties must not be
    broken by argsort order. Avoids adding scipy as a dependency.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    uniq = np.unique(x, return_inverse=True, return_counts=True)
    inv = uniq[1]
    counts = uniq[2]
    sums = np.bincount(inv, weights=r)
    return sums[inv] / counts[inv]


def spearman_corr(pred, target):
    """Spearman rank correlation between two 1D arrays: Pearson correlation
    of midranks, the standard tie-corrected definition. Before Phase 14
    ties were broken by argsort order, which made the value depend on the
    order of tied elements."""
    pred_rank = _avg_rank(np.asarray(pred).reshape(-1))
    target_rank = _avg_rank(np.asarray(target).reshape(-1))
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def hotspot_detection(pred, target, threshold=0.9):
    """Threshold-based hotspot detection for ONE channel (1D arrays).

    S = {gcells with target >= threshold}. The model's hotspot set is its
    top-|S| gcells by predicted value. Returns precision and recall of that
    set against S, and ROC-AUC of `pred` as a score for membership in S
    over the whole map.

    Well-defined under ties on both sides:
      - target ties (e.g. many gcells saturated at exactly 1.0) only
        decide membership in S, they are never ranked against each other;
      - pred ties straddling the top-|S| cutoff get fractional credit
        (the expected hit count over every ordering of the tied gcells),
        so the result never depends on argsort tie order;
      - ROC-AUC is the Mann-Whitney statistic with average ranks.

    With k = |S|, precision == recall by construction (both are
    hits / |S|); both are reported because the results table asks for
    them. Undefined cases (S empty, or S the whole map) return NaN.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    hot = target >= threshold
    k = int(hot.sum())
    n = len(target)
    out = dict(threshold=threshold, n_hotspots=k, n_gcells=n)
    if k == 0 or k == n:
        out.update(precision=float("nan"), recall=float("nan"), roc_auc=float("nan"))
        return out

    cutoff = np.sort(pred)[::-1][k - 1]
    above = pred > cutoff
    tied = pred == cutoff
    n_above = int(above.sum())
    hits = float((above & hot).sum())
    hits += (k - n_above) * float((tied & hot).sum()) / float(tied.sum())

    ranks = _avg_rank(pred) + 1.0
    n_neg = n - k
    auc = (ranks[hot].sum() - k * (k + 1) / 2.0) / (k * n_neg)

    out.update(precision=hits / k, recall=hits / k, roc_auc=float(auc))
    return out


def hotspot_detection_per_channel(pred, target, threshold=0.9):
    """hotspot_detection for each of the [horizontal, vertical] channels of
    (n, 2) arrays. Channels are reported separately, never averaged: under
    a band blockage the two have very different hotspot counts."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    return dict(h=hotspot_detection(pred[:, 0], target[:, 0], threshold),
                v=hotspot_detection(pred[:, 1], target[:, 1], threshold))


def kendall_tau(pred, target):
    """Kendall's tau-b rank correlation between two 1D arrays.

    O(n^2) pairwise concordance count - fine at gcell-grid scale (a few
    hundred points, so a few hundred thousand pairs). Avoids adding scipy
    as a dependency for something this small (same reasoning as
    spearman_corr above).
    """
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    n = len(pred)

    pred_diff = pred[:, None] - pred[None, :]
    target_diff = target[:, None] - target[None, :]
    iu = np.triu_indices(n, k=1)

    pred_sign = np.sign(pred_diff[iu])
    target_sign = np.sign(target_diff[iu])

    concordant = np.sum(pred_sign * target_sign > 0)
    discordant = np.sum(pred_sign * target_sign < 0)
    pred_ties = np.sum(pred_sign == 0)
    target_ties = np.sum(target_sign == 0)

    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - pred_ties) * (n0 - target_ties))
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)
