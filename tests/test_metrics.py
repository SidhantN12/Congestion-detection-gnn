"""hotspot_detection against brute-force references (Phase 14).

  [1] ROC-AUC == pairwise P(pos > neg) + 0.5 P(pos == neg), with heavy ties
  [2] tie-aware precision == mean over 2000 random orderings of tied preds
  [3] perfect / reversed / constant predictors -> 1 / 0 / chance
  [4] saturated-ceiling labels (many targets exactly 1.0) stay well-defined
  [5] empty hotspot set -> NaN
  [6] spearman_corr == Pearson of naive textbook midranks on a heavily tied
      vector (7,225 values, ~200 distinct, like a real dynamic_node map);
      exact hand-worked case 3/sqrt(10); invariant to the order of tied
      elements
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import hotspot_detection, spearman_corr  # noqa: E402


def brute_auc(pred, target, thr):
    pos = pred[target >= thr]
    neg = pred[target < thr]
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def sampled_precision(pred, target, thr, rng, reps=2000):
    hot = target >= thr
    k = int(hot.sum())
    vals = []
    for _ in range(reps):
        jitter = rng.random(len(pred))
        order = np.lexsort((jitter, -pred))
        vals.append(hot[order[:k]].sum() / k)
    return float(np.mean(vals))


def naive_midrank(x):
    """Textbook midrank, 1-based: #(x_j < x_i) + (#(x_j == x_i) + 1) / 2.
    O(n^2) and written independently of metrics._avg_rank."""
    x = np.asarray(x, dtype=np.float64)
    less = (x[None, :] < x[:, None]).sum(axis=1)
    equal = (x[None, :] == x[:, None]).sum(axis=1)
    return less + (equal + 1) / 2.0


def naive_spearman(a, b):
    ra = naive_midrank(a)
    rb = naive_midrank(b)
    da = ra - ra.mean()
    db = rb - rb.mean()
    return float((da * db).sum() / np.sqrt((da * da).sum() * (db * db).sum()))


def main():
    rng = np.random.default_rng(0)
    target = np.round(rng.random(600) * 12) / 12          # 13 distinct values
    target[rng.random(600) < 0.1] = 1.0                    # saturated ceiling
    pred = np.round(target + rng.normal(0, 0.15, 600), 1)  # coarse -> many ties

    r = hotspot_detection(pred, target)
    auc = brute_auc(pred, target, 0.9)
    assert abs(r["roc_auc"] - auc) < 1e-12, (r["roc_auc"], auc)
    print(f"[1] auc {r['roc_auc']:.6f} == brute {auc:.6f}")

    sp = sampled_precision(pred, target, 0.9, rng)
    assert abs(r["precision"] - sp) < 0.01, (r["precision"], sp)
    assert r["precision"] == r["recall"]
    print(f"[2] tie-aware precision {r['precision']:.4f} vs sampled {sp:.4f}; recall == precision")

    perfect = hotspot_detection(target, target)
    reverse = hotspot_detection(-target, target)
    const = hotspot_detection(np.zeros(600), target)
    assert perfect["precision"] == 1.0 and perfect["roc_auc"] == 1.0
    assert reverse["precision"] == 0.0 and reverse["roc_auc"] == 0.0
    assert abs(const["precision"] - const["n_hotspots"] / 600) < 1e-12 and const["roc_auc"] == 0.5
    print(f"[3] perfect {perfect['precision']}/{perfect['roc_auc']}  reversed {reverse['precision']}/"
          f"{reverse['roc_auc']}  constant {const['precision']:.4f}/{const['roc_auc']}")

    sat = np.where(rng.random(600) < 0.3, 1.0, rng.random(600) * 0.8)
    s = hotspot_detection(sat + rng.normal(0, 0.05, 600), sat)
    assert np.isfinite(s["roc_auc"]) and 0 <= s["precision"] <= 1
    print(f"[4] {s['n_hotspots']} hotspots all at 1.0: precision {s['precision']:.4f} auc {s['roc_auc']:.4f}")

    e = hotspot_detection(pred, np.zeros(600))
    assert np.isnan(e["precision"]) and np.isnan(e["roc_auc"])
    print("[5] empty hotspot set -> NaN")

    hand = spearman_corr([1, 2, 2, 3], [1, 2, 3, 4])
    assert abs(hand - 3 / np.sqrt(10)) < 1e-12, hand
    n = 7225
    cap = rng.choice([12.0, 14.0, 15.0, 17.0, 18.0, 20.0], size=n)
    use = np.minimum(rng.poisson(cap * 0.6), cap)
    y = use / cap                                  # integer demand / 6 capacities
    p_tied = np.round(y + rng.normal(0, 0.1, n), 2)  # tied predictions too
    print(f"    tied vector: {len(np.unique(y))} distinct targets, "
          f"{len(np.unique(p_tied))} distinct preds over {n}")
    got = spearman_corr(p_tied, y)
    ref = naive_spearman(p_tied, y)
    assert abs(got - ref) < 1e-12, (got, ref)
    perm = rng.permutation(n)
    shuffled = spearman_corr(p_tied[perm], y[perm])
    assert abs(shuffled - got) < 1e-12, (shuffled, got)
    # the tie-breaking order inside a block of equal targets must not matter:
    # reverse the order of tied targets only (stable sort keys differ)
    order = np.lexsort((-np.arange(n), y))
    assert abs(spearman_corr(p_tied[order], y[order]) - got) < 1e-12
    print(f"[6] spearman hand case {hand:.6f} == 3/sqrt(10); tied vector {got:.6f} == naive midrank {ref:.6f}; "
          f"order-invariant")
    print("PASS")


if __name__ == "__main__":
    main()
