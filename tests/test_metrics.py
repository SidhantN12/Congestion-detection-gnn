"""hotspot_detection against brute-force references (Phase 14).

  [1] ROC-AUC == pairwise P(pos > neg) + 0.5 P(pos == neg), with heavy ties
  [2] tie-aware precision == mean over 2000 random orderings of tied preds
  [3] perfect / reversed / constant predictors -> 1 / 0 / chance
  [4] saturated-ceiling labels (many targets exactly 1.0) stay well-defined
  [5] empty hotspot set -> NaN
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metrics import hotspot_detection  # noqa: E402


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
    print("PASS")


if __name__ == "__main__":
    main()
