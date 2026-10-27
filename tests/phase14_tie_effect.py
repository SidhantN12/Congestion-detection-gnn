"""Phase 14: how much the midrank Spearman fix changes the number on real
corpus samples.

For each sample and channel, Spearman between a predictor and the true
congestion ratio, computed three ways:
  - midrank: metrics.spearman_corr (the current, tie-corrected definition)
  - old:     the pre-8c2f0e9 implementation, ranks from np.argsort (ties
             broken by whatever order argsort leaves them in)
  - random:  ties broken by a random permutation of the GCells, 50 draws;
             min and max reported (the spread an arbitrary tie-break can
             produce on this sample)

Predictors: RUDY (continuous, so ties are almost all on the target side)
and the capacity-only baseline fitted on the training designs of the
Gate 3 split (dynamic_node, tinyRocket; heavily tied on both sides).

Usage: python tests/phase14_tie_effect.py <graph.pt> ...
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines import CapacityBaseline, RUDYBaseline  # noqa: E402
from metrics import spearman_corr  # noqa: E402

TRAIN_DESIGNS = ("dynamic_node", "tinyRocket")


def old_rank(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def old_spearman(pred, target):
    return float(np.corrcoef(old_rank(pred), old_rank(target))[0, 1])


def random_tiebreak_spearman(pred, target, rng):
    perm = rng.permutation(len(pred))
    pred_perm = pred[perm]
    target_perm = target[perm]
    return old_spearman_stable(pred_perm, target_perm)


def old_spearman_stable(pred, target):
    pr = np.empty(len(pred))
    pr[np.argsort(pred, kind="stable")] = np.arange(len(pred))
    tr = np.empty(len(target))
    tr[np.argsort(target, kind="stable")] = np.arange(len(target))
    return float(np.corrcoef(pr, tr)[0, 1])


def main():
    paths = sys.argv[1:]
    manifest = json.load(open("data/processed/corpus/manifest.json"))
    train = [torch.load(m["path"], weights_only=False) for m in manifest if m["design"] in TRAIN_DESIGNS]
    cap = CapacityBaseline().fit(train)
    del train
    rudy = RUDYBaseline()
    rng = np.random.default_rng(0)

    print("| sample | predictor | ch | distinct target / pred | midrank | old argsort | old - midrank | random tie-break min..max | max abs diff vs midrank |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        g = torch.load(path, weights_only=False)
        name = Path(path).name.removesuffix("_graph.pt")
        target = g["gcell"].y_ratio.double().numpy()
        preds = dict(RUDY=rudy(g).detach().double().numpy(), capacity=cap(g).detach().double().numpy())
        for pname, pred in preds.items():
            for c, ch in enumerate(("H", "V")):
                p = pred[:, c]
                t = target[:, c]
                mid = spearman_corr(p, t)
                old = old_spearman(p, t)
                draws = [random_tiebreak_spearman(p, t, rng) for _ in range(50)]
                worst = max(abs(d - mid) for d in draws + [old])
                print(f"| {name} | {pname} | {ch} | {len(np.unique(t))} / {len(np.unique(p))} "
                      f"| {mid:.4f} | {old:.4f} | {old - mid:+.4f} | {min(draws):.4f}..{max(draws):.4f} | {worst:.4f} |")


if __name__ == "__main__":
    main()
