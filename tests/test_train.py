"""Phase 13 checks for the config-driven trainer (src/train.py, config.py,
dataset.py). CPU, ~1-2 minutes, writes only to a temp directory.

1. Config: unknown keys / bad types rejected; --set overrides parsed as
   TOML; the hash ignores train.epochs but nothing else.
2. Design-level split: every design in exactly one split, every sample of
   a design in its design's split; explicit lists honoured.
3. sampled_ranking_loss matches the full O(n^2) pairwise_ranking_loss
   when enough pairs are sampled.
4. Resume is exact: 2 epochs + resume to 4 gives bit-identical weights and
   identical val metrics to 4 uninterrupted epochs (with partitions and
   gradient accumulation on).
5. Resume refuses a checkpoint from a different config.
6. Gradient accumulation: optimizer steps per epoch = ceil(units / accum).
7. Early stopping fires on val Spearman: with lr ~ 0 nothing improves after
   epoch 0, so the run stops after `patience` more epochs.
8. Mixed precision degrades gracefully on CPU: auto/fp16 -> fp32 with a
   reason; bf16 is probed and either used or refused, never a crash.

Usage:
    python tests/test_train.py
"""

import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, "src")
from config import load_config
from dataset import build_samples, split_designs
from train import pairwise_ranking_loss, resolve_amp, run, sampled_ranking_loss

TMP = tempfile.mkdtemp(prefix="phase13_test_")
BASE = [
    f'out_dir="{TMP}/runs"', f'data.cache_dir="{TMP}/cache"',
    "data.synthetic.n_designs=5", "data.synthetic.cells_min=1500", "data.synthetic.cells_max=2500",
    "data.synthetic.variants=2", "split.val_frac=0.2", "split.test_frac=0.2",
    "partition.enabled=true", "partition.budget_gib=0.185", "train.accum_steps=3",
    "train.rank_pairs=5000", "early_stopping.patience=10",
]


def cfg(*extra):
    return load_config(None, BASE + list(extra))


def read_log(name):
    with open(f"{TMP}/runs/{name}/metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def check_config():
    for bad in (["train.epochz=1"], ['train.amp="fp8"'], ['train.lr="x"'], ["split.val_frac=0.9"]):
        try:
            load_config(None, bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    c = load_config(None, ['split.val_designs=["a"]', "train.lr=3e-4", "partition.enabled=true"])
    assert c.split.val_designs == ["a"] and c.train.lr == 3e-4 and c.partition.enabled is True
    assert cfg("train.epochs=2").hash() == cfg("train.epochs=9").hash()
    assert cfg("train.lr=0.1").hash() != cfg("train.lr=0.2").hash()
    print("[1] config: bad keys/types/choices rejected, overrides parsed, hash ignores only train.epochs")


def check_split():
    c = cfg()
    samples = build_samples(c, log=lambda *_: None)
    split = split_designs(samples, c)
    where = {d: k for k, v in split.items() for d in v}
    assert sorted(where) == sorted({s["design"] for s in samples})
    assert all(sum(d in v for v in split.values()) == 1 for d in where)
    c2 = cfg('split.val_designs=["synth001"]', 'split.test_designs=["synth004"]')
    s2 = split_designs(samples, c2)
    assert s2["val"] == ["synth001"] and s2["test"] == ["synth004"] and len(s2["train"]) == 3
    print(f"[2] split by design: {split}; each of {len(samples)} samples follows its design")


def check_sampled_rank():
    g = torch.Generator().manual_seed(0)
    pred, target = torch.randn(300, 2, generator=g), torch.randn(300, 2, generator=g)
    full = pairwise_ranking_loss(pred, target).item()
    est = sampled_ranking_loss(pred, target, 2_000_000, torch.Generator().manual_seed(1)).item()
    assert abs(est - full) / full < 0.01
    assert sampled_ranking_loss(pred, target, 0, g).item() == 0
    print(f"[3] sampled ranking loss {est:.5f} vs full O(n^2) {full:.5f} (rel diff {abs(est - full) / full:.2e})")


def check_resume():
    run(cfg("train.epochs=4", 'name="straight"'))
    run(cfg("train.epochs=2", 'name="resumed"'))
    run(cfg("train.epochs=4", 'name="resumed"'), resume="auto")
    a = torch.load(f"{TMP}/runs/straight/last.pt", weights_only=False)
    b = torch.load(f"{TMP}/runs/resumed/last.pt", weights_only=False)
    same_w = all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    same_opt = all(torch.equal(x, y) for sa, sb in zip(a["optimizer"]["state"].values(),
                                                         b["optimizer"]["state"].values())
                   for x, y in zip(sa.values(), sb.values()) if torch.is_tensor(x))
    va = [r["val"] for r in read_log("straight") if r["type"] == "epoch"]
    vb = [r["val"] for r in read_log("resumed") if r["type"] == "epoch"]
    assert same_w and same_opt and va == vb, (same_w, same_opt)
    print(f"[4] 2 epochs + resume to 4 == 4 straight: weights bit-identical, Adam state identical, "
          f"val metrics identical for all 4 epochs (last val spearman {va[-1]['spearman']:.6f})")
    return read_log("straight")


def check_resume_refuses():
    try:
        run(cfg("train.epochs=4", 'name="resumed"', "train.lr=0.5"), resume="auto")
        raise AssertionError("resumed a different config")
    except ValueError as e:
        print(f"[5] resume with changed train.lr refused: {str(e)[:90]}...")


def check_accum(log):
    start = next(r for r in log if r["type"] == "start")
    epochs = [r for r in log if r["type"] == "epoch"]
    units = start["n_train_units"]
    per_epoch = [b["optimizer_steps"] - a for a, b in zip([0] + [e["optimizer_steps"] for e in epochs], epochs)]
    assert all(s == math.ceil(units / 3) for s in per_epoch)
    print(f"[6] accum_steps=3 over {units} units: {per_epoch[0]} optimizer steps/epoch (= ceil({units}/3))")


def check_early_stop():
    run(cfg("train.epochs=20", 'name="frozen"', "train.lr=1e-30", "early_stopping.patience=2"))
    log = read_log("frozen")
    stop = [r for r in log if r["type"] == "early_stop"]
    epochs = [r for r in log if r["type"] == "epoch"]
    assert stop and stop[0]["epoch"] == 2 and len(epochs) == 3 and stop[0]["best_epoch"] == 0
    print(f"[7] lr=1e-30, patience=2: early stop at epoch {stop[0]['epoch']} after {len(epochs)} epochs "
          f"(val spearman stayed {epochs[0]['val']['spearman']:.6f})")


def check_amp():
    from model import CongestionGNN
    from synthetic import generate_graph
    g = generate_graph(1500, seed=0)
    m = CongestionGNN({nt: g[nt].x.shape[1] for nt in g.node_types}, list(g.edge_types))
    cpu = torch.device("cpu")
    for amp in ("auto", "off", "fp16", "bf16"):
        dtype, scaler, reason = resolve_amp(load_config(None, [f'train.amp="{amp}"']), cpu, m, g, print)
        assert amp == "bf16" or dtype is None
        print(f"[8] amp={amp}: {reason}")
    if dtype is not None:
        run(cfg("train.epochs=1", 'name="bf16"', 'train.amp="bf16"'))
        rec = [r for r in read_log("bf16") if r["type"] == "epoch"][0]
        assert math.isfinite(rec["train"]["loss"]) and math.isfinite(rec["val"]["spearman"])
        print(f"[8] bf16 epoch on cpu: train loss {rec['train']['loss']:.4f}, "
              f"val spearman {rec['val']['spearman']:.4f} (finite)")


def main():
    check_config()
    check_split()
    check_sampled_rank()
    log = check_resume()
    check_resume_refuses()
    check_accum(log)
    check_early_stop()
    check_amp()
    print(f"\nPASS  (artifacts in {TMP})")


if __name__ == "__main__":
    main()
