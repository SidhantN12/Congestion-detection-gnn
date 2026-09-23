"""Samples, design-level splits, and training units for src/train.py.

A *sample* is one placed graph; a *design* is a netlist. Several samples
can share a design (Phase 6's placement-seed sweep; synthetic `variants`).
Splits are made over designs, never over samples: samples of one design
share netlist topology (two synthetic placements of one 5K-cell design
had a 0.81 horizontal-label correlation, notes/phase13-training.md), so
splitting them across
train/val/test would leak and inflate every metric.

A *unit* is what one training micro-batch sees: either a whole sample
graph, or (partition.enabled) one Phase 11 spatial partition of it. Every
unit carries gcell `core` (the gcells it is supervised/evaluated on) and
`n_id` (their index in the full sample), so predictions can be stitched
back into full-design maps for evaluation.

Sources:
  synthetic - `data.synthetic.n_designs` designs, sizes drawn from
              [cells_min, cells_max], `variants` placements each.
  manifest  - a JSON list of {"path": graph.pt, "design": name} entries
              (e.g. real graphs from build_dataset.py later).
"""

import hashlib
import json
import os

import numpy as np
import torch

from partition import make_partitions, plan
from synthetic import generate_graph


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def build_samples(cfg, log=print):
    """List of {"design", "variant", "path"} for every sample, generating
    (and caching) synthetic graphs if needed."""
    if cfg.data.source == "manifest":
        with open(cfg.data.manifest) as f:
            entries = json.load(f)
        return [dict(design=str(e["design"]), variant=e.get("variant", i), path=e["path"])
                for i, e in enumerate(entries)]

    syn = cfg.data.synthetic
    root = os.path.join(cfg.data.cache_dir, "synthetic_" + _digest([cfg.seed, vars(syn)]))
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    sizes = rng.integers(syn.cells_min, syn.cells_max + 1, size=syn.n_designs)
    samples, made = [], 0
    for d, n_cells in enumerate(sizes):
        design = f"synth{d:03d}"
        for v in range(syn.variants):
            path = os.path.join(root, f"{design}_v{v}.pt")
            if not os.path.exists(path):
                g = generate_graph(int(n_cells), seed=cfg.seed * 10_007 + d,
                                   variant=v, variant_sigma=syn.variant_sigma)
                torch.save(g, path)
                made += 1
            samples.append(dict(design=design, variant=v, path=path, cells=int(n_cells)))
    log(f"synthetic dataset {root}: {syn.n_designs} designs x {syn.variants} variants "
        f"({made} generated, {len(samples) - made} cached)")
    return samples


def split_designs(samples, cfg):
    """{"train"|"val"|"test": sorted design names}; disjoint by construction."""
    designs = sorted({s["design"] for s in samples})
    s = cfg.split
    unknown = (set(s.val_designs) | set(s.test_designs)) - set(designs)
    if unknown:
        raise ValueError(f"split lists name unknown designs: {sorted(unknown)}")
    if s.val_designs or s.test_designs:
        val, test = sorted(s.val_designs), sorted(s.test_designs)
    else:
        order = [designs[i] for i in np.random.default_rng(cfg.seed).permutation(len(designs))]
        n_test = int(round(s.test_frac * len(designs))) if s.test_frac > 0 else 0
        n_val = int(round(s.val_frac * len(designs))) if s.val_frac > 0 else 0
        n_test = max(n_test, 1 if s.test_frac > 0 else 0)
        n_val = max(n_val, 1 if s.val_frac > 0 else 0)
        test, val = sorted(order[:n_test]), sorted(order[n_test:n_test + n_val])
    train = sorted(set(designs) - set(val) - set(test))
    if not train:
        raise ValueError(f"no designs left for training ({len(designs)} designs total)")
    split = dict(train=train, val=val, test=test)
    # A design in two splits would be exactly the leak this module exists to prevent.
    for a in split:
        for b in split:
            assert a == b or not set(split[a]) & set(split[b]), f"design in both {a} and {b}"
    return split


def samples_in(samples, designs):
    designs = set(designs)
    return [s for s in samples if s["design"] in designs]


def whole_graph_unit(g):
    n = g["gcell"].num_nodes
    g["gcell"].core = torch.ones(n, dtype=torch.bool)
    g["gcell"].n_id = torch.arange(n)
    return g


def unit_paths(sample, cfg, log=print):
    """Paths of the units for one sample (partitions cached on disk)."""
    if not cfg.partition.enabled:
        return [sample["path"]]
    p = cfg.partition
    key = _digest([sample["path"], os.path.getsize(sample["path"]), p.budget_gib, p.degree_cap,
                   cfg.model.num_layers])
    root = os.path.join(cfg.data.cache_dir, "parts", key)
    done = os.path.join(root, "DONE")
    if not os.path.exists(done):
        os.makedirs(root, exist_ok=True)
        g = torch.load(sample["path"], weights_only=False)
        k, _ = plan(g, p.budget_gib * 2 ** 30, num_hops=cfg.model.num_layers, degree_cap=p.degree_cap)
        for i, sub in make_partitions(g, k, k, num_hops=cfg.model.num_layers, degree_cap=p.degree_cap):
            torch.save(sub, os.path.join(root, f"part_{i:03d}.pt"))
        open(done, "w").close()
        log(f"  partitioned {os.path.basename(sample['path'])}: {k}x{k} grid")
    return sorted(os.path.join(root, f) for f in os.listdir(root) if f.startswith("part_"))


def load_unit(path, partitioned):
    g = torch.load(path, weights_only=False)
    return g if partitioned else whole_graph_unit(g)
