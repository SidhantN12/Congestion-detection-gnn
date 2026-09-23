"""Declarative experiment configs: TOML files (read with the standard
library's tomllib - no new dependency) mapped onto typed dataclasses.

- Every key has a default here; a config file only states what differs.
- Unknown keys and wrong types are errors, not silently ignored - a typo in
  an experiment file must not quietly run a different experiment.
- `--set section.key=value` overrides (value parsed as TOML, so
  `train.lr=3e-4`, `train.amp="bf16"`, `split.val_designs=["d1"]`).
- The fully resolved config is written into each run directory and hashed;
  the hash is stored in every checkpoint so a resume can refuse a changed
  experiment.

Usage:
    cfg = load_config("configs/synthetic_smoke.toml", overrides=["train.epochs=3"])
"""

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from typing import List


@dataclass
class SyntheticData:
    n_designs: int = 8
    cells_min: int = 4000
    cells_max: int = 8000
    variants: int = 3           # placements per design (same netlist)
    variant_sigma: float = 2.0  # displacement sigma, gcell pitches


@dataclass
class DataConfig:
    source: str = "synthetic"   # "synthetic" | "manifest"
    manifest: str = ""          # JSON list of {"path": ..., "design": ...} for source="manifest"
    cache_dir: str = "data/processed/datasets"
    synthetic: SyntheticData = field(default_factory=SyntheticData)


@dataclass
class SplitConfig:
    val_frac: float = 0.25
    test_frac: float = 0.125
    val_designs: List[str] = field(default_factory=list)   # explicit lists override fractions
    test_designs: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 3


@dataclass
class PartitionConfig:
    enabled: bool = False
    budget_gib: float = 2.0     # planner target per partition (see partition.plan)
    degree_cap: int = 512


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    accum_steps: int = 1        # micro-batches (one graph/partition each) per optimizer step
    lambda_rank: float = 0.5
    rank_pairs: int = 20000     # sampled gcell pairs per micro-batch for the ranking loss
    grad_clip: float = 0.0      # 0 = off
    amp: str = "auto"           # "auto" | "off" | "bf16" | "fp16"
    device: str = "auto"        # "auto" | "cpu" | "cuda"
    # 1 intra-op thread + torch deterministic algorithms. Measured: with 4 CPU
    # threads, identical runs in separate processes drift (max weight diff
    # 1.9e-4 after 1 epoch); with this on they are bit-identical, ~45% slower.
    deterministic: bool = True


@dataclass
class EarlyStoppingConfig:
    patience: int = 5           # epochs without val Spearman improvement before stopping
    min_delta: float = 0.0


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def to_dict(self):
        return dataclasses.asdict(self)

    def hash(self, exclude=("train.epochs",)):
        """Stable hash of the experiment definition. train.epochs is excluded
        by default so a run can be resumed with a longer schedule."""
        d = self.to_dict()
        for key in exclude:
            *path, last = key.split(".")
            node = d
            for p in path:
                node = node[p]
            node.pop(last, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


CHOICES = {
    ("data", "source"): {"synthetic", "manifest"},
    ("train", "amp"): {"auto", "off", "bf16", "fp16"},
    ("train", "device"): {"auto", "cpu", "cuda"},
}


def _build(cls, values, path):
    if not isinstance(values, dict):
        raise ValueError(f"[{'.'.join(path)}] must be a table, got {type(values).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown config key(s) in [{'.'.join(path) or 'top level'}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        f = fields[name]
        default = getattr(cls(), name)
        key = ".".join(path + [name])
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, path + [name])
            continue
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        elif isinstance(default, int):
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, str):
            ok = isinstance(value, str)
        elif isinstance(default, list):
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        else:
            ok = True
        if not ok:
            raise ValueError(f"{key}: expected {type(default).__name__}, got {value!r}")
        allowed = CHOICES.get((path[-1] if path else "", name))
        if allowed and value not in allowed:
            raise ValueError(f"{key}: must be one of {sorted(allowed)}, got {value!r}")
        kwargs[name] = value
    return cls(**kwargs)


def _apply_override(raw, override):
    if "=" not in override:
        raise ValueError(f"override must be section.key=value, got {override!r}")
    key, value = override.split("=", 1)
    parsed = tomllib.loads(f"v = {value}")["v"] if value.strip() else ""
    *path, last = key.strip().split(".")
    node = raw
    for p in path:
        node = node.setdefault(p, {})
    node[last] = parsed


def load_config(path=None, overrides=()):
    raw = {}
    if path:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    for o in overrides:
        _apply_override(raw, o)
    cfg = _build(Config, raw, [])
    validate(cfg)
    return cfg


def validate(cfg):
    t, s, d = cfg.train, cfg.split, cfg.data.synthetic
    checks = [
        (t.epochs >= 1, "train.epochs must be >= 1"),
        (t.accum_steps >= 1, "train.accum_steps must be >= 1"),
        (t.lr > 0, "train.lr must be > 0"),
        (t.rank_pairs >= 0, "train.rank_pairs must be >= 0"),
        (0 <= s.val_frac < 1 and 0 <= s.test_frac < 1 and s.val_frac + s.test_frac < 1,
         "split fractions must be in [0,1) and sum to < 1"),
        (not set(s.val_designs) & set(s.test_designs), "a design cannot be in both val and test"),
        (cfg.data.source != "manifest" or cfg.data.manifest, "data.manifest is required for source='manifest'"),
        (d.cells_min <= d.cells_max and d.n_designs >= 1 and d.variants >= 1, "invalid [data.synthetic] sizes"),
        (cfg.early_stopping.patience >= 1, "early_stopping.patience must be >= 1"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)
