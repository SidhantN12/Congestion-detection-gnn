"""Phase 14 Gate 3: the design-level split on the real corpus manifest, and
both places that reject a design listed in two splits.

  [1] the corpus_smoke split: designs and sample counts per split, disjoint
  [2] config validation rejects val_designs/test_designs overlap
  [3] split_designs itself asserts if an overlap gets past validation

Usage: python tests/phase14_split_check.py [configs/corpus_smoke.toml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config  # noqa: E402
from dataset import build_samples, samples_in, split_designs  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/corpus_smoke.toml"
    cfg = load_config(path)
    samples = build_samples(cfg)
    split = split_designs(samples, cfg)
    print("[1] split from", path)
    for name, designs in split.items():
        variants = sorted(s["variant"] for s in samples_in(samples, designs))
        print(f"    {name:5s} designs={designs} samples={len(variants)} {variants}")
    seen = {}
    for name, designs in split.items():
        for i, s in enumerate(samples):
            if s in samples_in(samples, designs):
                seen.setdefault(i, []).append(name)
    multi = [p for p, names in seen.items() if len(names) > 1]
    print(f"    {len(seen)} of {len(samples)} samples assigned, {len(multi)} in more than one split")
    assert len(seen) == len(samples) and not multi

    overlap = cfg.split.val_designs[0]
    try:
        load_config(path, [f'split.test_designs=["{overlap}"]'])
    except ValueError as e:
        print(f"[2] load_config with {overlap} in val and test -> ValueError: {e}")
    else:
        raise AssertionError("config validation accepted a design in both val and test")

    cfg.split.test_designs = [overlap]
    try:
        split_designs(samples, cfg)
    except AssertionError as e:
        print(f"[3] split_designs with {overlap} in val and test -> AssertionError: {e}")
    else:
        raise AssertionError("split_designs accepted a design in both val and test")
    print("PASS")


if __name__ == "__main__":
    main()
