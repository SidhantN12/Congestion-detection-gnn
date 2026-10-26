"""Generate Gate 1 stub fixtures: synthetic designs sized close to the
real nangate45 corpus (15K-25K cells) that is being generated on another
machine and has not landed yet.

Truth is synthetic.py's RUDY-style proxy label (already attached to the
built graph as y_ratio) - a stand-in for router ground truth, not router
ground truth itself.

Prediction is deliberately NOT a model forward pass. A random-weight
model's output doesn't correlate with anything, which makes every Gate 2
animation (bloom-in, predicted/truth wipe, hotspot zoom) impossible to
judge - noise on one side of the wipe, structure on the other, no way to
tell whether the animation or the data is broken. Instead the fixture
prediction is the truth plus spatially-smoothed noise, with the noise
scale calibrated per channel so Spearman lands in [0.6, 0.8] - enough
structure to look and behave like a real prediction, not perfect enough
to be mistaken for one.

This is a fixture, not a measurement. The API must expose model_source
("fixture" here, "checkpoint" once a trained model exists) on every
response that carries it, and the frontend must never present the
fixture's Spearman as a model result.

Written to data/stub/, never data/processed/ - the real corpus lands in
data/processed/ and the Gate 4 trainer globs that directory, so stub and
real graphs must never share a location (a silent way to end up training
on synthetic fixtures).

Usage: python scripts/generate_stub_fixtures.py
"""

import json
import os
import sys

import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

from synthetic import generate_placement
from synthetic import knn_tiled
from build_graph import build_graph
from metrics import spearman_corr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB_DIR = os.path.join(ROOT, "data", "stub")

DESIGNS = [
    dict(id="synth_train_20k", cells=20000, seed=101, split="train"),
    dict(id="synth_val_25k", cells=25000, seed=102, split="val"),
    dict(id="synth_test_30k", cells=30000, seed=103, split="test"),
]

TARGET_SPEARMAN_LOW = 0.6
TARGET_SPEARMAN_HIGH = 0.8


def smooth_noise_field(rng, ny, nx, coarse_frac=0.12):
    """Spatially-correlated noise: coarse white noise, bilinearly
    upsampled to the full grid. Pure numpy (no scipy), separable via
    np.interp along each axis - gives smooth blobs rather than
    per-gcell-independent speckle.
    """
    coarse_ny = max(2, int(round(ny * coarse_frac)))
    coarse_nx = max(2, int(round(nx * coarse_frac)))
    coarse = rng.standard_normal((coarse_ny, coarse_nx))

    coarse_x = np.linspace(0, nx - 1, coarse_nx)
    fine_x = np.arange(nx, dtype=np.float64)
    stage_x = np.empty((coarse_ny, nx))
    for row in range(coarse_ny):
        stage_x[row] = np.interp(fine_x, coarse_x, coarse[row])

    coarse_y = np.linspace(0, ny - 1, coarse_ny)
    fine_y = np.arange(ny, dtype=np.float64)
    field = np.empty((ny, nx))
    for col in range(nx):
        field[:, col] = np.interp(fine_y, coarse_y, stage_x[:, col])

    field_std = field.std()
    if field_std == 0:
        field_std = 1.0
    return field / field_std


def calibrate_noise_scale(truth, noise, target_low, target_high):
    """Bisects the noise multiplier k so that
    Spearman(clip(truth + k*noise, 0, None), truth) lands in
    [target_low, target_high]. Spearman decreases monotonically with k
    in practice for additive noise of growing scale.
    """
    truth_std = float(truth.std())
    if truth_std == 0:
        truth_std = 1.0

    k_hi = truth_std * 0.25
    spearman_hi = spearman_corr(np.clip(truth + k_hi * noise, 0, None), truth)
    expand_steps = 0
    while spearman_hi > target_low and expand_steps < 30:
        k_hi *= 1.4
        spearman_hi = spearman_corr(np.clip(truth + k_hi * noise, 0, None), truth)
        expand_steps += 1

    k_lo = 0.0
    best_k = k_hi
    best_spearman = spearman_hi
    for _ in range(40):
        k_mid = (k_lo + k_hi) / 2.0
        fixture_mid = np.clip(truth + k_mid * noise, 0, None)
        spearman_mid = spearman_corr(fixture_mid, truth)
        best_k = k_mid
        best_spearman = spearman_mid
        if spearman_mid > target_high:
            k_lo = k_mid
        elif spearman_mid < target_low:
            k_hi = k_mid
        else:
            break
    return dict(scale=best_k, spearman=best_spearman)


def build_fixture_prediction(rng, truth_ratio, ny, nx):
    truth_h = truth_ratio[:, 0].reshape(ny, nx)
    truth_v = truth_ratio[:, 1].reshape(ny, nx)

    noise_h = smooth_noise_field(rng, ny, nx)
    noise_v = smooth_noise_field(rng, ny, nx)

    calibration_h = calibrate_noise_scale(truth_h, noise_h, TARGET_SPEARMAN_LOW, TARGET_SPEARMAN_HIGH)
    calibration_v = calibrate_noise_scale(truth_v, noise_v, TARGET_SPEARMAN_LOW, TARGET_SPEARMAN_HIGH)

    pred_h = np.clip(truth_h + calibration_h["scale"] * noise_h, 0, None)
    pred_v = np.clip(truth_v + calibration_v["scale"] * noise_v, 0, None)
    prediction = np.stack([pred_h.reshape(-1), pred_v.reshape(-1)], axis=1)
    achieved = dict(spearman_h=calibration_h["spearman"], spearman_v=calibration_v["spearman"])
    return dict(prediction=prediction, achieved=achieved)


def main():
    os.makedirs(STUB_DIR, exist_ok=True)
    manifest = []

    for spec in DESIGNS:
        fixture_rng = np.random.default_rng(spec["seed"] + 1_000_000)
        placement_and_congestion = generate_placement(spec["cells"], seed=spec["seed"])
        placement = placement_and_congestion[0]
        congestion = placement_and_congestion[1]
        data = build_graph(placement, congestion, knn_fn=knn_tiled)

        grid_shape = data["gcell"].grid_shape.numpy()
        ny = int(grid_shape[0])
        nx = int(grid_shape[1])
        truth_ratio = data["gcell"].y_ratio.numpy()

        fixture_result = build_fixture_prediction(fixture_rng, truth_ratio, ny, nx)
        achieved = fixture_result["achieved"]
        data["gcell"].y_pred_fixture = torch.tensor(fixture_result["prediction"], dtype=torch.float32)

        graph_path = os.path.join(STUB_DIR, f"{spec['id']}_graph.pt")
        torch.save(data, graph_path)

        n_gcells = ny * nx
        entry = dict(
            id=spec["id"],
            name=spec["id"],
            cell_count=spec["cells"],
            seed=spec["seed"],
            split=spec["split"],
            grid_shape=[ny, nx],
            n_gcells=n_gcells,
            die_w_dbu=int(placement["die_w"]),
            die_h_dbu=int(placement["die_h"]),
            macro_outlines=[],
            fixture_achieved_spearman=achieved,
        )
        manifest.append(entry)
        print(
            f"{spec['id']}: {spec['cells']} cells, grid {ny}x{nx} ({n_gcells} gcells), "
            f"fixture Spearman H={achieved['spearman_h']:.3f} V={achieved['spearman_v']:.3f}"
        )

    manifest_path = os.path.join(STUB_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest for {len(manifest)} designs to {manifest_path}")


if __name__ == "__main__":
    main()
