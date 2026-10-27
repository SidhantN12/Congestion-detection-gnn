"""Design manifest, graph cache, and the derived quantities (prediction,
truth, metrics, hotspots) every route in main.py reads.

Latency honesty: inference_seconds is a genuine measurement - the median
of LATENCY_BENCHMARK_RUNS timed forward passes through the loaded model,
after one untimed warm-up call (so first-call cold cost, which Gate 1
found dominated single-shot timings and made per-design latencies
overlap noisily, doesn't leak into the reported figure). It's real
regardless of whether the weights are trained, since it's the same
architecture doing the same arithmetic on the same graph. Benchmarked
once per design and cached (benchmark_latency), not re-run on every
request - a demo badge needs one stable number, not a fresh 20-run
benchmark per click. What differs by model_source is only which values
get *shown* as the prediction: in "checkpoint" mode the forward pass's
own output is the prediction; in "fixture" mode the displayed congestion
values are the pre-baked fixture array from
scripts/generate_stub_fixtures.py, since an untrained model's raw output
carries no structure (see that script's docstring for why that matters
for the frontend animations) - but inference_seconds still reflects the
real, benchmarked cost of that architecture's forward pass.
"""

import json
import os
import time

import numpy as np
import torch

from metrics import kendall_tau
from metrics import spearman_corr
from metrics import hotspot_detection

KENDALL_EXACT_MAX_GCELLS = 14000
KENDALL_SAMPLE_SIZE = 2000
KENDALL_SAMPLE_SEED = 0

LATENCY_WARMUP_RUNS = 1
LATENCY_BENCHMARK_RUNS = 25


def select_kendall(pred_channel, truth_channel, kendall_tau_fn=kendall_tau):
    """Exact Kendall's tau under KENDALL_EXACT_MAX_GCELLS gcells, falling
    back to a fixed-seed KENDALL_SAMPLE_SIZE-gcell sample above that
    threshold or if the exact O(n^2) computation raises MemoryError even
    under it (Phase 12 measured the OOM boundary at ~15,376 gcells, so
    14,000 leaves headroom, but the try/except is the real guarantee,
    not the threshold alone). A standalone function so the fallback path
    is testable without needing an actual 14,000+ gcell graph.
    """
    n_gcells = pred_channel.shape[0]
    kendall = None
    kendall_sampled = None
    if n_gcells <= KENDALL_EXACT_MAX_GCELLS:
        try:
            kendall = kendall_tau_fn(pred_channel, truth_channel)
            kendall_sampled = False
        except MemoryError:
            kendall = None
    if kendall is None:
        sample_rng = np.random.default_rng(KENDALL_SAMPLE_SEED)
        sample_size = min(KENDALL_SAMPLE_SIZE, n_gcells)
        sample_idx = sample_rng.choice(n_gcells, size=sample_size, replace=False)
        kendall = kendall_tau_fn(pred_channel[sample_idx], truth_channel[sample_idx])
        kendall_sampled = True
    return dict(kendall=kendall, kendall_sampled=kendall_sampled)


class DesignStore:
    def __init__(self, stub_dir, processed_dir, model, model_source, manifest_path=None):
        self.stub_dir = stub_dir
        self.processed_dir = processed_dir
        self.model = model
        self.model_source = model_source
        self.manifest_path = manifest_path or os.path.join(stub_dir, "manifest.json")
        self.manifest = self._load_manifest()
        self.graph_cache = {}
        self.latency_cache = {}

    def _load_manifest(self):
        manifest_path = self.manifest_path
        with open(manifest_path) as manifest_file:
            entries = json.load(manifest_file)
        by_id = {}
        for entry in entries:
            by_id[entry["id"]] = entry
        return by_id

    def list_designs(self):
        return list(self.manifest.values())

    def get_entry(self, design_id):
        return self.manifest.get(design_id)

    def get_graph(self, design_id):
        if design_id in self.graph_cache:
            return self.graph_cache[design_id]
        graph_path = os.path.join(self.stub_dir, f"{design_id}_graph.pt")
        data = torch.load(graph_path, weights_only=False)
        self.graph_cache[design_id] = data
        return data

    def truth_maps(self, design_id):
        data = self.get_graph(design_id)
        ratio = data["gcell"].y_ratio.numpy()
        horizontal = ratio[:, 0].tolist()
        vertical = ratio[:, 1].tolist()
        return dict(horizontal=horizontal, vertical=vertical)

    def benchmark_latency(self, design_id):
        """Median forward-pass latency over LATENCY_BENCHMARK_RUNS timed
        runs, after LATENCY_WARMUP_RUNS untimed warm-up calls. Computed
        once per design per process and cached - see the module
        docstring for why a median replaces the earlier single-shot
        timing.
        """
        if design_id in self.latency_cache:
            return self.latency_cache[design_id]

        data = self.get_graph(design_id)

        for _ in range(LATENCY_WARMUP_RUNS):
            with torch.no_grad():
                self.model(data)

        samples = []
        for _ in range(LATENCY_BENCHMARK_RUNS):
            start = time.perf_counter()
            with torch.no_grad():
                self.model(data)
            elapsed = time.perf_counter() - start
            samples.append(elapsed)

        median_latency = float(np.median(samples))
        self.latency_cache[design_id] = median_latency
        return median_latency

    def prediction_maps(self, design_id):
        """Returns dict(horizontal=, vertical=, inference_seconds=). See
        the module docstring for how inference_seconds is measured.
        """
        data = self.get_graph(design_id)
        latency = self.benchmark_latency(design_id)

        if self.model_source == "checkpoint":
            with torch.no_grad():
                prediction = self.model(data).numpy()
        else:
            prediction = data["gcell"].y_pred_fixture.numpy()

        horizontal = prediction[:, 0].tolist()
        vertical = prediction[:, 1].tolist()
        return dict(horizontal=horizontal, vertical=vertical, inference_seconds=latency)

    def metrics(self, design_id):
        data = self.get_graph(design_id)
        truth = data["gcell"].y_ratio.numpy()
        prediction_result = self.prediction_maps(design_id)
        prediction_horizontal = np.array(prediction_result["horizontal"])
        prediction_vertical = np.array(prediction_result["vertical"])
        prediction = np.stack([prediction_horizontal, prediction_vertical], axis=1)

        channel_names = ["horizontal", "vertical"]
        result = {}
        for channel_index in range(len(channel_names)):
            channel_name = channel_names[channel_index]
            pred_channel = prediction[:, channel_index]
            truth_channel = truth[:, channel_index]

            spearman = spearman_corr(pred_channel, truth_channel)
            mae = float(np.mean(np.abs(pred_channel - truth_channel)))
            rmse = float(np.sqrt(np.mean((pred_channel - truth_channel) ** 2)))

            kendall_result = select_kendall(pred_channel, truth_channel)
            hotspot = hotspot_detection(pred_channel, truth_channel)

            result[channel_name] = dict(
                spearman=spearman,
                kendall=kendall_result["kendall"],
                kendall_sampled=kendall_result["kendall_sampled"],
                mae=mae,
                rmse=rmse,
                hotspot_precision=hotspot["precision"],
                hotspot_recall=hotspot["recall"],
                hotspot_auc=hotspot["roc_auc"],
            )
        return result

    def hotspots(self, design_id, k):
        data = self.get_graph(design_id)
        truth = data["gcell"].y_ratio.numpy()
        prediction_result = self.prediction_maps(design_id)
        prediction_horizontal = np.array(prediction_result["horizontal"])
        prediction_vertical = np.array(prediction_result["vertical"])
        prediction = np.stack([prediction_horizontal, prediction_vertical], axis=1)

        grid_shape = data["gcell"].grid_shape.numpy()
        nx = int(grid_shape[1])

        truth_combined = np.maximum(truth[:, 0], truth[:, 1])
        pred_combined = np.maximum(prediction[:, 0], prediction[:, 1])

        truth_order = np.argsort(-truth_combined)
        pred_order = np.argsort(-pred_combined)
        true_rank_by_index = np.empty_like(truth_order)
        true_rank_by_index[truth_order] = np.arange(len(truth_order))
        pred_rank_by_index = np.empty_like(pred_order)
        pred_rank_by_index[pred_order] = np.arange(len(pred_order))

        top_indices = truth_order[:k]
        rows = []
        for gcell_index in top_indices:
            gcell_index = int(gcell_index)
            y_coord = gcell_index // nx
            x_coord = gcell_index % nx
            true_rank = int(true_rank_by_index[gcell_index]) + 1
            predicted_rank = int(pred_rank_by_index[gcell_index]) + 1
            row = dict(
                gcell_index=gcell_index,
                x=x_coord,
                y=y_coord,
                predicted=float(pred_combined[gcell_index]),
                true=float(truth_combined[gcell_index]),
                true_rank=true_rank,
                predicted_rank=predicted_rank,
                rank_delta=predicted_rank - true_rank,
            )
            rows.append(row)
        return rows
