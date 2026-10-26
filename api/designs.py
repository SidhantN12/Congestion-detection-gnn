"""Design manifest, graph cache, and the derived quantities (prediction,
truth, metrics, hotspots) every route in main.py reads.

Latency honesty: prediction_maps() always runs a real forward pass
through the loaded model and times it with perf_counter - that cost is
real regardless of whether the weights are trained, since it's the same
architecture doing the same arithmetic on the same graph. What differs
by model_source is which numbers get shown: in "checkpoint" mode the
forward pass's own output is the prediction; in "fixture" mode (no
checkpoint yet) the forward pass still runs (so inference_seconds stays
a genuine measurement) but the displayed congestion values are the
pre-baked fixture array from scripts/generate_stub_fixtures.py, since an
untrained model's raw output carries no structure (see that script's
docstring for why that matters for the frontend animations).
"""

import json
import os
import time

import numpy as np
import torch

from metrics import kendall_tau
from metrics import spearman_corr

KENDALL_EXACT_MAX_GCELLS = 10000
KENDALL_SAMPLE_SIZE = 2000
KENDALL_SAMPLE_SEED = 0


class DesignStore:
    def __init__(self, stub_dir, processed_dir, model, model_source):
        self.stub_dir = stub_dir
        self.processed_dir = processed_dir
        self.model = model
        self.model_source = model_source
        self.manifest = self._load_manifest()
        self.graph_cache = {}

    def _load_manifest(self):
        manifest_path = os.path.join(self.stub_dir, "manifest.json")
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

    def prediction_maps(self, design_id):
        """Returns dict(horizontal=, vertical=, inference_seconds=). See
        the module docstring for why inference_seconds is always a real
        measurement even in fixture mode.
        """
        data = self.get_graph(design_id)

        start = time.perf_counter()
        with torch.no_grad():
            model_output = self.model(data).numpy()
        elapsed = time.perf_counter() - start

        if self.model_source == "checkpoint":
            prediction = model_output
        else:
            prediction = data["gcell"].y_pred_fixture.numpy()

        horizontal = prediction[:, 0].tolist()
        vertical = prediction[:, 1].tolist()
        return dict(horizontal=horizontal, vertical=vertical, inference_seconds=elapsed)

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

            n_gcells = pred_channel.shape[0]
            if n_gcells <= KENDALL_EXACT_MAX_GCELLS:
                kendall = kendall_tau(pred_channel, truth_channel)
                kendall_sampled = False
            else:
                sample_rng = np.random.default_rng(KENDALL_SAMPLE_SEED)
                sample_idx = sample_rng.choice(n_gcells, size=KENDALL_SAMPLE_SIZE, replace=False)
                kendall = kendall_tau(pred_channel[sample_idx], truth_channel[sample_idx])
                kendall_sampled = True

            result[channel_name] = dict(
                spearman=spearman,
                kendall=kendall,
                kendall_sampled=kendall_sampled,
                mae=mae,
                rmse=rmse,
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
