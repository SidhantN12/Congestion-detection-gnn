"""Evaluate the validation-selected model and train-fitted baselines on aes."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))
from baselines import CapacityBaseline, DensityBaseline, RUDYBaseline
from designs import select_kendall
from metrics import hotspot_detection, spearman_corr
from model import CongestionGNN


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    run_dir = ROOT / "runs/corpus_final"
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    split = json.loads((run_dir / "split.json").read_text())
    entries = json.loads((ROOT / checkpoint["config"]["data"]["manifest"]).read_text())
    train = [torch.load(ROOT / entry["path"], weights_only=False) for entry in entries if entry["design"] in split["train"]]
    model = CongestionGNN(checkpoint["in_dims"], checkpoint["edge_types"],
                          hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    methods = {"CongestionGNN": model, "RUDY": RUDYBaseline(),
               "Density-only": DensityBaseline().fit(train),
               "Capacity-only": CapacityBaseline().fit(train)}
    del train
    result = dict(checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  best_epoch=checkpoint["best_epoch"], config_hash=checkpoint["config_hash"],
                  split=split, aggregation="unweighted mean over test variants, per channel",
                  latency_runs=25, latency_warmup_runs=1, torch_threads=torch.get_num_threads(),
                  reference_router_seconds=122, samples=[], summary={})
    for entry in entries:
        if entry["design"] not in split["test"]:
            continue
        graph = torch.load(ROOT / entry["path"], weights_only=False)
        truth = graph["gcell"].y_ratio.numpy()
        for name in methods:
            method = methods[name]
            with torch.no_grad():
                prediction = method(graph).numpy()
                timings = []
                for _ in range(25):
                    start = time.perf_counter()
                    method(graph)
                    timings.append(time.perf_counter() - start)
            assert np.isfinite(prediction).all()
            row = dict(design=entry["design"], variant=entry["variant"], method=name,
                       n_gcells=len(truth), inference_seconds=float(np.median(timings)),
                       latency_samples_seconds=timings, channels={})
            for channel in range(2):
                pred = prediction[:, channel]
                target = truth[:, channel]
                hotspot = hotspot_detection(pred, target)
                row["channels"]["HV"[channel]] = dict(
                    spearman=spearman_corr(pred, target), **select_kendall(pred, target),
                    precision=hotspot["precision"], recall=hotspot["recall"], roc_auc=hotspot["roc_auc"],
                    mae=float(np.mean(np.abs(pred - target))),
                    rmse=float(np.sqrt(np.mean((pred - target) ** 2))),
                    n_hotspots=hotspot["n_hotspots"])
            result["samples"].append(row)
            print(f"{entry['variant']} {name}: {row['channels']}", flush=True)
    for name in methods:
        rows = [row for row in result["samples"] if row["method"] == name]
        result["summary"][name] = {}
        for channel in "HV":
            result["summary"][name][channel] = {
                metric: float(np.mean([row["channels"][channel][metric] for row in rows]))
                for metric in ("spearman", "kendall", "precision", "recall", "roc_auc", "mae", "rmse")}
    target = ROOT / "notes/results/gate4_evaluation.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
