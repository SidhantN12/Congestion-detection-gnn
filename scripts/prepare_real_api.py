"""Build API metadata from the verified archive without replacing training data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    expected = "ecc795c836255c9c8f858166ad33b89daa97b27b0733bcf3eaa9b7f29b2626f4"
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == expected
    entries = json.loads(Path("data/processed/corpus/manifest.json").read_text())
    output = []
    prefix = "Congestion-detection-gnn/"
    with tarfile.open(args.archive) as archive:
        for entry in entries:
            sample_id = entry["design"] + "__" + entry["variant"]
            graph_path = Path(entry["path"])
            assert archive.extractfile(prefix + str(graph_path)).read() == graph_path.read_bytes()
            graph = torch.load(graph_path, weights_only=False)
            placement = np.load(io.BytesIO(archive.extractfile(
                prefix + "data/raw/sweep_" + sample_id + "_placement.npz").read()))
            timing = archive.extractfile(prefix + "logs/" + entry["design"] + "_" + entry["variant"] + ".timing").read().decode()
            router_lines = [line for line in timing.splitlines() if line.startswith("reference_router_seconds=")]
            router_seconds = float(router_lines[0].split("=", 1)[1])
            split = "train"
            if entry["design"] == "ibex":
                split = "val"
            if entry["design"] == "aes":
                split = "test"
            output.append(dict(id=sample_id, name=sample_id, split=split,
                               cell_count=graph["cell"].num_nodes,
                               grid_shape=graph["gcell"].grid_shape.tolist(),
                               die_w_dbu=float(placement["die_w"]),
                               die_h_dbu=float(placement["die_h"]),
                               macro_outlines=[], reference_router_seconds=router_seconds))
    target = Path("data/processed/corpus/api_manifest.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Verified {len(output)} graphs; wrote {target}")


if __name__ == "__main__":
    main()
