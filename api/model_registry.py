"""Loads the model once, at API startup.

Prefers a real training checkpoint (data/processed/*checkpoint*.pt) when
one exists. Otherwise falls back to an untrained CongestionGNN, built
from a stub graph's own node-feature dims and edge types rather than any
hard-coded schema, with a fixed seed so stub-mode demo runs are
repeatable across restarts.

This is the only place the "real checkpoint vs. stub fixture" decision
gets made. Every route downstream just reads store.model_source off the
DesignStore - that's what the frontend's fixture banner keys off, and
it's also what flips the shown prediction from the stub fixture array to
the model's own forward pass (see designs.py).
"""

import glob
import os

import torch

from model import CongestionGNN

HIDDEN_DIM = 64
NUM_LAYERS = 3
UNTRAINED_SEED = 0


def find_checkpoint(processed_dir):
    pattern = os.path.join(processed_dir, "*checkpoint*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_model(processed_dir, sample_graph):
    """Returns dict(model=, model_source=, checkpoint_path=).

    model_source is "checkpoint" or "fixture" - the single flag the rest
    of the API and the frontend key off.
    """
    checkpoint_path = find_checkpoint(processed_dir)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = CongestionGNN(
            checkpoint["in_dims"],
            checkpoint["edge_types"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return dict(model=model, model_source="checkpoint", checkpoint_path=checkpoint_path)

    in_dims = {}
    for node_type in sample_graph.node_types:
        in_dims[node_type] = sample_graph[node_type].x.shape[1]
    edge_types = sample_graph.edge_types

    torch.manual_seed(UNTRAINED_SEED)
    model = CongestionGNN(in_dims, edge_types, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model.eval()
    return dict(model=model, model_source="fixture", checkpoint_path=None)
