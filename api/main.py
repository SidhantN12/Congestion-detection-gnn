"""FastAPI backend for the congestion-GNN demo frontend.

Serves the real corpus when a checkpoint and corpus/api_manifest.json
exist. Without a checkpoint, serves explicitly labelled stub fixtures.
Prepare real metadata with scripts/prepare_real_api.py.

Run: uvicorn api.main:app --reload --port 8000
"""

import json
import os
import sys

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(API_DIR)
SRC_DIR = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, API_DIR)

from designs import DesignStore
from model_registry import load_model
from model_registry import find_checkpoint

STUB_DIR = os.path.join(ROOT, "data", "stub")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
WEB_DIR = os.path.join(ROOT, "web")

state = {}


@asynccontextmanager
async def lifespan(app):
    real_dir = os.path.join(PROCESSED_DIR, "corpus")
    real_manifest = os.path.join(real_dir, "api_manifest.json")
    has_checkpoint = find_checkpoint(PROCESSED_DIR) is not None
    if has_checkpoint and not os.path.exists(real_manifest):
        raise RuntimeError("Checkpoint found but corpus/api_manifest.json is missing; prepare the real corpus first.")
    use_real = has_checkpoint
    data_dir = real_dir if use_real else STUB_DIR
    manifest_path = real_manifest if use_real else os.path.join(STUB_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            "data/stub/manifest.json not found. Run "
            "`python scripts/generate_stub_fixtures.py` first."
        )

    with open(manifest_path) as manifest_file:
        manifest_preview = json.load(manifest_file)
    first_id = manifest_preview[0]["id"]
    sample_path = os.path.join(data_dir, f"{first_id}_graph.pt")
    sample_graph = torch.load(sample_path, weights_only=False)

    model_result = load_model(PROCESSED_DIR, sample_graph)
    store = DesignStore(data_dir, PROCESSED_DIR, model_result["model"], model_result["model_source"], manifest_path)
    state["store"] = store
    state["checkpoint_path"] = model_result["checkpoint_path"]

    yield

    state.clear()


app = FastAPI(title="Congestion GNN API", lifespan=lifespan)


def get_store():
    return state["store"]


def require_entry(store, design_id):
    entry = store.get_entry(design_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown design '{design_id}'")
    return entry


@app.get("/designs")
def list_designs():
    store = get_store()
    designs = []
    for entry in store.list_designs():
        designs.append(dict(
            id=entry["id"],
            name=entry["name"],
            cell_count=entry["cell_count"],
            grid_shape=entry["grid_shape"],
            split=entry["split"],
            model_source=store.model_source,
        ))
    return designs


@app.get("/designs/{design_id}")
def get_design(design_id: str):
    store = get_store()
    entry = require_entry(store, design_id)
    return dict(
        id=entry["id"],
        name=entry["name"],
        cell_count=entry["cell_count"],
        grid_shape=entry["grid_shape"],
        split=entry["split"],
        die_w_dbu=entry["die_w_dbu"],
        die_h_dbu=entry["die_h_dbu"],
        macro_outlines=entry["macro_outlines"],
        model_source=store.model_source,
    )


@app.get("/designs/{design_id}/prediction")
def get_prediction(design_id: str):
    store = get_store()
    entry = require_entry(store, design_id)
    prediction_result = store.prediction_maps(design_id)
    return dict(
        design_id=design_id,
        grid_shape=entry["grid_shape"],
        horizontal=prediction_result["horizontal"],
        vertical=prediction_result["vertical"],
        inference_seconds=prediction_result["inference_seconds"],
        model_source=store.model_source,
        reference_router_seconds=122,
        sample_router_seconds=entry.get("reference_router_seconds"),
    )


@app.get("/designs/{design_id}/truth")
def get_truth(design_id: str):
    store = get_store()
    entry = require_entry(store, design_id)
    truth_result = store.truth_maps(design_id)
    return dict(
        design_id=design_id,
        grid_shape=entry["grid_shape"],
        horizontal=truth_result["horizontal"],
        vertical=truth_result["vertical"],
    )


@app.get("/designs/{design_id}/hotspots")
def get_hotspots(design_id: str, k: int = 20):
    store = get_store()
    require_entry(store, design_id)
    rows = store.hotspots(design_id, k)
    return dict(design_id=design_id, k=k, hotspots=rows)


@app.get("/designs/{design_id}/metrics")
def get_metrics(design_id: str):
    store = get_store()
    require_entry(store, design_id)
    channel_metrics = store.metrics(design_id)
    return dict(design_id=design_id, model_source=store.model_source, channels=channel_metrics)


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
