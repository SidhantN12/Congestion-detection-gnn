"""FastAPI backend for the congestion-GNN demo frontend.

Serves the Gate 1 stub designs (data/stub/, see
scripts/generate_stub_fixtures.py) today. The same routes will serve the
real nangate45 corpus once it's copied into data/processed/ and a
trained checkpoint exists (Gate 4) - model_registry.load_model() picks
that up automatically, so nothing here needs to change for that switch.

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

STUB_DIR = os.path.join(ROOT, "data", "stub")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
WEB_DIR = os.path.join(ROOT, "web")

state = {}


@asynccontextmanager
async def lifespan(app):
    manifest_path = os.path.join(STUB_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            "data/stub/manifest.json not found. Run "
            "`python scripts/generate_stub_fixtures.py` first."
        )

    with open(manifest_path) as manifest_file:
        manifest_preview = json.load(manifest_file)
    first_id = manifest_preview[0]["id"]
    sample_path = os.path.join(STUB_DIR, f"{first_id}_graph.pt")
    sample_graph = torch.load(sample_path, weights_only=False)

    model_result = load_model(PROCESSED_DIR, sample_graph)
    store = DesignStore(STUB_DIR, PROCESSED_DIR, model_result["model"], model_result["model_source"])
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
        reference_router_seconds=None,
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
