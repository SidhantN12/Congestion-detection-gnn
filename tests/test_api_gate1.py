"""Gate 1 check: every endpoint returns correctly-shaped data for every
stub design, plus the measured inference latency the gate asks for.

Requires data/stub/ to exist (python scripts/generate_stub_fixtures.py).
Uses FastAPI's TestClient - runs the real app (lifespan, routes) without
a live uvicorn process.

Usage: python tests/test_api_gate1.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from fastapi.testclient import TestClient

import designs as designs_module
from main import app
from designs import select_kendall


def check_designs_list(client):
    response = client.get("/designs")
    assert response.status_code == 200
    designs = response.json()
    assert len(designs) == 3
    splits = set()
    for design in designs:
        assert "id" in design
        assert "cell_count" in design
        assert len(design["grid_shape"]) == 2
        assert design["split"] in ("train", "val", "test")
        assert design["model_source"] == "fixture"
        splits.add(design["split"])
    assert splits == {"train", "val", "test"}
    print(f"[1] GET /designs: {len(designs)} designs, splits {sorted(splits)}")
    return designs


def check_design_detail(client, design_id):
    response = client.get(f"/designs/{design_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["id"] == design_id
    assert detail["die_w_dbu"] > 0
    assert detail["die_h_dbu"] > 0
    assert isinstance(detail["macro_outlines"], list)
    ny = detail["grid_shape"][0]
    nx = detail["grid_shape"][1]
    print(f"    {design_id}: grid {ny}x{nx}, die {detail['die_w_dbu']}x{detail['die_h_dbu']} DBU")
    return detail


def check_prediction(client, design_id, grid_shape):
    response = client.get(f"/designs/{design_id}/prediction")
    assert response.status_code == 200
    body = response.json()
    n_gcells = grid_shape[0] * grid_shape[1]
    assert len(body["horizontal"]) == n_gcells
    assert len(body["vertical"]) == n_gcells
    assert body["inference_seconds"] > 0
    assert body["reference_router_seconds"] is None
    assert body["model_source"] == "fixture"
    for value in body["horizontal"][:5] + body["vertical"][:5]:
        assert value >= 0
    return body


def check_truth(client, design_id, grid_shape):
    response = client.get(f"/designs/{design_id}/truth")
    assert response.status_code == 200
    body = response.json()
    n_gcells = grid_shape[0] * grid_shape[1]
    assert len(body["horizontal"]) == n_gcells
    assert len(body["vertical"]) == n_gcells
    return body


def check_hotspots(client, design_id):
    response = client.get(f"/designs/{design_id}/hotspots", params={"k": 20})
    assert response.status_code == 200
    body = response.json()
    rows = body["hotspots"]
    assert len(rows) == 20
    true_ranks = []
    for row in rows:
        assert "x" in row
        assert "y" in row
        assert "predicted" in row
        assert "true" in row
        assert "rank_delta" in row
        true_ranks.append(row["true_rank"])
    assert true_ranks == sorted(true_ranks)
    assert true_ranks[0] == 1
    return rows


def check_metrics(client, design_id, n_gcells):
    response = client.get(f"/designs/{design_id}/metrics")
    assert response.status_code == 200
    body = response.json()
    channels = body["channels"]
    assert set(channels.keys()) == {"horizontal", "vertical"}
    expect_sampled = n_gcells > designs_module.KENDALL_EXACT_MAX_GCELLS
    for channel_name in channels:
        channel = channels[channel_name]
        assert -1.0 <= channel["spearman"] <= 1.0
        assert -1.0 <= channel["kendall"] <= 1.0
        assert channel["mae"] >= 0
        assert channel["rmse"] >= 0
        assert channel["kendall_sampled"] == expect_sampled
    return body


def check_select_kendall_thresholds():
    """select_kendall's exact/sampled/MemoryError-fallback logic, tested
    directly rather than via the stub designs - none of the three stub
    grids exceed KENDALL_EXACT_MAX_GCELLS (14,000), so this is the only
    place the sampled and fallback paths are exercised at all.
    """
    rng = np.random.default_rng(0)

    small_pred = rng.random(500)
    small_truth = rng.random(500)
    small_result = select_kendall(small_pred, small_truth)
    assert small_result["kendall_sampled"] is False

    large_pred = rng.random(designs_module.KENDALL_EXACT_MAX_GCELLS + 1)
    large_truth = rng.random(designs_module.KENDALL_EXACT_MAX_GCELLS + 1)
    large_result = select_kendall(large_pred, large_truth)
    assert large_result["kendall_sampled"] is True

    call_count = {"n": 0}

    def fail_once_then_succeed(pred_channel, truth_channel):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise MemoryError("simulated OOM")
        from metrics import kendall_tau
        return kendall_tau(pred_channel, truth_channel)

    fallback_pred = rng.random(500)
    fallback_truth = rng.random(500)
    fallback_result = select_kendall(fallback_pred, fallback_truth, kendall_tau_fn=fail_once_then_succeed)
    assert fallback_result["kendall_sampled"] is True
    assert call_count["n"] == 2

    print("[0] select_kendall: exact under threshold, sampled over threshold, "
          "MemoryError under threshold falls back to sampled")


def main():
    check_select_kendall_thresholds()
    client = TestClient(app)

    with client:
        designs = check_designs_list(client)

        latency_rows = []
        for design in designs:
            design_id = design["id"]
            detail = check_design_detail(client, design_id)
            grid_shape = detail["grid_shape"]
            n_gcells = grid_shape[0] * grid_shape[1]

            prediction = check_prediction(client, design_id, grid_shape)
            check_truth(client, design_id, grid_shape)
            hotspots = check_hotspots(client, design_id)
            metrics = check_metrics(client, design_id, n_gcells)

            latency_rows.append(dict(
                id=design_id,
                n_gcells=n_gcells,
                inference_ms=prediction["inference_seconds"] * 1000,
                spearman_h=metrics["channels"]["horizontal"]["spearman"],
                spearman_v=metrics["channels"]["vertical"]["spearman"],
                kendall_sampled=metrics["channels"]["horizontal"]["kendall_sampled"],
            ))
            top_rank_delta = hotspots[0]["rank_delta"]
            print(f"    {design_id}: top hotspot rank_delta={top_rank_delta}")

        response = client.get("/designs/does-not-exist")
        assert response.status_code == 404
        print("[2] unknown design -> 404")

    print("\n=== Measured inference latency per design (Gate 1) ===")
    print(f"{'design':<20}{'gcells':>10}{'latency_ms':>14}{'spearman_h':>12}{'spearman_v':>12}{'kendall_sampled':>17}")
    for row in latency_rows:
        print(
            f"{row['id']:<20}{row['n_gcells']:>10}{row['inference_ms']:>14.2f}"
            f"{row['spearman_h']:>12.3f}{row['spearman_v']:>12.3f}{str(row['kendall_sampled']):>17}"
        )

    print("\nPASS")


if __name__ == "__main__":
    main()
