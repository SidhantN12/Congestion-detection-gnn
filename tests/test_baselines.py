"""Phase 12 checks for src/baselines.py - runs in seconds, no data needed.

1. rect_average (the O(N + grid) exact gcell integration used by RUDY)
   equals a brute-force per-net, per-gcell overlap computation, on random
   rectangles including grid-aligned, single-gcell and die-sized ones.
2. RUDY on a hand-built 3x3 design with one 2-pin net gives the values
   worked out by hand from eqs. (1)-(3) of Spindler & Johannes.
3. RUDY on a synthetic design equals a direct, loop-based evaluation of
   eqs. (1)-(3) per net, and D^H + D^V equals the paper's scalar D.
4. Degenerate nets: 1-pin nets contribute nothing; same-row nets finite.
5. DensityBaseline recovers an exactly-linear target, refuses to predict
   unfitted, and ignores connectivity (dropping every pin edge changes
   nothing).
6. Interface: both baselines return the same shape/dtype as
   CongestionGNN(data), and every function in src/metrics.py accepts them.

Usage:
    python tests/test_baselines.py
"""

import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, "src")
from baselines import DensityBaseline, RUDYBaseline, net_bboxes, rect_average
from metrics import combined_congestion, fraction_above, kendall_tau, spearman_corr
from model import CongestionGNN
from synthetic import generate_graph


def brute_rect_average(x0, x1, y0, y1, w, ny, nx):
    out = np.zeros((ny, nx))
    for a, b, c, d, wt in zip(x0, x1, y0, y1, w):
        for r in range(ny):
            oy = max(0.0, min(d, r + 1) - max(c, r))
            if oy == 0:
                continue
            for col in range(nx):
                ox = max(0.0, min(b, col + 1) - max(a, col))
                out[r, col] += wt * ox * oy
    return out


def check_rect_average():
    rng = np.random.default_rng(0)
    ny, nx, n = 7, 9, 400
    x0 = rng.uniform(0, nx, n); x1 = np.minimum(x0 + rng.exponential(2, n), nx)
    y0 = rng.uniform(0, ny, n); y1 = np.minimum(y0 + rng.exponential(2, n), ny)
    # grid-aligned, single-gcell interior, and whole-die rectangles
    x0[:5], x1[:5], y0[:5], y1[:5] = [0, 2, 3.2, 0, 4], [nx, 3, 3.7, 1, 4.5], [0, 1, 2.1, 0, 6], [ny, 2, 2.4, ny, 7]
    w = rng.uniform(0.1, 2, n)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    fast = rect_average(t(x0), t(x1), t(y0), t(y1), t(w), ny, nx).numpy()
    diff = np.abs(fast - brute_rect_average(x0, x1, y0, y1, w, ny, nx)).max()
    print(f"[1] rect_average vs brute force, {n} rectangles on {ny}x{nx}: max diff {diff:.2e}")
    assert diff < 1e-9


def tiny_design(pins_xy, ny=3, nx=3):
    """Hand-built design: one cell per pin (positions in gcell units), one net."""
    data = HeteroData()
    n = len(pins_xy)
    x = torch.zeros(n, 5)
    x[:, 3] = torch.tensor([p[0] / nx for p in pins_xy])
    x[:, 4] = torch.tensor([p[1] / ny for p in pins_xy])
    data["cell"].x = x
    data["net"].x = torch.zeros(1, 3)
    g = torch.zeros(ny * nx, 4)
    g[:, 2:] = 1.0  # capacity 1 track each way
    data["gcell"].x = g
    data["gcell"].grid_shape = torch.tensor([ny, nx])
    data["cell", "pin", "net"].edge_index = torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long)])
    return data


def check_hand_example():
    # 2-pin net from (0.5, 0.5) to (2.5, 1.5): w=2, h=1, NA=2, p=1
    # d^H = w*p/NA = 1.0, d^V = h*p/NA = 0.5 over x in [0.5,2.5], y in [0.5,1.5]
    # x-overlap per column: [0.5, 1, 0.5]; y-overlap per row: [0.5, 0.5, 0]
    got = RUDYBaseline().demand(tiny_design([(0.5, 0.5), (2.5, 1.5)])).numpy()
    ox, oy = np.array([0.5, 1.0, 0.5]), np.array([0.5, 0.5, 0.0])
    cover = np.outer(oy, ox)
    assert np.allclose(got[:, :, 0], 1.0 * cover) and np.allclose(got[:, :, 1], 0.5 * cover)
    assert np.isclose(got.sum(), (2 + 1) * 1.0)  # total wire length = HPWL * p
    print(f"[2] hand example: H row 0 = {got[0, :, 0].tolist()}, V row 0 = {got[0, :, 1].tolist()}, "
          f"total demand {got.sum():.3f} = HPWL 3")


def check_against_paper(data):
    ny, nx = (int(v) for v in data["gcell"].grid_shape)
    x0, x1, y0, y1, has = (t.numpy() for t in net_bboxes(data))
    D_ref = np.zeros((ny, nx))
    H_ref = np.zeros((ny, nx))
    for a, b, c, d in zip(x0[has], x1[has], y0[has], y1[has]):
        w, h = b - a, d - c
        L = w + h                                  # HPWL
        pw, ph = max(1.0 - w, 0) / 2, max(1.0 - h, 0) / 2
        a, b, c, d = a - pw, b + pw, c - ph, d + ph
        NA = (b - a) * (d - c)
        dn = L * 1.0 / NA                          # eq. (1), p = 1
        cov = brute_rect_average([a], [b], [c], [d], [1.0], ny, nx)
        D_ref += dn * cov                          # eq. (3), area-averaged per gcell
        H_ref += (w / NA) * cov
    rudy = RUDYBaseline().demand(data).numpy()
    d_all = np.abs(rudy.sum(-1) - D_ref).max()
    d_h = np.abs(rudy[:, :, 0] - H_ref).max()
    print(f"[3] synthetic design, {int(has.sum())} nets: max |D^H + D^V - paper D| = {d_all:.2e}, "
          f"max |D^H - direct| = {d_h:.2e} (D range 0-{D_ref.max():.2f})")
    assert d_all < 1e-9 and d_h < 1e-9


def check_degenerate():
    one_pin = RUDYBaseline().demand(tiny_design([(1.2, 1.3)]))
    same_row = RUDYBaseline().demand(tiny_design([(0.2, 1.3), (2.6, 1.3)]))
    assert one_pin.abs().sum() == 0
    assert torch.isfinite(same_row).all() and np.isclose(same_row[..., 0].sum().item(), 2.4)
    assert same_row[..., 1].sum() == 0
    print(f"[4] 1-pin net: total demand {one_pin.sum().item()}; same-row net: finite, "
          f"H total {same_row[..., 0].sum().item():.2f} (= w), V total {same_row[..., 1].sum().item()}")


def check_density(data):
    try:
        DensityBaseline()(data)
        raise AssertionError("unfitted DensityBaseline should raise")
    except RuntimeError:
        pass
    dens = DensityBaseline()
    X = dens.features(data)
    true_w = torch.tensor([[0.3, -0.1], [1.0, 0.2], [0.0, 0.5], [-0.4, 0.0],
                           [0.2, 0.2], [0.0, 1.5], [0.7, -0.3]], dtype=torch.float64)
    data["gcell"].y_synthetic = (X @ true_w).float()
    dens.fit([data], target="y_synthetic")
    err = (dens.weight - true_w).abs().max().item()
    no_edges = data.clone()
    for et in no_edges.edge_types:
        if et[1] in ("pin", "rev_pin", "near"):
            no_edges[et].edge_index = no_edges[et].edge_index[:, :0]
    same = torch.equal(dens(data), dens(no_edges))
    print(f"[5] density: recovers exact linear weights (max err {err:.2e}); output unchanged with "
          f"all pin/rev_pin/near edges removed: {same}")
    assert err < 1e-6 and same
    del data["gcell"].y_synthetic
    return DensityBaseline().fit([data])


def check_interface(data, dens):
    model = CongestionGNN({nt: data[nt].x.shape[1] for nt in data.node_types}, list(data.edge_types))
    with torch.no_grad():
        ref = model(data)
    ny, nx = (int(v) for v in data["gcell"].grid_shape)
    y = data["gcell"].y_ratio
    for name, b in [("RUDY", RUDYBaseline()), ("density", dens)]:
        pred = b(data)
        assert pred.shape == ref.shape and pred.dtype == ref.dtype and torch.isfinite(pred).all()
        sp = [spearman_corr(pred[:, c], y[:, c]) for c in range(2)]
        kt = [kendall_tau(pred[:, c], y[:, c]) for c in range(2)]
        grid = pred.reshape(ny, nx, 2).numpy()
        cc = combined_congestion(grid)
        fa = fraction_above(grid, 0.9)
        assert all(np.isfinite(sp + kt)) and cc.shape == (ny, nx) and 0 <= fa <= 1
        print(f"[6] {name}: shape {tuple(pred.shape)} {pred.dtype} == model {tuple(ref.shape)} {ref.dtype}; "
              f"spearman {sp[0]:.3f}/{sp[1]:.3f}, kendall {kt[0]:.3f}/{kt[1]:.3f}, "
              f"combined_congestion {cc.shape}, fraction_above(0.9) {fa:.3f}")


def main():
    check_rect_average()
    check_hand_example()
    data = generate_graph(3000, seed=0)
    check_against_paper(data)
    check_degenerate()
    dens = check_density(data)
    check_interface(data, dens)
    print("\nPASS")


if __name__ == "__main__":
    main()
