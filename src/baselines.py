"""Analytical reference predictors for the results table. Both take the same
HeteroData the GNN takes and return the same (num_gcells, 2) tensor of
[horizontal, vertical] congestion-ratio estimates, as torch.nn.Modules, so
`baseline(data)` is a drop-in for `model(data)`.

RUDYBaseline - Rectangular Uniform wire DensitY
    P. Spindler, F. M. Johannes, "Fast and Accurate Routing Demand
    Estimation for Efficient Routability-driven Placement", DATE 2007, sec. 2:
      (1)  d_n = WA_n / NA_n,  NA_n = w_n * h_n,  WA_n = L_n * p,
           p = (average wire-to-wire pitch) / (number of routing layers),
           L_n = HPWL = w_n + h_n  (the estimator the paper uses)
      (2)  R(x,y; x_ll,y_ll,w,h) = 1 inside the net's enclosing rectangle, else 0
      (3)  D(x,y) = sum_n d_n * R(x,y; x_n,y_n,w_n,h_n)
    The paper stresses RUDY is not bin-based; on the gcell grid we report
    the exact area-average of D(x,y) over each gcell (the integral of (3)
    over the gcell / gcell area), not a rasterised approximation.

    Choices the paper does not make (documented, not hidden):
      - Direction split. The paper's RUDY is one scalar field. The two
        output channels split L_n into its horizontal (w_n) and vertical
        (h_n) parts: d_n^H = w_n*p/NA_n, d_n^V = h_n*p/NA_n, so
        D^H + D^V == the paper's D exactly (tested).
      - Degenerate rectangles. Eq. (1) divides by w_n*h_n, undefined for
        nets whose pins share a row or column. The rectangle is widened to
        at least `min_extent` (default one gcell) in each dimension; the
        wire length L_n keeps the true w_n + h_n. A 1-pin net has L_n = 0
        and contributes nothing.
      - Units. Coordinates are in gcell pitches (the graph stores
        normalised positions x/die_w, y/die_h, rescaled by the grid shape;
        pitch_x == pitch_y in this pipeline). With p = 1, D is wire length
        per gcell area in gcell pitches = routing tracks used per gcell,
        the same unit as the gcell capacity features, so dividing by
        capacity gives a ratio estimate (`normalize_by_capacity`). Absolute
        scale still depends on the unknown p; rank metrics do not.
      - Pin positions. The graph carries each cell's lower-left corner,
        not pin coordinates; every pin sits at its cell's corner.

DensityBaseline - density only, no connectivity
    Per gcell, the cell_density and pin_density input features, each also
    box-averaged over (2r+1)x(2r+1) gcell windows (radii `radii`), plus a
    bias -> a least-squares linear map to [H, V] ratio, fitted with
    .fit(graphs). Uses no net or pin edges at all - only gcell features
    and the gcell grid layout.

Usage:
    from baselines import RUDYBaseline, DensityBaseline
    rudy = RUDYBaseline();   pred = rudy(data)
    dens = DensityBaseline().fit(train_graphs);   pred = dens(data)
"""

import torch
import torch.nn as nn

PIN = ("cell", "pin", "net")
CELL_X, CELL_Y = 3, 4          # cell features: x/die_w, y/die_h (build_graph.py)
CELL_DENSITY, PIN_DENSITY = 0, 1
CAP_H, CAP_V = 2, 3            # gcell features: capacity_horizontal, capacity_vertical


def _grid(data):
    ny, nx = (int(v) for v in data["gcell"].grid_shape)
    assert ny * nx == data["gcell"].num_nodes, "baselines need the full gcell grid (not a partition)"
    return ny, nx


def net_bboxes(data):
    """Per-net enclosing rectangle of its pins in gcell-pitch units:
    (x0, x1, y0, y1), float64, from pin edges and cell positions."""
    ny, nx = _grid(data)
    cell, net = data[PIN].edge_index
    n_nets = data["net"].num_nodes
    px = data["cell"].x[cell, CELL_X].double() * nx
    py = data["cell"].x[cell, CELL_Y].double() * ny
    inf = torch.full((n_nets,), float("inf"), dtype=torch.float64)
    x0 = inf.scatter_reduce(0, net, px, "amin")
    x1 = (-inf).scatter_reduce(0, net, px, "amax")
    y0 = inf.scatter_reduce(0, net, py, "amin")
    y1 = (-inf).scatter_reduce(0, net, py, "amax")
    has_pins = torch.isfinite(x0)
    return x0, x1, y0, y1, has_pins


def rect_average(x0, x1, y0, y1, weight, ny, nx):
    """sum_n weight_n * area(rect_n intersect gcell) / gcell_area for every
    gcell of a unit-pitch ny x nx grid, exactly, in O(N + grid).

    1D overlap of [a, b] with column c = [c, c+1]: full (1) for columns
    strictly inside, partial at the end columns. Written as
    I(c in [ca, cb]) + corrections at ca and cb, the 2D overlap is a sum of
    rectangle / row-segment / column-segment / point terms, all accumulated
    in one 2D difference array.
    """
    x0 = x0.clamp(0, nx); x1 = x1.clamp(0, nx)
    y0 = y0.clamp(0, ny); y1 = y1.clamp(0, ny)
    ca = x0.floor().clamp(max=nx - 1).long(); cb = (x1.ceil() - 1).clamp(min=0, max=nx - 1).long()
    ra = y0.floor().clamp(max=ny - 1).long(); rb = (y1.ceil() - 1).clamp(min=0, max=ny - 1).long()
    cb = torch.maximum(cb, ca); rb = torch.maximum(rb, ra)
    # corrections (<= 0) removing the uncovered part of the first/last column/row
    ax = -(x0 - ca); bx = -((cb + 1) - x1)
    ay = -(y0 - ra); by = -((rb + 1) - y1)

    diff = torch.zeros((ny + 1) * (nx + 1), dtype=torch.float64)

    def rect(r0, r1, c0, c1, w):
        for rr, cc, s in ((r0, c0, 1.0), (r0, c1 + 1, -1.0), (r1 + 1, c0, -1.0), (r1 + 1, c1 + 1, 1.0)):
            diff.index_add_(0, rr * (nx + 1) + cc, s * w)

    # (I_x + A_x)(I_y + A_y), A_x = ax*delta(ca) + bx*delta(cb), same for y
    rect(ra, rb, ca, cb, weight)                                  # I_x I_y
    rect(ra, ra, ca, cb, weight * ay); rect(rb, rb, ca, cb, weight * by)   # I_x A_y
    rect(ra, rb, ca, ca, weight * ax); rect(ra, rb, cb, cb, weight * bx)   # A_x I_y
    for rr, wy in ((ra, ay), (rb, by)):                            # A_x A_y
        for cc, wx in ((ca, ax), (cb, bx)):
            rect(rr, rr, cc, cc, weight * wx * wy)
    return diff.reshape(ny + 1, nx + 1).cumsum(0).cumsum(1)[:ny, :nx]


class RUDYBaseline(nn.Module):
    def __init__(self, wire_width=1.0, min_extent=1.0, normalize_by_capacity=True):
        super().__init__()
        self.wire_width = wire_width
        self.min_extent = min_extent
        self.normalize_by_capacity = normalize_by_capacity

    def demand(self, data):
        """(ny, nx, 2) float64 RUDY [horizontal, vertical], in tracks per gcell."""
        ny, nx = _grid(data)
        x0, x1, y0, y1, has_pins = net_bboxes(data)
        x0, x1, y0, y1 = x0[has_pins], x1[has_pins], y0[has_pins], y1[has_pins]
        w, h = x1 - x0, y1 - y0
        # widen degenerate rectangles symmetrically to min_extent; L_n keeps true w + h
        pad_x = (self.min_extent - w).clamp(min=0) / 2
        pad_y = (self.min_extent - h).clamp(min=0) / 2
        x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
        area = (x1 - x0) * (y1 - y0)
        dh = w * self.wire_width / area   # d_n^H: horizontal share of eq. (1)
        dv = h * self.wire_width / area
        return torch.stack([rect_average(x0, x1, y0, y1, dh, ny, nx),
                            rect_average(x0, x1, y0, y1, dv, ny, nx)], dim=-1)

    @torch.no_grad()
    def forward(self, data):
        d = self.demand(data).reshape(-1, 2)
        if self.normalize_by_capacity:
            cap = data["gcell"].x[:, [CAP_H, CAP_V]].double()
            d = torch.where(cap > 0, d / cap.clamp(min=1e-12), torch.zeros_like(d))
        return d.float()


def box_average(grid, r):
    """Mean over the (2r+1)x(2r+1) window around each cell of a (ny, nx)
    grid, truncated at the edges (averaged over the in-grid part only)."""
    if r == 0:
        return grid
    ny, nx = grid.shape
    pad = torch.zeros(ny + 1, nx + 1, dtype=grid.dtype)
    pad[1:, 1:] = grid.cumsum(0).cumsum(1)
    ones = torch.zeros(ny + 1, nx + 1, dtype=grid.dtype)
    ones[1:, 1:] = torch.ones_like(grid).cumsum(0).cumsum(1)
    ys, xs = torch.arange(ny), torch.arange(nx)
    y0, y1 = (ys - r).clamp(min=0), (ys + r + 1).clamp(max=ny)
    x0, x1 = (xs - r).clamp(min=0), (xs + r + 1).clamp(max=nx)

    def window_sum(t):
        return (t[y1][:, x1] - t[y0][:, x1] - t[y1][:, x0] + t[y0][:, x0])
    return window_sum(pad) / window_sum(ones)


class DensityBaseline(nn.Module):
    def __init__(self, radii=(0, 1, 2)):
        super().__init__()
        self.radii = tuple(radii)
        self.register_buffer("weight", torch.empty(0))

    def features(self, data):
        ny, nx = _grid(data)
        cols = [torch.ones(ny * nx, dtype=torch.float64)]
        for ch in (CELL_DENSITY, PIN_DENSITY):
            g = data["gcell"].x[:, ch].double().reshape(ny, nx)
            cols += [box_average(g, r).reshape(-1) for r in self.radii]
        return torch.stack(cols, dim=1)

    def fit(self, graphs, target="y_ratio"):
        X = torch.cat([self.features(g) for g in graphs])
        Y = torch.cat([g["gcell"][target].double() for g in graphs])
        self.weight = torch.linalg.lstsq(X, Y).solution  # (n_features, 2)
        return self

    @torch.no_grad()
    def forward(self, data):
        if self.weight.numel() == 0:
            raise RuntimeError("DensityBaseline is not fitted - call .fit(graphs) first")
        return (self.features(data) @ self.weight).float()
