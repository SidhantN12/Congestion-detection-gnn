"""Label-distribution report for a set of congestion .npz files: how much
mass is tied at the 1.0 ceiling vs spread as a gradient. Used to pick the
blocked-layer range (Phase 14).

Usage:
    python tests/blockage_layer_sweep.py <label>=<congestion.npz> ...
"""

import sys

import numpy as np


def main():
    print(f"{'setting':<14}{'==1.0':>8}{'%==1.0':>8}{'>0.9':>8}{'%>0.9':>8}"
          f"{'>1.0':>6}{'IQR comb':>10}{'IQR H':>8}{'IQR V':>8}{'p25':>7}{'p75':>7}{'max':>8}{'uniq':>7}")
    for arg in sys.argv[1:]:
        label = arg.split("=", 1)[0]
        path = arg.split("=", 1)[1]
        c = np.load(path)
        r = c["ratio"]
        comb = r.max(axis=2).reshape(-1)
        n = comb.size
        at_one = int((comb == 1.0).sum())
        above = int((comb > 0.9).sum())
        over = int((comb > 1.0).sum())
        q = np.percentile(comb, [25, 75])
        qh = np.percentile(r[..., 0], [25, 75])
        qv = np.percentile(r[..., 1], [25, 75])
        print(f"{label:<14}{at_one:>8}{100 * at_one / n:>7.2f}%{above:>8}{100 * above / n:>7.2f}%"
              f"{over:>6}{q[1] - q[0]:>10.4f}{qh[1] - qh[0]:>8.4f}{qv[1] - qv[0]:>8.4f}"
              f"{q[0]:>7.3f}{q[1]:>7.3f}{comb.max():>8.4f}{len(np.unique(comb)):>7}")


if __name__ == "__main__":
    main()
