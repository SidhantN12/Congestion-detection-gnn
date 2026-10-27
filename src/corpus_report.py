"""Phase 14 corpus table: one row per sample in a raw corpus directory.

Reports cells, nets, GCell grid, max congestion ratio and fraction of
GCells > 0.9 per channel (H and V, never averaged), count at exactly 1.0,
combined IQR, distinct ratio values (combined / H / V: ties in the
labels, which bound how much Spearman's midrank correction matters), and
the run times from logs/<design>_<variant>.timing. Then
checks for duplicate congestion maps (sha256 of the ratio array) and lists
designs where no variant has a GCell > 0.9 in either channel.

Usage: python src/corpus_report.py <raw_corpus_dir> <logs_dir>
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

from metrics import combined_congestion


def read_timing(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key = line.split("=", 1)[0]
            out[key] = line.split("=", 1)[1]
    return out


def fmt_s(value):
    if value is None or value == "" or value == "NA":
        return "NA"
    return f"{float(value):.0f}"


def main():
    raw_dir = Path(sys.argv[1])
    logs_dir = Path(sys.argv[2])
    rows = []
    hashes = {}
    for cong_path in sorted(raw_dir.glob("sweep_*_congestion.npz")):
        sample = cong_path.name.removeprefix("sweep_").removesuffix("_congestion.npz")
        design = sample.split("__")[0]
        variant = sample.split("__")[-1]
        cong = np.load(cong_path)
        place = np.load(raw_dir / f"sweep_{sample}_placement.npz")
        ratio = cong["ratio"]
        n = ratio.shape[0] * ratio.shape[1]
        comb = combined_congestion(ratio)
        p25 = np.percentile(comb, 25)
        p75 = np.percentile(comb, 75)
        digest = hashlib.sha256(np.ascontiguousarray(ratio).tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(sample)
        timing = read_timing(logs_dir / f"{design}_{variant}.timing")
        rows.append(dict(
            sample=sample, design=design,
            cells=len(place["cell_names"]), nets=len(place["net_names"]),
            grid=f"{ratio.shape[1]}x{ratio.shape[0]}",
            max_h=ratio[:, :, 0].max(), max_v=ratio[:, :, 1].max(),
            hot_h=int((ratio[:, :, 0] > 0.9).sum()), hot_v=int((ratio[:, :, 1] > 0.9).sum()),
            at_one=int((comb == 1.0).sum()), n=n, iqr=p75 - p25,
            uniq=len(np.unique(comb)), uniq_h=len(np.unique(ratio[:, :, 0])),
            uniq_v=len(np.unique(ratio[:, :, 1])),
            synth=timing.get("synth_s"), p2g=timing.get("place_to_grt_s"),
            grt=timing.get("reference_router_seconds"), total=timing.get("total_s"),
        ))

    print("| sample | cells | nets | grid | max H / V | > 0.9 H | > 0.9 V | == 1.0 | IQR comb | distinct comb / H / V | synth s | place->GRT s | GRT step s | total s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['sample']} | {r['cells']:,} | {r['nets']:,} | {r['grid']} "
              f"| {r['max_h']:.4f} / {r['max_v']:.4f} "
              f"| {r['hot_h']} ({100 * r['hot_h'] / r['n']:.2f}%) "
              f"| {r['hot_v']} ({100 * r['hot_v'] / r['n']:.2f}%) "
              f"| {r['at_one']} ({100 * r['at_one'] / r['n']:.2f}%) | {r['iqr']:.4f} "
              f"| {r['uniq']} / {r['uniq_h']} / {r['uniq_v']} "
              f"| {fmt_s(r['synth'])} | {fmt_s(r['p2g'])} | {fmt_s(r['grt'])} | {fmt_s(r['total'])} |")

    dups = [names for names in hashes.values() if len(names) > 1]
    print(f"\n{len(rows)} samples, {len(hashes)} distinct congestion maps, {len(dups)} duplicate groups")
    for names in dups:
        print("  duplicate:", ", ".join(names))

    designs = sorted({r["design"] for r in rows})
    for d in designs:
        hot = [r["hot_h"] + r["hot_v"] for r in rows if r["design"] == d]
        if max(hot) == 0:
            print(f"  {d}: no GCell > 0.9 in any variant")


if __name__ == "__main__":
    main()
