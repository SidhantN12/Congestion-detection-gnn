#!/usr/bin/env bash
# Phase 14 corpus: 4 nangate45 designs x 3 placement variants, sequential,
# then graph caching. One command regenerates everything:
#
#   bash flow/sweep_corpus.sh
#
# Each run is flow/run_flow.sh (uniform recipe there: no incremental
# reroute, no fixed floorplan, signal stack capped at metal5 at GRT).
# Variants: three CORE_UTILIZATION values per design (below), with
# GPL_RANDOM_SEED 1 / 2 / 3. A failing run steps utilisation back by STEP
# points (at most MAX_STEPBACKS times, never to or below the previous
# variant's value); a design that still fails is recorded and the sweep
# moves on.
#
# SKIP_EXISTING=1 reuses any variant whose timing file says status=OK and
# whose .npz files exist, so only new or changed variants are run.
#
# After the first two dynamic_node variants, their congestion maps are
# hashed and the sweep stops if they are identical.
#
# Output: data/raw/sweep_<design>__<variant>_*.npz (from run_flow.sh),
# symlinked into data/raw/corpus/ (rebuilt every time, so it holds exactly
# this sweep's samples), graphs + manifest.json in data/processed/corpus/,
# run list in logs/corpus_runs.tsv.
set -uo pipefail
# Every rm is in ${VAR:?} form (here and in run_flow.sh) so an unset variable
# stops the command instead of widening the target.

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
RAW="$PROJECT/data/raw"
CORPUS_RAW="$RAW/corpus"
PROCESSED="$PROJECT/data/processed/corpus"
RUNS="$PROJECT/logs/corpus_runs.tsv"
PY="$PROJECT/.venv/bin/python"
STEP=5
MAX_STEPBACKS=2

# design:CORE_UTILIZATION per variant. First value is the design's default
# (aes ships a fixed floorplan implying ~30, which would make it far
# sparser than the rest; 45 is used). Steps were chosen so that no sample
# saturates: the first sweep used +15 / +25 everywhere, and aes 60 / 70
# and tinyRocket 75 / 85 came out degenerate (combined IQR 0.080-0.127
# vs 0.17-0.23 elsewhere, 21.6-51.6% of GCells at exactly 1.0), so their
# steps were tightened. See notes/phase14-real-corpus.md.
DESIGNS="dynamic_node:40,55,65 aes:45,52,58 ibex:50,65,75 tinyRocket:60,66,72"
SEEDS="1 2 3"

mkdir -p "$PROJECT/logs" "$CORPUS_RAW"
find "${CORPUS_RAW:?}" -maxdepth 1 -type l -delete
echo -e "design\tvariant\tutil\tseed\tstatus" > "$RUNS"

ratio_hash() {
  "$PY" -c "import hashlib,sys,numpy as np; r=np.load(sys.argv[1])['ratio']; print(hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()+' '+str(r.shape))" "$1"
}

is_done() {
  grep -q '^status=OK$' "$PROJECT/logs/${1}_${2}.timing" 2>/dev/null \
    && [ -f "$RAW/sweep_${1}__${2}_congestion.npz" ] && [ -f "$RAW/sweep_${1}__${2}_placement.npz" ]
}

for spec in $DESIGNS; do
  design=${spec%%:*}
  utils=${spec##*:}
  prev_util=0
  set -- $SEEDS
  idx=0
  for util in ${utils//,/ }; do
    seed=$1
    shift
    tries=0
    while :; do
      variant=u${util}_s${seed}
      if [ "${SKIP_EXISTING:-0}" = 1 ] && is_done "$design" "$variant"; then
        status=OK
        echo "$(date '+%F %T') reuse $design $variant"
      else
        echo "$(date '+%F %T') start $design $variant"
        bash "$PROJECT/flow/run_flow.sh" "$design" "$variant" CORE_UTILIZATION=$util GPL_RANDOM_SEED=$seed
        rc=$?
        status=FAIL
        if [ $rc -eq 0 ]; then
          status=OK
        fi
        echo "$(date '+%F %T') $status $design $variant (rc $rc)"
      fi
      echo -e "$design\t$variant\t$util\t$seed\t$status" >> "$RUNS"
      if [ $status = OK ]; then
        ln -sf "../sweep_${design}__${variant}_placement.npz" "$CORPUS_RAW/"
        ln -sf "../sweep_${design}__${variant}_congestion.npz" "$CORPUS_RAW/"
        prev_util=$util
        break
      fi
      tries=$((tries + 1))
      util=$((util - STEP))
      if [ $tries -gt $MAX_STEPBACKS ] || [ $idx -eq 0 ] || [ $util -le $prev_util ]; then
        echo "$(date '+%F %T') giving up on $design variant $idx"
        break
      fi
    done

    if [ "$design" = dynamic_node ] && [ $idx -eq 1 ]; then
      ok=$(awk -F'\t' '$1=="dynamic_node" && $5=="OK" {print $2}' "$RUNS")
      if [ "$(echo "$ok" | wc -l)" -ge 2 ]; then
        h1=$(ratio_hash "$RAW/sweep_dynamic_node__$(echo "$ok" | sed -n 1p)_congestion.npz")
        h2=$(ratio_hash "$RAW/sweep_dynamic_node__$(echo "$ok" | sed -n 2p)_congestion.npz")
        echo "hash check: $(echo "$ok" | sed -n 1p) $h1"
        echo "hash check: $(echo "$ok" | sed -n 2p) $h2"
        if [ "${h1%% *}" = "${h2%% *}" ]; then
          echo "hash check FAILED: identical congestion maps, stopping"
          exit 2
        fi
        echo "hash check passed: maps differ"
      fi
    fi
    idx=$((idx + 1))
  done
done

cd "$PROJECT" || exit 1
# Graphs are rebuilt from scratch so no graph of a dropped variant remains.
mkdir -p "$PROCESSED"
find "${PROCESSED:?}" -maxdepth 1 -name '*_graph.pt' -delete
"$PY" src/build_dataset.py data/raw/corpus data/processed/corpus
"$PY" src/corpus_report.py data/raw/corpus logs
