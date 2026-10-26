#!/usr/bin/env bash
# Phase 14 corpus: 4 nangate45 designs x 3 placement variants, sequential,
# then graph caching. One command regenerates everything:
#
#   bash flow/sweep_corpus.sh
#
# Each run is flow/run_flow.sh (uniform recipe there: no incremental
# reroute, no fixed floorplan, signal stack capped at metal5 at GRT).
# Variants: CORE_UTILIZATION default / +15 / +25, GPL_RANDOM_SEED 1 / 2 / 3.
# A failing run steps utilisation back by STEP points (at most
# MAX_STEPBACKS times, never to or below the previous variant's value);
# a design that still fails is recorded and the sweep moves on.
#
# After the first two dynamic_node variants, their congestion maps are
# hashed and the sweep stops if they are identical.
#
# Output: data/raw/sweep_<design>__<variant>_*.npz (from run_flow.sh),
# symlinked into data/raw/corpus/, graphs + manifest.json in
# data/processed/corpus/, run list in logs/corpus_runs.tsv.
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

# design:default CORE_UTILIZATION. aes ships a fixed floorplan instead of a
# utilisation. That floorplan implies ~30 (core 229.9 x 229.6 um from its
# ROW lines vs 15,351 um^2 synth std-cell area = 29.1%), which would make
# aes far sparser than the rest; 45 is used so it is under comparable
# pressure (step-back covers a placement failure).
DESIGNS="dynamic_node:40 aes:45 ibex:50 tinyRocket:60"
OFFSETS="0 15 25"
SEEDS="1 2 3"

mkdir -p "$PROJECT/logs" "$CORPUS_RAW"
echo -e "design\tvariant\tutil\tseed\tstatus" > "$RUNS"

ratio_hash() {
  "$PY" -c "import hashlib,sys,numpy as np; r=np.load(sys.argv[1])['ratio']; print(hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()+' '+str(r.shape))" "$1"
}

for spec in $DESIGNS; do
  design=${spec%%:*}
  base_util=${spec##*:}
  prev_util=0
  set -- $SEEDS
  for off in $OFFSETS; do
    seed=$1
    shift
    util=$((base_util + off))
    tries=0
    while :; do
      variant=u${util}_s${seed}
      echo "$(date '+%F %T') start $design $variant"
      bash "$PROJECT/flow/run_flow.sh" "$design" "$variant" CORE_UTILIZATION=$util GPL_RANDOM_SEED=$seed
      rc=$?
      if [ $rc -eq 0 ]; then
        echo -e "$design\t$variant\t$util\t$seed\tOK" >> "$RUNS"
        ln -sf "../sweep_${design}__${variant}_placement.npz" "$CORPUS_RAW/"
        ln -sf "../sweep_${design}__${variant}_congestion.npz" "$CORPUS_RAW/"
        echo "$(date '+%F %T') ok $design $variant"
        prev_util=$util
        break
      fi
      echo -e "$design\t$variant\t$util\t$seed\tFAIL" >> "$RUNS"
      echo "$(date '+%F %T') FAIL $design $variant (rc $rc)"
      tries=$((tries + 1))
      util=$((util - STEP))
      if [ $tries -gt $MAX_STEPBACKS ] || [ $off -eq 0 ] || [ $util -le $prev_util ]; then
        echo "$(date '+%F %T') giving up on $design offset +$off"
        break
      fi
    done

    if [ "$design" = dynamic_node ] && [ "$off" = 15 ]; then
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
  done
done

cd "$PROJECT" || exit 1
"$PY" src/build_dataset.py data/raw/corpus data/processed/corpus
"$PY" src/corpus_report.py data/raw/corpus logs
