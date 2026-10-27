#!/usr/bin/env bash
# Phase 14 corpus: 4 nangate45 designs x 3 placement variants, sequential,
# then graph caching. One command regenerates everything:
#
#   bash flow/sweep_corpus.sh
#
# Each run is flow/run_flow.sh (uniform recipe there: no incremental
# reroute, no fixed floorplan, signal stack capped at metal5 at GRT).
# Variants: three CORE_UTILIZATION/GPL_RANDOM_SEED pairs per design
# (below). A failing run steps utilisation back by STEP points (at most
# MAX_STEPBACKS times); a design that still fails is recorded and the
# sweep moves on.
#
# Label-quality rule, fixed before any model was trained on this corpus
# and applied to all four designs alike: utilisation steps are chosen so
# no sample saturates. Operationally, a sample whose combined-congestion
# IQR is below MIN_IQR is re-run IQR_STEP points lower (same seed), at
# most MAX_IQR_STEPS times; after that the last run is kept as is. A
# step-down never goes to or below the previous variant's utilisation.
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
MIN_IQR=0.17
IQR_STEP=3
MAX_IQR_STEPS=2

# design:util/seed,... per variant. The label-quality rule above set the
# steps per design: the first sweep used default / +15 / +25 everywhere,
# and aes 60 / 70 and tinyRocket 75 / 85 came out saturated (combined IQR
# 0.080-0.127 vs 0.17-0.23 elsewhere, 21.6-51.6% of GCells at exactly
# 1.0); aes 52 / 58 were still compressed (IQR 0.160 / 0.128), so aes
# steps down from 45 instead. aes ships a fixed floorplan implying ~30
# rather than a utilisation. See notes/phase14-real-corpus.md.
DESIGNS="dynamic_node:40/1,55/2,65/3 aes:35/2,40/3,45/1 ibex:50/1,65/2,75/3 tinyRocket:60/1,66/2,72/3"

mkdir -p "$PROJECT/logs" "$CORPUS_RAW"
find "${CORPUS_RAW:?}" -maxdepth 1 -type l -delete
echo -e "design\tvariant\tutil\tseed\tstatus\tiqr" > "$RUNS"

combined_iqr() {
  "$PY" -c "import sys,numpy as np; r=np.load(sys.argv[1])['ratio']; c=np.maximum(r[:,:,0],r[:,:,1]); q=np.percentile(c,[25,75]); print(f'{q[1]-q[0]:.4f}')" "$1"
}

ratio_hash() {
  "$PY" -c "import hashlib,sys,numpy as np; r=np.load(sys.argv[1])['ratio']; print(hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()+' '+str(r.shape))" "$1"
}

is_done() {
  grep -q '^status=OK$' "$PROJECT/logs/${1}_${2}.timing" 2>/dev/null \
    && [ -f "$RAW/sweep_${1}__${2}_congestion.npz" ] && [ -f "$RAW/sweep_${1}__${2}_placement.npz" ]
}

for spec in $DESIGNS; do
  design=${spec%%:*}
  pairs=${spec##*:}
  prev_util=0
  idx=0
  for pair in ${pairs//,/ }; do
    util=${pair%%/*}
    seed=${pair##*/}
    tries=0
    iqr_steps=0
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
      iqr=NA
      if [ $status = OK ]; then
        iqr=$(combined_iqr "$RAW/sweep_${design}__${variant}_congestion.npz")
      fi
      echo -e "$design\t$variant\t$util\t$seed\t$status\t$iqr" >> "$RUNS"
      if [ $status = OK ]; then
        next=$((util - IQR_STEP))
        if awk -v a="$iqr" -v b="$MIN_IQR" 'BEGIN{exit !(a < b)}' \
            && [ $iqr_steps -lt $MAX_IQR_STEPS ] && [ $next -gt $prev_util ]; then
          echo "$(date '+%F %T') $design $variant IQR $iqr < $MIN_IQR, stepping down to $next"
          iqr_steps=$((iqr_steps + 1))
          util=$next
          continue
        fi
        echo "$(date '+%F %T') keep $design $variant IQR $iqr"
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
