#!/usr/bin/env bash
# One ORFS run, synth -> global route, plus congestion/placement extraction.
# Used directly for single runs and by flow/sweep_corpus.sh for the corpus.
#
# Usage: bash flow/run_flow.sh <design> <variant> [extra make VAR=value ...]
#
# Runs from the project root on the Linux host (not inside the container).
# ORFS work tree goes to $ORFS_WORK (default /root/orfs_work), mounted as
# /work; this repo is mounted as /project so hook scripts and the
# extractors are read from here directly.
#
# Applied to EVERY run, uniformly (see notes/phase14-real-corpus.md):
#   SKIP_INCREMENTAL_REPAIR=1  skips the post-GRT repair_design/repair_timing
#                              passes and their incremental reroutes
#   FLOORPLAN_DEF=             ignores any fixed floorplan DEF (aes ships one),
#                              so CORE_UTILIZATION actually sizes the core
#
# Timing: synth and floorplan->grt are separate make invocations, each timed
# with wall clock around its own `docker run` (container start included).
# Written to logs/<design>_<variant>.timing as key=value lines.
#
# Exit 0 only if both extraction .npz files were written.
#
# EXTRACT_ONLY=1 skips the flow and only re-runs extraction on an existing
# run (timing file is left as it was).
set -uo pipefail

design=$1
variant=$2
shift 2
extra=("$@")

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
ORFS_WORK=${ORFS_WORK:-/root/orfs_work}
IMAGE=${ORFS_IMAGE:-openroad/orfs:latest}
LOG="$PROJECT/logs/${design}_${variant}.log"
TIMING="$PROJECT/logs/${design}_${variant}.timing"
RAW="$PROJECT/data/raw"
mkdir -p "$PROJECT/logs" "$RAW" "$ORFS_WORK"

make_args="DESIGN_CONFIG=./designs/nangate45/$design/config.mk FLOW_VARIANT=$variant"
make_args="$make_args SKIP_INCREMENTAL_REPAIR=1 FLOORPLAN_DEF= ${extra[*]:-}"

in_container() {
  docker run --rm -v "$ORFS_WORK:/work" -v "$PROJECT:/project" \
    -e WORK_HOME=/work "$IMAGE" bash -c "cd /OpenROAD-flow-scripts/flow && source ../env.sh >/dev/null && $1" \
    >> "$LOG" 2>&1
}

if [ "${EXTRACT_ONLY:-0}" != 1 ]; then
  for d in results logs objects reports; do
    rm -rf "${ORFS_WORK:?}/${d:?}/nangate45/${design:?}/${variant:?}"
  done
  rm -f "$RAW/sweep_${design}__${variant}_congestion.npz" "$RAW/sweep_${design}__${variant}_placement.npz"
  : > "$LOG"

  t0=$(date +%s.%N)
  in_container "make $make_args synth"
  synth_rc=$?
  t1=$(date +%s.%N)
  pnr_rc=1
  if [ $synth_rc -eq 0 ]; then
    in_container "make $make_args grt"
    pnr_rc=$?
  fi
  t2=$(date +%s.%N)

  {
    echo "design=$design"
    echo "variant=$variant"
    echo "make_args=$make_args"
    echo "synth_rc=$synth_rc"
    echo "grt_rc=$pnr_rc"
    echo "synth_s=$(echo "$t1 - $t0" | bc)"
    echo "place_to_grt_s=$(echo "$t2 - $t1" | bc)"
    echo "total_s=$(echo "$t2 - $t0" | bc)"
  } > "$TIMING"
else
  rm -f "$RAW/sweep_${design}__${variant}_congestion.npz" "$RAW/sweep_${design}__${variant}_placement.npz"
  sed -i '/^grt_odb=/d; /^status=/d' "$TIMING"
fi

res=/work/results/nangate45/$design/$variant
grt_odb=$res/5_1_grt.odb
if [ ! -f "$ORFS_WORK/results/nangate45/$design/$variant/5_1_grt.odb" ]; then
  grt_odb=$res/5_1_grt-failed.odb
fi
echo "grt_odb=$grt_odb" >> "$TIMING"

# Extraction: this image's openroad has no Python, so the Tcl ports dump
# text and the venv converts it to the same .npz the .py extractors wrote.
stem=sweep_${design}__${variant}
in_container "EXTRACT_ODB=$grt_odb EXTRACT_OUT=/project/data/raw/${stem}_congestion.txt \
  openroad -no_init -no_splash -exit /project/src/extract_congestion.tcl && \
  EXTRACT_ODB=$res/3_place.odb EXTRACT_OUT=/project/data/raw/${stem}_placement.txt \
  openroad -no_init -no_splash -exit /project/src/extract_placement.tcl"
"$PROJECT/.venv/bin/python" "$PROJECT/src/odb_text_to_npz.py" congestion \
  "$RAW/${stem}_congestion.txt" "$RAW/${stem}_congestion.npz" >> "$LOG" 2>&1
"$PROJECT/.venv/bin/python" "$PROJECT/src/odb_text_to_npz.py" placement \
  "$RAW/${stem}_placement.txt" "$RAW/${stem}_placement.npz" >> "$LOG" 2>&1

if [ -f "$RAW/sweep_${design}__${variant}_congestion.npz" ] && [ -f "$RAW/sweep_${design}__${variant}_placement.npz" ]; then
  echo "status=OK" >> "$TIMING"
  exit 0
fi
echo "status=FAIL" >> "$TIMING"
exit 1
