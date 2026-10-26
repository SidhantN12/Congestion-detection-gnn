#!/usr/bin/env bash
# Phase 14 blocked-layer sweep on one design: clone a finished run (same
# placement, same CTS), then re-run ONLY global route with the band
# blockage on a different upper-layer range, and extract.
#
# Usage: bash flow/blockage_layer_sweep.sh <design> <base_variant> <min_layer>:<tag> ...
#   e.g. bash flow/blockage_layer_sweep.sh dynamic_node g1 metal5:x2 metal6:x3
set -uo pipefail

design=$1
base=$2
shift 2
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
ORFS_WORK=${ORFS_WORK:-/root/orfs_work}

for spec in "$@"; do
  min_layer=${spec%%:*}
  tag=${spec##*:}
  for d in results logs objects reports; do
    rm -rf "${ORFS_WORK:?}/${d:?}/nangate45/${design:?}/${tag:?}"
    cp -a "$ORFS_WORK/$d/nangate45/$design/$base" "$ORFS_WORK/$d/nangate45/$design/$tag"
  done
  rm -f "$ORFS_WORK/results/nangate45/$design/$tag/5_1_grt.odb" "$ORFS_WORK/results/nangate45/$design/$tag/5_1_grt.sdc"
  start=$(date +%s)
  docker run --rm -v "$ORFS_WORK:/work" -v "$PROJECT:/project" -e WORK_HOME=/work openroad/orfs:latest \
    bash -c "cd /OpenROAD-flow-scripts/flow && source ../env.sh >/dev/null && \
      make DESIGN_CONFIG=./designs/nangate45/$design/config.mk FLOW_VARIANT=$tag \
        SKIP_INCREMENTAL_REPAIR=1 FLOORPLAN_DEF= GLOBAL_ROUTE_ARGS=-allow_congestion \
        PRE_GLOBAL_ROUTE_TCL=/project/flow/add_die_blockage.tcl \
        BLOCKAGE_BAND_CENTER=0.5 BLOCKAGE_BAND_WIDTH=0.2 \
        BLOCKAGE_MIN_LAYER=$min_layer BLOCKAGE_MAX_LAYER=metal10 grt" \
    > "$PROJECT/logs/${design}_${tag}.log" 2>&1
  echo "$tag $min_layer-metal10 grt_rc=$? secs=$(( $(date +%s) - start ))"
  touch "$PROJECT/logs/${design}_${tag}.timing"
  EXTRACT_ONLY=1 bash "$PROJECT/flow/run_flow.sh" "$design" "$tag"
  echo "$tag extract_rc=$?"
done
