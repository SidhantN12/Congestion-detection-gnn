#!/usr/bin/env bash
# Phase 14 routing-layer-cap sweep on one design: clone a finished run
# (same placement and CTS), re-run ONLY global route with the stack capped
# by flow/cap_routing_layers.tcl, and extract.
#
# Usage: bash flow/layer_cap_sweep.sh <design> <base_variant> <max_layer>:<tag> ...
#   e.g. bash flow/layer_cap_sweep.sh dynamic_node g1 metal4:k4 metal5:k5 metal6:k6
# EXTRA_MAKE='VAR=value ...' appends make variables (GLOBAL_ROUTE_ARGS can be overridden there).
set -uo pipefail

design=$1
base=$2
shift 2
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
ORFS_WORK=${ORFS_WORK:-/root/orfs_work}

for spec in "$@"; do
  max_layer=${spec%%:*}
  tag=${spec##*:}
  for d in results logs objects reports; do
    rm -rf "${ORFS_WORK:?}/${d:?}/nangate45/${design:?}/${tag:?}"
    cp -a "$ORFS_WORK/$d/nangate45/$design/$base" "$ORFS_WORK/$d/nangate45/$design/$tag"
  done
  rm -f "${ORFS_WORK:?}/results/nangate45/${design:?}/${tag:?}/5_1_grt.odb" "${ORFS_WORK:?}/results/nangate45/${design:?}/${tag:?}/5_1_grt.sdc"
  start=$(date +%s)
  docker run --rm -v "$ORFS_WORK:/work" -v "$PROJECT:/project" -e WORK_HOME=/work openroad/orfs:latest \
    bash -c "cd /OpenROAD-flow-scripts/flow && source ../env.sh >/dev/null && \
      make DESIGN_CONFIG=./designs/nangate45/$design/config.mk FLOW_VARIANT=$tag \
        SKIP_INCREMENTAL_REPAIR=1 FLOORPLAN_DEF= GLOBAL_ROUTE_ARGS=-allow_congestion \
        PRE_GLOBAL_ROUTE_TCL=/project/flow/cap_routing_layers.tcl CAP_MAX_LAYER=$max_layer ${EXTRA_MAKE:-} grt" \
    > "$PROJECT/logs/${design}_${tag}.log" 2>&1
  echo "$tag cap metal2-$max_layer grt_rc=$? secs=$(( $(date +%s) - start ))"
  touch "$PROJECT/logs/${design}_${tag}.timing"
  EXTRACT_ONLY=1 bash "$PROJECT/flow/run_flow.sh" "$design" "$tag"
  echo "$tag extract_rc=$?"
done
