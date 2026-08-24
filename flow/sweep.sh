#!/usr/bin/env bash
# Phase 6 sweep driver, v2. Runs from ~/congestion-gnn/OpenROAD-flow-scripts/flow
# (WSL, the ORFS checkout's bind-mounted directory) - NOT from the project
# repo, since it needs to be next to util/docker_shell and the Makefile.
#
# v1 swept (density, layer-cap tier, seed) and found that 30/45 samples
# (the "mid"/"tight" layer-cap tiers) produced BIT-IDENTICAL congestion
# ground truth regardless of density/seed, despite placement genuinely
# differing - see notes/phase6-mini-dataset.md. Layer-cap tier is now
# FIXED at "tight" (metal2-metal3 - the config Phase 3 proved actually
# produces real overflow) and BLOCKAGE BAND POSITION is swept instead,
# alongside density and seed.
#
# v2 swept (density, band position, seed) and found PLACE_DENSITY *also*
# produces bit-identical placement regardless of target (confirmed at
# both default and CORE_UTILIZATION=70 core sizing - RePlAce converges to
# the same solution for this design's ~500 cells regardless of density
# target, when there's this much slack either way). PLACE_DENSITY is now
# FIXED at 0.70 and BLOCKAGE BAND WIDTH is swept instead of density -
# same proven mechanism as band position, just its other parameter.
#
# For each (band_center, band_width, seed) combination: runs `make grt`,
# then extracts congestion + placement, in a single container invocation
# (both /work and /project mounted) to cut per-sample container startup
# overhead roughly in third versus three separate `docker run`s.
#
# Usage: bash sweep.sh
set -uo pipefail

PROJECT="/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
FLOW_DIR="$HOME/congestion-gnn/OpenROAD-flow-scripts/flow"
SUMMARY="$PROJECT/data/raw/sweep_summary.tsv"

cd "$FLOW_DIR" || exit 1
cp "$PROJECT/flow/add_die_blockage.tcl" "$FLOW_DIR/add_die_blockage.tcl"

echo -e "sample\tband_center\tband_width\tseed\tstatus" > "$SUMMARY"

PLACE_DENSITY_FIXED="0.70"
BAND_CENTERS="0.25 0.50 0.75"
BAND_WIDTHS="0.10 0.20 0.30"
SEEDS="1 2 3 4 5"

for band_center in $BAND_CENTERS; do
  for band_width in $BAND_WIDTHS; do
    for seed in $SEEDS; do
      bcpct=$(printf "%.0f" "$(echo "$band_center * 100" | bc)")
      bwpct=$(printf "%.0f" "$(echo "$band_width * 100" | bc)")
      sample="bc${bcpct}_bw${bwpct}_s${seed}"

      rm -rf "results/nangate45/gcd/$sample" "logs/nangate45/gcd/$sample" \
             "objects/nangate45/gcd/$sample" "reports/nangate45/gcd/$sample"

      docker run --rm -u 1000:1000 \
        -v "$FLOW_DIR:/work" \
        -v "$PROJECT:/project" \
        -e FLOW_HOME=/OpenROAD-flow-scripts/flow/ -e WORK_HOME=/work \
        openroad/orfs:latest bash -c "
          set -e
          cd /OpenROAD-flow-scripts/flow
          source ../env.sh
          make grt FLOW_VARIANT=$sample PLACE_DENSITY=$PLACE_DENSITY_FIXED GPL_RANDOM_SEED=$seed \
            MAX_ROUTING_LAYER=metal3 MIN_CLK_ROUTING_LAYER=metal2 \
            PRE_GLOBAL_ROUTE_TCL=/work/add_die_blockage.tcl \
            BLOCKAGE_BAND_CENTER=$band_center BLOCKAGE_BAND_WIDTH=$band_width \
            > /work/logs/nangate45/gcd/${sample}_sweep.log 2>&1 || true

          ODB=/work/results/nangate45/gcd/$sample/5_1_grt.odb
          if [ ! -f \"\$ODB\" ]; then
            ODB=/work/results/nangate45/gcd/$sample/5_1_grt-failed.odb
          fi
          PLACE_ODB=/work/results/nangate45/gcd/$sample/3_place.odb

          if [ -f \"\$ODB\" ] && [ -f \"\$PLACE_ODB\" ]; then
            openroad -python \"/project/src/extract_congestion.py\" \"\$ODB\" \
              \"/project/data/raw/sweep_${sample}_congestion.npz\" \
              >> /work/logs/nangate45/gcd/${sample}_sweep.log 2>&1
            openroad -python \"/project/src/extract_placement.py\" \"\$PLACE_ODB\" \
              \"/project/data/raw/sweep_${sample}_placement.npz\" \
              >> /work/logs/nangate45/gcd/${sample}_sweep.log 2>&1
            echo OK
          else
            echo MISSING_ODB
          fi
        " > /tmp/sweep_step_result.txt 2>&1

      result=$(tail -1 /tmp/sweep_step_result.txt)
      if [ "$result" = "OK" ] && [ -f "$PROJECT/data/raw/sweep_${sample}_placement.npz" ]; then
        status="OK"
      else
        status="FAIL"
      fi

      echo -e "${sample}\t${band_center}\t${band_width}\t${seed}\t${status}" >> "$SUMMARY"
      echo "[$sample] $status"
    done
  done
done

echo "Sweep complete. Summary: $SUMMARY"
