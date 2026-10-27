# PRE_GLOBAL_ROUTE_TCL hook (Phase 14): cut routing CAPACITY over a
# horizontal band across the full die width, on the upper routing layers.
#
# Replaces add_die_blockage.tcl for the corpus. A hard odb obstruction is
# recorded by GRT as *usage* in dbGCellGrid, not as reduced capacity:
# measured on dynamic_node, the capacity arrays with and without the
# obstruction were bit-identical while band demand rose (V 20.10 -> 38.20).
# No input feature then marks the blockage, and band labels are mostly
# blocked tracks counted as demand. set_global_routing_region_adjustment
# reduces capacity instead, so the cut is visible in the gcell capacity
# feature and demand stays routing demand (verified in the extracted .npz,
# notes/phase14-real-corpus.md).
#
# Parameters (env, same names as add_die_blockage.tcl):
#   BLOCKAGE_BAND_CENTER  fraction of die height (default 0.50)
#   BLOCKAGE_BAND_WIDTH   fraction of die height (default 0.20)
#   BLOCKAGE_MIN_LAYER / BLOCKAGE_MAX_LAYER  adjusted layer range
#                         (default MIN_ROUTING_LAYER / MAX_ROUTING_LAYER)
#   CAPACITY_ADJUSTMENT   fraction of capacity removed, 0-1 (default 1.0)

set db [::ord::get_db]
set block [[$db getChip] getBlock]
set tech [$db getTech]
set dbu [$block getDbUnitsPerMicron]

set die [$block getDieArea]
set die_llx [$die xMin]
set die_urx [$die xMax]
set die_lly [$die yMin]
set die_ury [$die yMax]

set band_center 0.50
if { [info exists ::env(BLOCKAGE_BAND_CENTER)] && $::env(BLOCKAGE_BAND_CENTER) ne "" } {
  set band_center $::env(BLOCKAGE_BAND_CENTER)
}
set band_width 0.20
if { [info exists ::env(BLOCKAGE_BAND_WIDTH)] && $::env(BLOCKAGE_BAND_WIDTH) ne "" } {
  set band_width $::env(BLOCKAGE_BAND_WIDTH)
}
set adjustment 1.0
if { [info exists ::env(CAPACITY_ADJUSTMENT)] && $::env(CAPACITY_ADJUSTMENT) ne "" } {
  set adjustment $::env(CAPACITY_ADJUSTMENT)
}

set die_height [expr {$die_ury - $die_lly}]
set band_lly [expr {$die_lly + int($die_height * ($band_center - $band_width / 2.0))}]
set band_ury [expr {$die_lly + int($die_height * ($band_center + $band_width / 2.0))}]

set min_layer $::env(MIN_ROUTING_LAYER)
if { [info exists ::env(BLOCKAGE_MIN_LAYER)] && $::env(BLOCKAGE_MIN_LAYER) ne "" } {
  set min_layer $::env(BLOCKAGE_MIN_LAYER)
}
set max_layer $::env(MAX_ROUTING_LAYER)
if { [info exists ::env(BLOCKAGE_MAX_LAYER)] && $::env(BLOCKAGE_MAX_LAYER) ne "" } {
  set max_layer $::env(BLOCKAGE_MAX_LAYER)
}
set min_level [[$tech findLayer $min_layer] getRoutingLevel]
set max_level [[$tech findLayer $max_layer] getRoutingLevel]

# set_global_routing_region_adjustment takes the region in microns.
set region [list [expr {double($die_llx) / $dbu}] [expr {double($band_lly) / $dbu}] \
                 [expr {double($die_urx) / $dbu}] [expr {double($band_ury) / $dbu}]]

set cnt 0
for {set lvl $min_level} {$lvl <= $max_level} {incr lvl} {
  set layer [$tech findRoutingLayer $lvl]
  if { $layer == "NULL" } {
    continue
  }
  set_global_routing_region_adjustment $region -layer [$layer getName] -adjustment $adjustment
  incr cnt
}

puts "Capacity band: adjustment $adjustment on $cnt layers $min_layer-$max_layer over y=\[$band_lly,$band_ury\] DBU (region $region um, die y=\[$die_lly,$die_ury\])"
