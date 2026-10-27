# Custom PRE_GLOBAL_ROUTE_TCL hook: block routing over a horizontal band
# across the full die width, on every routing layer in the currently
# capped stack (MIN_ROUTING_LAYER-MAX_ROUTING_LAYER), to force detours.
# Design is already placed+CTS'd when this hook fires (global_route.tcl
# calls source_step_tcl PRE GLOBAL_ROUTE right after load_design).
#
# Band vertical position is parametrized via BLOCKAGE_BAND_CENTER (0.0-1.0
# fraction of die height, default 0.50) and BLOCKAGE_BAND_WIDTH (fraction,
# default 0.20) - Phase 6 sweeps BLOCKAGE_BAND_CENTER across samples so
# each gets a distinct, but equally real, forced congestion hotspot
# (Phase 3 found layer-cap/density alone don't reliably move congestion;
# only an explicit blockage does).

set db [::ord::get_db]
set block [[$db getChip] getBlock]
set tech [$db getTech]

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

set die_height [expr {$die_ury - $die_lly}]
set band_lo_frac [expr {$band_center - $band_width / 2.0}]
set band_hi_frac [expr {$band_center + $band_width / 2.0}]
set band_lly [expr {$die_lly + int($die_height * $band_lo_frac)}]
set band_ury [expr {$die_lly + int($die_height * $band_hi_frac)}]

# Blocked layer range: BLOCKAGE_MIN_LAYER/BLOCKAGE_MAX_LAYER, defaulting to
# the whole routing stack (Phase 3/6 behaviour). Phase 14 blocks only the
# upper layers so band gcells keep non-zero capacity on metal2/metal3:
# extract_congestion reports ratio 0 wherever capacity is 0, so a fully
# blocked band would be labelled uncongested.
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

set cnt 0
for {set lvl $min_level} {$lvl <= $max_level} {incr lvl} {
  set layer [$tech findRoutingLayer $lvl]
  if { $layer == "NULL" } {
    continue
  }
  odb::dbObstruction_create $block $layer $die_llx $band_lly $die_urx $band_ury
  incr cnt
}

puts "Created $cnt routing blockages over band y=\[$band_lly,$band_ury\] (die y=\[$die_lly,$die_ury\]) on layers $min_layer-$max_layer"
