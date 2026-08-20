# Custom PRE_GLOBAL_ROUTE_TCL hook: block routing over a horizontal band
# across the full die width, on every routing layer in the currently
# capped stack (MIN_ROUTING_LAYER-MAX_ROUTING_LAYER), to force detours.
# Design is already placed+CTS'd when this hook fires (global_route.tcl
# calls source_step_tcl PRE GLOBAL_ROUTE right after load_design).

set db [::ord::get_db]
set block [[$db getChip] getBlock]
set tech [$db getTech]

set die [$block getDieArea]
set die_llx [$die xMin]
set die_urx [$die xMax]
set die_lly [$die yMin]
set die_ury [$die yMax]

set die_height [expr {$die_ury - $die_lly}]
set band_lly [expr {$die_lly + int($die_height * 0.40)}]
set band_ury [expr {$die_lly + int($die_height * 0.60)}]

set min_layer $::env(MIN_ROUTING_LAYER)
set max_layer $::env(MAX_ROUTING_LAYER)
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
