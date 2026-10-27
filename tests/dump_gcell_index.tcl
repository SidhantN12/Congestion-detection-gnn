# Phase 5 check 1, OpenROAD side: record dbGCellGrid getXIdx/getYIdx for
# every placed cell position (from extract_placement.tcl's text dump - the
# positions build_graph.py assigns), every instance in the post-route odb,
# and boundary probes on and one DBU either side of every grid line.
# tests/verify_gcell_index.py compares these against build_graph.gcell_index.
#
# Usage (inside the ORFS container):
#   GRID_ODB=<grt.odb> PLACEMENT_TXT=<placement.txt> INDEX_OUT=<out.tsv> \
#     openroad -no_init -no_splash -exit tests/dump_gcell_index.tcl

read_db $::env(GRID_ODB)
set block [ord::get_db_block]
set grid [$block getGCellGrid]
set die [$block getDieArea]
set x_grid [$grid getGridX]
set y_grid [$grid getGridY]

set out [open $::env(INDEX_OUT) w]
puts $out "grid_x\t[join $x_grid \t]"
puts $out "grid_y\t[join $y_grid \t]"

set f [open $::env(PLACEMENT_TXT) r]
while { [gets $f line] >= 0 } {
  set parts [split $line \t]
  if { [lindex $parts 0] == "cell" } {
    set x [lindex $parts 2]
    set y [lindex $parts 3]
    puts $out "placed\t$x\t$y\t[$grid getXIdx $x]\t[$grid getYIdx $y]"
  }
}
close $f

foreach inst [$block getInsts] {
  set loc [$inst getLocation]
  set x [lindex $loc 0]
  set y [lindex $loc 1]
  puts $out "grt_inst\t$x\t$y\t[$grid getXIdx $x]\t[$grid getYIdx $y]"
}

set y0 [lindex $y_grid 0]
set x0 [lindex $x_grid 0]
foreach gx [concat $x_grid [list [$die xMax]]] {
  foreach dx {-1 0 1} {
    set x [expr {$gx + $dx}]
    if { $x >= [$die xMin] && $x <= [$die xMax] } {
      puts $out "boundary_x\t$x\t$y0\t[$grid getXIdx $x]\t[$grid getYIdx $y0]"
    }
  }
}
foreach gy [concat $y_grid [list [$die yMax]]] {
  foreach dy {-1 0 1} {
    set y [expr {$gy + $dy}]
    if { $y >= [$die yMin] && $y <= [$die yMax] } {
      puts $out "boundary_y\t$x0\t$y\t[$grid getXIdx $x0]\t[$grid getYIdx $y]"
    }
  }
}
close $out
