# Tcl port of extract_congestion.py, for OpenROAD builds without Python
# (the openroad/orfs:latest image used for Phase 14 has no `-python` flag
# and no odb Python module anywhere - see notes/phase14-real-corpus.md).
#
# Same odb calls, same order: dbGCellGrid getGridX/getGridY, then for every
# ROUTING layer (tech getLayers order) and every (x, y), getUsage/getCapacity.
# Writes raw per-layer values as text; src/odb_text_to_npz.py congestion
# aggregates them by direction exactly as extract_congestion.py does and
# writes the identical .npz schema.
#
# Usage (inside the ORFS container):
#   EXTRACT_ODB=<grt.odb> EXTRACT_OUT=<out.txt> \
#     openroad -no_init -no_splash -exit src/extract_congestion.tcl

read_db $::env(EXTRACT_ODB)
set block [ord::get_db_block]
set tech [ord::get_db_tech]
set grid [$block getGCellGrid]
if { $grid == "NULL" } {
  error "$::env(EXTRACT_ODB) has no GCell grid - was global_route run?"
}

set x_grid [$grid getGridX]
set y_grid [$grid getGridY]
set nx [llength $x_grid]
set ny [llength $y_grid]

set f [open $::env(EXTRACT_OUT) w]
puts $f "meta\tsource_odb\t$::env(EXTRACT_ODB)"
puts $f "meta\tdbu_per_micron\t[$block getDbUnitsPerMicron]"
puts $f "grid_x\t[join $x_grid \t]"
puts $f "grid_y\t[join $y_grid \t]"

foreach layer [$tech getLayers] {
  if { [$layer getType] != "ROUTING" } {
    continue
  }
  set dir [$layer getDirection]
  for {set xi 0} {$xi < $nx} {incr xi} {
    for {set yi 0} {$yi < $ny} {incr yi} {
      puts $f "usage\t[$layer getName]\t$dir\t$xi\t$yi\t[$grid getUsage $layer $xi $yi]\t[$grid getCapacity $layer $xi $yi]"
    }
  }
}
close $f
puts "Wrote $::env(EXTRACT_OUT): grid ${nx}x${ny}"
