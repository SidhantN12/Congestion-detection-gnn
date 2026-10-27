# Tcl port of extract_placement.py, for OpenROAD builds without Python
# (see extract_congestion.tcl). Same odb calls and filtering:
#   - cells: every instance, in getInsts order; pin_count = SIGNAL/CLOCK iterms
#   - nets: SIGNAL/CLOCK only, in getNets order; nets with zero iterms dropped
#   - pin edges: one per iterm of a kept net; position = iterm bbox centre
# src/odb_text_to_npz.py placement turns the text into the identical .npz
# schema (HPWL, bbox and fanout computed there exactly as the .py did).
#
# Usage (inside the ORFS container):
#   EXTRACT_ODB=<placed.odb> EXTRACT_OUT=<out.txt> \
#     openroad -no_init -no_splash -exit src/extract_placement.tcl

read_db $::env(EXTRACT_ODB)
set block [ord::get_db_block]
set rows [$block getRows]
if { [llength $rows] == 0 } {
  error "$::env(EXTRACT_ODB) has no placement rows - is this a placed odb?"
}
set site [[lindex $rows 0] getSite]
set die [$block getDieArea]

set f [open $::env(EXTRACT_OUT) w]
puts $f "meta\tdie_w\t[expr {[$die xMax] - [$die xMin]}]"
puts $f "meta\tdie_h\t[expr {[$die yMax] - [$die yMin]}]"
puts $f "meta\tsite_w\t[$site getWidth]"
puts $f "meta\tsite_h\t[$site getHeight]"
puts $f "meta\tdbu_per_micron\t[$block getDbUnitsPerMicron]"
puts $f "meta\tsource_odb\t$::env(EXTRACT_ODB)"

set idx 0
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  set loc [$inst getLocation]
  set pins 0
  foreach it [$inst getITerms] {
    set sig [$it getSigType]
    if { $sig == "SIGNAL" || $sig == "CLOCK" } {
      incr pins
    }
  }
  set cell_idx([$inst getId]) $idx
  puts $f "cell\t[$inst getName]\t[lindex $loc 0]\t[lindex $loc 1]\t[$master getWidth]\t[$master getHeight]\t$pins"
  incr idx
}

set net_idx 0
foreach net [$block getNets] {
  set sig [$net getSigType]
  if { $sig != "SIGNAL" && $sig != "CLOCK" } {
    continue
  }
  set iterms [$net getITerms]
  if { [llength $iterms] == 0 } {
    continue
  }
  puts $f "net\t[$net getName]"
  foreach it $iterms {
    set bbox [$it getBBox]
    puts $f "pin\t$cell_idx([[$it getInst] getId])\t$net_idx\t[$bbox xCenter]\t[$bbox yCenter]"
  }
  incr net_idx
}
close $f
puts "Wrote $::env(EXTRACT_OUT): $idx cells, $net_idx nets"
