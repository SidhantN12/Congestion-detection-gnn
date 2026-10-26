# Dump OpenROAD's own "Routing" congestion heatmap for an .odb to CSV -
# the independent reference tests/verify_congestion_extraction.py compares
# extract_congestion.py against (Phase 2 method, as a committed script).
#
# Usage (inside the ORFS container, headless):
#   HEATMAP_ODB=<grt.odb> HEATMAP_CSV=<out.csv> QT_QPA_PLATFORM=offscreen \
#     openroad -no_init -exit tests/dump_routing_heatmap.tcl

read_db $::env(HEATMAP_ODB)
gui::show "gui::set_heatmap Routing rebuild; gui::dump_heatmap Routing $::env(HEATMAP_CSV)" false
