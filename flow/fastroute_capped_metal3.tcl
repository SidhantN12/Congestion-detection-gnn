# Copy of platforms/nangate45/fastroute.tcl, referenced via FASTROUTE_TCL
# override, needed only because MAX_ROUTING_LAYER < the platform's default
# MIN_CLK_ROUTING_LAYER (metal4) requires MIN_CLK_ROUTING_LAYER to be
# overridden too - see notes/phase3-inducing-congestion.md.
set_global_routing_layer_adjustment metal2-metal3 0.5
set_global_routing_layer_adjustment metal4-$::env(MAX_ROUTING_LAYER) 0.25

set_routing_layers -clock $::env(MIN_CLK_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
set_routing_layers -signal $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
