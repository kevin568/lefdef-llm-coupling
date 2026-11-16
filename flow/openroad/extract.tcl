# Parasitic extraction (OpenRCX). Env: LEFS, DEF_IN, RULES (RCX rules file), SPEF_OUT, DEF_OUT (optional re-write).
foreach l $::env(LEFS) { read_lef $l }
read_def $::env(DEF_IN)
if {[info exists ::env(DEF_OUT)] && $::env(DEF_OUT) != ""} { write_def $::env(DEF_OUT) }
define_process_corner -ext_model_index 0 X
extract_parasitics -ext_model_file $::env(RULES) -max_res 0 -coupling_threshold 0.1
write_spef $::env(SPEF_OUT)
puts "FLOW_DONE"
