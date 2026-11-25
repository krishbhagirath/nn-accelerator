# Quartus build for Intel (Altera) FPGAs: synthesis, place & route, timing.
# run_pipeline.py runs it as:
#     quartus_sh -t synth/quartus.tcl <NUM_MACS> <rtl dir> [family] [device]
# Default device: Cyclone IV E EP4CE115F29C7 (Terasic DE2-115 board, 50 MHz clock).
#
# Like the ECP5 flow, this is out of context: every port is a virtual pin, so the
# numbers are the accelerator's own size and speed, not a full board design.

load_package flow

set num_macs [lindex $argv 0]
set rtl_dir  [lindex $argv 1]
set family   [expr {[llength $argv] > 2 ? [lindex $argv 2] : "Cyclone IV E"}]
set device   [expr {[llength $argv] > 3 ? [lindex $argv 3] : "EP4CE115F29C7"}]

project_new nn_accelerator -overwrite
set_global_assignment -name FAMILY $family
set_global_assignment -name DEVICE $device
set_global_assignment -name TOP_LEVEL_ENTITY nn_accelerator_top
foreach f [glob -directory $rtl_dir *.sv] {
    set_global_assignment -name SYSTEMVERILOG_FILE $f
}
set_parameter -name NUM_MACS $num_macs
set_instance_assignment -name VIRTUAL_PIN ON -to *
set_global_assignment -name SDC_FILE [file join [file dirname [info script]] quartus.sdc]

execute_module -tool map
execute_module -tool fit
execute_module -tool sta
project_close
