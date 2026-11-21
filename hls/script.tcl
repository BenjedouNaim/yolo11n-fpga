# Vitis HLS flow for conv_ip (KV260 / XCK26, 200 MHz).
#
# Run from the hls/ folder. Vitis 2025.2 (vitis-run has no Tcl arguments, so options come
# from environment variables):
#   set HLS_VER=v0 & vitis-run --mode hls --tcl script.tcl              v0: csim + csynth
#   set HLS_VER=v1 & set HLS_COSIM=1 & vitis-run --mode hls --tcl script.tcl
#                                                                       v1: + cosim + IP export
# Older releases: vitis_hls -f script.tcl -tclargs v1 cosim

set ROOT  [file normalize [file join [file dirname [info script]] ..]]
set DATA  [file join $ROOT data]
set ver   [expr {[info exists ::env(HLS_VER)] ? $::env(HLS_VER) : ([lsearch $argv v0] >= 0 ? "v0" : "v1")}]
set do_cosim [expr {([info exists ::env(HLS_COSIM)] && $::env(HLS_COSIM) eq "1") || [lsearch $argv cosim] >= 0}]
puts "conv_ip: version $ver, cosim $do_cosim"
set CFLAGS "-std=c++17"
set TBFLAGS [expr {$ver eq "v1" ? "$CFLAGS -DCONV_IP_V1" : $CFLAGS}]

open_project -reset proj_conv_ip_$ver
set_top conv_ip
add_files src/conv_ip_$ver.cpp -cflags $CFLAGS
add_files -tb tb/tb.cpp -cflags $TBFLAGS
foreach f {model.cpp ops.cpp postprocess.cpp} {
    add_files -tb [file join $ROOT cpp_ref $f] -cflags $TBFLAGS
}

open_solution -reset sol_$ver -flow_target vivado
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default

# Under Vitis the testbench runs --unit: the IP alone on 4 YOLO11n layer shapes covering every mode
# (the full network is far too slow with Vitis's csim instrumentation). The full-network check
# (174 IP calls, detections vs PyTorch) runs natively: hls/tb/build_csim.bat.
csim_design -O -argv "$DATA --unit"
csynth_design
file mkdir [file join $ROOT reports]
file copy -force proj_conv_ip_$ver/sol_$ver/syn/report/conv_ip_csynth.rpt [file join $ROOT reports conv_ip_${ver}_csynth.rpt]

if {$do_cosim} {
    # Reduced layer sizes, no waveform: XSIM's UVM AXI models crash on long, traced transactions
    # on a 16 GB machine. Set HLS_TRACE=1 to dump port waveforms.
    set trace [expr {[info exists ::env(HLS_TRACE)] && $::env(HLS_TRACE) eq "1" ? "port" : "none"}]
    cosim_design -O -argv "$DATA --unit-small" -trace_level $trace
    file mkdir [file join $ROOT ip_export]
    export_design -format ip_catalog -output [file join $ROOT ip_export conv_ip_$ver.zip]
}
exit
