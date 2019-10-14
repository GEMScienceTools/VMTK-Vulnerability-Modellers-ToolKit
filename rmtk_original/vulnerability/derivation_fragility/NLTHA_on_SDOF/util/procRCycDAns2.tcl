#####################################################################################################
# #
# procRCycDAns2.tcl #
# procedure for reverse cyclic displacement control analysis given the peak pts. #
# analysis type used : STATIC #
# Written : N.Mitra #
# Modified 1/Nov/2005 by D.Vamvatsikos: Now it is much more stable, as it uses RunPushover2Converge
#          to make sure that the displacement control analysis always converges right!           
#####################################################################################################

proc procRCycDAns2 { incre nodeTag dofTag peakpts} { 

  set x [lindex $peakpts 0]
  set fir [expr $x/$incre]
  
  integrator DisplacementControl $nodeTag $dofTag 0.0 1 $fir $fir
  # create the analysis object
  analysis Static
  # perform the analysis
  RunPushover2Converge  $nodeTag $x $incre
  integrator DisplacementControl $nodeTag $dofTag 0.0 1 [expr -$fir] [expr -$fir]
  RunPushover2Converge  $nodeTag [expr -$x] [expr 2*$incre] 1 1
  integrator DisplacementControl $nodeTag $dofTag 0.0 1 $fir $fir
  RunPushover2Converge  $nodeTag 0.0 $incre 1 1

for {set j 1} {$j < [llength $peakpts]} {incr j 1} {
    set tx [lindex $peakpts $j]
    set tinc [expr $tx/$fir]
    set rt [expr int($tinc)]
    integrator DisplacementControl $nodeTag $dofTag 0.0 1 $fir $fir
    RunPushover2Converge  $nodeTag $tx $rt
    integrator DisplacementControl $nodeTag $dofTag 0.0 1 [expr -$fir] [expr -$fir]
    RunPushover2Converge  $nodeTag [expr -$tx] [expr 2*$rt] 1 1
    integrator DisplacementControl $nodeTag $dofTag 0.0 1 $fir $fir
    RunPushover2Converge  $nodeTag 0.0 $rt  1 1
  }

################################ end procRCycDAns.tcl #######################################
}
