# Simple template for a TCL file, openSEES oriented!
# ---------RunGravityAnalysis.tcl----------------------------------------------------------------
# Will simply run a basic Gravity (Load Control) Analysis with all the load patterns defined up to this time
# This routine should work for practically anything
#
proc RunGravityAnalysis {{GravSteps 10}} {
  puts "Gravity Analysis...."
  system UmfPack

  test NormDispIncr 1.0e-8 20 0
  algorithm Newton 
  numberer RCM
  integrator LoadControl [expr 1./$GravSteps] 1 [expr 1./$GravSteps] [expr 1./$GravSteps] 
  analysis Static
  #initialize;

  # run gravity analysis
  set ok [analyze $GravSteps]
  # keep gravity load and restart time -- lead to lateral-load analysis
  loadConst -time 0.0

  if {$ok!=0} {
    # this should be extremely rare!!!
    puts "Gravity Analysis Failed!"
  }
  return $ok
}