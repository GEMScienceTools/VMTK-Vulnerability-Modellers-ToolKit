# This is a comment for Tcl !
# generate2DFrameNodes
# the nodes are numbered as "$story$column". So a node on column 4, 1st story 
# is 14, or 104 or 1004, depending on option "cstor" being 10, 100 or 1000.
# colDist and storHeight are lists that are reused! For example if you only provide one
# number, it will be assumed to be the same for all columns and stories.
# To use this you must have sourced "..\extral\lmath.tcl"
proc generate2DFrameNodes {nbay nstory columnDist storyHeight {cstor 10} {matfile 0}} {
  set cy 0.0
  set nvaly [llength $storyHeight]
  set nvalx [llength $columnDist]
  # start from 0 as you need a story at height==0 for ground floor and a column at x=0 too.
  # In general you need +1 column and +1 story more than the length of columnDist and storyHeight.

  for {set i 0} {$i<=$nstory} {incr i} {
    set cx 0.0
    for {set j 0} {$j<=$nbay} {incr j} {
      node [expr $i*$cstor+$j+1] $cx $cy
      if {$matfile!=0} {
        puts $matfile "node [expr $i*$cstor+$j+1] $cx $cy"
      }
      set cx [expr $cx + [lindex $columnDist [expr $j%$nvalx]]]
    }
    set cy [expr $cy + [lindex $storyHeight [expr $i%$nvaly]]]
  } 
}