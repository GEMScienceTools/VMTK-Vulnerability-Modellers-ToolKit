# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:05:12 2019

@author: LuisMartins
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import csm_python_implementation as cpi
import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
selectionParams={}
selectionParams["databaseFile"]="NGA_w1_meta_data"
selectionParams["cond"]=1
selectionParams["arb"]=2
selectionParams["RotD"]=50

selectionParams["nGM"]=40
selectionParams["Tcond"]=1.5
selectionParams["Tmin"]=0.1
selectionParams["Tmax"]=10
selectionParams["SaTcond"]=0.3

selectionParams["isScaled"]=1
selectionParams["maxScale"]=5
selectionParams["tol"]=0.10
selectionParams["optType"]=0
selectionParams["penalty"]=0
selectionParams["weights"]=np.array([1,2,0.3])
selectionParams["nLoop"]=2
selectionParams["useVar"]=1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rup={}
rup["M_bar"]=6.5
rup["Rjb"]=11
rup["eps_bar"]=1.9
rup["Vs30"]=259
rup["z1"]=999
rup["region"]=1
rup["Fault_Type"]=0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allowedRecs={}
allowedRecs["Vs30"]=np.array([-1*np.inf,np.inf])
allowedRecs["Mag"]=np.array([6,7])
allowedRecs["D"]=np.array([0,50])

showPlots=1
copyFiles=0
seedValue=0
nTrials=20
outputDir="data_py_implementation"
outputFile="out_file.dat"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMs={}
# Load the specified ground motion database and screen for suitable motions
selectionParams["TgtPer"]=np.logspace(np.log10(selectionParams["Tmin"]),np.log10(selectionParams["Tmax"]),30)

[SaKnown,selectionParams,indPer,knownPer,metadata] =cpi.screen_gmrs_folder(selectionParams)

temp=np.zeros([np.shape(SaKnown)[0],len(indPer)])
for i in range(np.shape(SaKnown)[0]):
      for j in range(len(indPer)):
            row_idx=i
            col_idx=indPer[j]
            temp[i][j]=np.log(SaKnown[row_idx][col_idx])

IMs["sampleBig"] = temp

# Compute target means and covariances of spectral values
targetSa=cpi.get_target_spectrum(knownPer,selectionParams,indPer,rup)

selectionParams["lnSa1"]=targetSa["meanReq"][selectionParams["indTcond"]]

# Simulate response spectra matching the computed targets
simulatedSpectra=cpi.simulate_spectra(targetSa,selectionParams,seedValue,nTrials)

# Find best matches to the simulated spectra from ground-motion database
IMs =cpi.find_ground_motions(selectionParams,simulatedSpectra,IMs)

#IMs["stageOneScaleFac"]=IMs["scaleFac"]
#IMs["stageOneMeans"]=np.mean(np.log(SaKnown[IMs["recIDs"]]*np.matlib.repmat(IMs["stageOneScaleFac"],1,np.shape(SaKnown)[1])))
#IMs["stageOneStdevs"]=np.std(np.log(SaKnown[IMs["recIDs"]]*np.matlib.repmat(IMs["stageOneScaleFac"],1,np.shape(SaKnown)[1])))

# Further optimize the ground motion selection, if needed
if cpi.within_tolerance(IMs["sampleSmall"], targetSa, selectionParams)==1:
      print("Error metric is within tolerance, skipping optimization")
else:
      IMs =cpi.optimize_ground_motions(selectionParams, targetSa, IMs)

## results
if showPlots==1:
      cpi.plot_results(selectionParams, targetSa, IMs, simulatedSpectra, SaKnown, knownPer )
      
recIdx=np.zeros([selectionParams["nGM"],1])

for i in range(selectionParams["nGM"]):
      row_idx=IMs["recIDs"][i]
      recIdx[i]=metadata["allowedIndex"][int(row_idx)]

cpi.write_output(recIdx, IMs, outputDir, outputFile, metadata)