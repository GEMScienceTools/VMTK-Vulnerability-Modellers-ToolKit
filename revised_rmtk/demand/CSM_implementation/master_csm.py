#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Variable definitions and user inputs
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% selectionParams       : parameters controlling how the ground motion 
#%                         selection is performed
#%           .databaseFile : filename of the target database. This file should exist 
#%                           in the 'Databases' subfolder. Further documentation of 
#%                           these databases can be found at 
#%                           'Databases/WorkspaceDocumentation***.txt'.
#%           .cond       : 0 to run unconditional selection
#%                         1 to run conditional
#%           .arb        : 1 for single-component selection and arbitrary component sigma
#%                         2 for two-component selection and average component sigma
#%           .RotD       : 50 to use SaRotD50 data
#%                       : 100 to use SaRotD100 data
#%           .isScaled   : =1 to allow records to be scaled, =0 otherwise 
#%                         (note that the algorithm is slower when .isScaled
#%                         = 1)
#%           .maxScale   : The maximum allowable scale factor
#%           .tol        : Tolerable percent error to skip optimization 
#%           .optType    : =0 to use the sum of squared errors to 
#%                         optimize the selected spectra, =1 to use 
#%                         D-statistic calculations from the KS-test
#%                         (the algorithm is slower when .optType
#%                         = 1)
#%           .penalty    : >0 to penalize selected spectra more than 
#%                         3 sigma from the target at any period, 
#%                         =0 otherwise.
#%          . weights    : [Weights for error in mean, standard deviation 
#%                         and skewness] e.g., [1.0,2.0 0.3] 
#%           .nLoop      : Number of loops of optimization to perform.
#%           .nBig       : The number of spectra that will be searched
#%           .indTcond   : Index of Tcond, the conditioning period
#%           .recID      : Vector of index values for the selected
#%                         spectra
#% 
#% rup                   :  A structure with parameters that specify the rupture scenario
#%                          for the purpose of evaluating a GMPE. Here we
#%                          use the following parameters
#%           .M_bar            : earthquake magnitude
#%           .Rjb              : closest distance to surface projection of the fault rupture (km)
#%           .Fault_Type       : =0 for unspecified fault
#%                               =1 for strike-slip fault
#%                               =2 for normal fault
#%                               =3 for reverse fault
#%           .region           : =0 for global (incl. Taiwan)
#%                               =1 for California 
#%                               =2 for Japan 
#%                               =3 for China or Turkey 
#%                               =4 for Italy
#%           .z1               : basin depth (km); depth from ground surface to the 1km/s shear-wave horizon, =999 if unknown
#%           .Vs30             : average shear wave velocity in the top 30m of the soil (m/s)
#%
#% targetSa              :  Response spectrum target values to match
#%           .meanReq            : Estimated target response spectrum means (vector of
#%                                 logarithmic spectral values, one at each period)
#%           .covReq             : Matrix of response spectrum covariances
#%           .stdevs             : A vector of standard deviations at each period
#% 
#% IMs                   :  The intensity measure values (from SaKnown) chosen and the 
#%                           values available
#%           .recID              : indices of selected spectra
#%           .scaleFac           : scale factors for selected spectra
#%           .sampleSmall        : matrix of selected logarithmic response spectra 
#%           .sampleBig          : The matrix of logarithmic spectra that will be 
#%                                 searched
#%           .stageOneScaleFac   : scale factors for selected spectra, after
#%                                 the first stage of selection
#%           .stageOneMeans      : mean log response spectra, after
#%                                 the first stage of selection
#%           .stageOneStdevs     : standard deviation of log response spectra, after
#%                                 the first stage of selection
#%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%
#% OUTPUT VARIABLES
#%
#% IMs.recID      : Record numbers of selected records
#% IMs.scaleFac   : Corresponding scale factors
#%
#% (these variables are also output to a text file in write_output.m)
#%
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

[SaKnown,selectionParams,indPer,knownPer,metadata] =cpi.screen_database(selectionParams,allowedRecs)

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%