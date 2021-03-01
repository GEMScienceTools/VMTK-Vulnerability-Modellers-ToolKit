# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:27:44 2019

@author: LuisMartins
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.io import loadmat
import urllib.request
import shutil
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_target_spectrum(knownPer,selectionParams,indPer,rup):
    
    #% Calculate and return the target mean spectrum, target and covariance
    #% matrix at available periods. Predicted spectral accelerations and
    #% corresponding standard deviations are computed using gmpeBSSA_2014 but
    #% can be replaced by a user-defined GMPE. If the GMPE is altered then the
    #% input arguments will likely also need to be adjusted
    #%
    #% INPUTS
    #%           knownPer        : available periods from the database
    #%           indPer          : period indicies of target response spectrum
    #%           eps_bar         : user input for epsilon (conditional
    #%                             selection)
    #% 
    #%           selectionParams : parameters controlling how the ground motion 
    #%                             selection is performed
    #%               .databaseFile : filename of the target database. This file should exist
    #%                               in the 'Databases' subfolder. Further documentation of
    #%                               these databases can be found at
    #%                               'Databases/WorkspaceDocumentation***.txt'.
    #%               .cond         : 0 to run unconditional selection
    #%                               1 to run conditional
    #%               .arb          : 1 for single-component selection and arbitrary component sigma
    #%                               2 for two-component selection and average component sigma
    #%               .RotD         : 50 to use SaRotD50 data
    #%                             : 100 to use SaRotD100 data
    #%               .isScaled     : =1 to allow records to be
    #%                               scaled, =0 otherwise
    #%               .maxScale     : The maximum allowable scale factor
    #%               .tol          : Tolerable percent error to skip optimization (only
    #%                               used for SSE optimization)
    #%               .optType      : =0 to use the sum of squared errors to
    #%                               optimize the selected spectra, =1 to use
    #%                               D-statistic calculations from the KS-test
    #%               .penalty      : >0 to penalize selected spectra more than
    #%                               3 sigma from the target at any period,
    #%                               =0 otherwise.
    #%               . weights     : [Weights for error in mean, standard deviation
    #%                               and skewness] e.g., [1.0,2.0 0.3]
    #%               .nLoop        : Number of loops of optimization to perform.
    #%               .nBig         : The number of spectra that will be searched
    #%               .indTcond     : Index of Tcond, the conditioning period
    #%               .recID        : Vector of index values for the selected
    #%                               spectra
    #%
    #%           rup             :  A structure with parameters that specify the rupture scenario
    #%                              for the purpose of evaluating a GMPE
    #%
    #%               .M_bar            : earthquake magnitude
    #%               .Rjb              : closest distance to surface projection of the fault rupture (km)
    #%               .Fault_Type       : =0 for unspecified fault
    #%                                   =1 for strike-slip fault
    #%                                   =2 for normal fault
    #%                                   =3 for reverse fault
    #%               .region           : =0 for global (incl. Taiwan)
    #%                                   =1 for California 
    #%                                   =2 for Japan 
    #%                                   =3 for China or Turkey 
    #%                                   =4 for Italy
    #%               .z1               : basin depth (km); depth from ground surface to the 1km/s shear-wave horizon, =999 if unknown
    #%               .Vs30             : average shear wave velocity in the top 30m of the soil (m/s)
    #%
    #%
    #% Outputs (these could be replaced by user-specified matrices if desired
    #%                 targetSa.meanReq = target mean log Sa; 
    #%                 targetSa.covReq  = target coveriance matrix for log Sa;
    #%                 targetSa.stdevs  = target standard deviations for log Sa;
    
    
    [sa,sigma]=gmpe_bssa_2014(rup["M_bar"],knownPer,rup["Rjb"],rup["Fault_Type"],rup["region"], rup["z1"],rup["Vs30"])
    
    if selectionParams["RotD"]==100 and selectionParams["arb"]==2:
        [rotD100Ratio,rotD100Sigma]=gmpeSB_2014_ratios(knownPer)
        sa=sa*rotD100Ratio
        sigma=np.sqrt(sigma**2+rotD100Sigma**2)
        
    if "SaTcond" in selectionParams:
        medianSaTcond=np.exp(np.interp(np.log(selectionParams["Tcond"]),np.log(knownPer),np.log(sa)))
        sigmaSaTcond=np.exp(np.interp(np.log(selectionParams["Tcond"]),np.log(knownPer),np.log(sigma)))
        eps_bar=(np.log(selectionParams["SaTcond"])-np.log(medianSaTcond))/sigmaSaTcond
    else:
        eps_bar=rup["eps_bar"]
        
    if selectionParams["cond"]==1:
        rho=np.zeros([len(sa)])
        for i in range(len(sa)):
            rho[i]=gmpe_bj_2008_corr(knownPer[i], selectionParams["TgtPer"][selectionParams["indTcond"]])
            
        TgtMean=np.log(sa)+(np.array(sigma)*np.array(eps_bar)*np.array(rho))
        
    elif selectionParams["cond"]==0:
         TgtMean=np.log(sa)   
         
    TgtCovs=np.zeros([len(sa),len(sa)])
    for i in range(len(sa)):
         for j in range(len(sa)):
             Ti=knownPer[i]
             Tj=knownPer[j]
             
             varT=(sigma[selectionParams["indTcond"]])**2
             sigma22=varT
             var1=sigma[i]**2
             var2=sigma[j]**2
             
             if selectionParams["cond"]==1:
                 sigmaCorr=(gmpe_bj_2008_corr(Ti,Tj))*math.sqrt(var1*var2)
                 sigma11=np.array([[var1,sigmaCorr],[sigmaCorr,var2]])
                 rho1=gmpe_bj_2008_corr(Ti, selectionParams["Tcond"])*np.sqrt(var1*varT)
                 rho2=gmpe_bj_2008_corr(Tj, selectionParams["Tcond"])*np.sqrt(var2*varT)
                 sigma12=np.array([[rho1],[rho2]])
                 sigmaCond=sigma11-np.matmul(sigma12,np.transpose(sigma12))*(1/(sigma22))
                 TgtCovs[i,j]=sigmaCond[0,1]
             elif selectionParams["cond"]==0:
                   rho=gmpe_bj_2008_corr(Ti, Tj)
                   TgtCovs[i,j] = rho*np.sqrt(var1*var2)
                 
    if selectionParams["useVar"]==0:
         TgtCovs=np.zeros(np.shape(TgtCovs))
         
    TgtCovs[abs(TgtCovs)<1e-10]=1e-10
     
    targetSa={}
    temp=np.zeros([len(indPer),len(indPer)])
    for i in range(len(indPer)):
          for j in range(len(indPer)):
                temp[i,j]=TgtCovs[indPer[i],indPer[j]]
    
    targetSa["meanReq"]=TgtMean[indPer]
    targetSa["covReq"]=temp
    targetSa["stdevs"]=np.sqrt(np.diag(targetSa["covReq"]))
    targetSa["meanAllT"]=TgtMean
    targetSa["covAllT"]=TgtCovs   
                 
    return targetSa
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gmpe_bj_2008_corr(T1,T2):
    
    T_min=min([T1,T2])
    T_max=max([T1,T2])
    
    C1=(1-math.cos(math.pi*(1/2)-math.log(T_max/max([T_min,0.109]))*0.366))
    
    if T_max<0.2:
        C2=1-0.105*(1-1/(1+math.exp(100*T_max-5)))*(T_max-T_min)/(T_max-0.0099)
        
    if T_max<0.109:
        C3=C2
    else:
        C3=C1
        
    C4=C1+0.5*(math.sqrt(C3)-C3)*(1+math.cos(math.pi*(T_min)/0.109))
        
    if T_max<=0.109:
        rho=C2
    elif T_min>0.109:
        rho=C1
    elif T_max<0.2:
        rho=min([C2,C4])
    else:
        rho=C4

    return rho
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gmpeSB_2014_ratios(T):
    
    periods_orig =np.array([0.0100000000000000,0.0200000000000000, 0.0300000000000000,0.0500000000000000,0.0750000000000000,0.100000000000000,0.150000000000000,0.200000000000000,0.250000000000000,0.300000000000000,0.400000000000000,0.500000000000000,0.750000000000000,1,1.50000000000000,2,3,4,5,7.50000000000000,10])
    ratios_orig =np.array([1.19243805900000,1.19124621700000,1.18767783300000,1.18649074900000,1.18767783300000,1.18767783300000,1.19961419400000,1.20562728500000,1.21652690500000,1.21896239400000,1.22875320400000,1.22875320400000,1.23738465100000,1.24110237900000,1.24234410200000,1.24358706800000,1.24732343100000,1.25985923900000,1.264908769000,1.28531008400000,1.29433881900000])
    sigma_orig = np.array([0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08])
    phi_orig =np.array([0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.07])
    tau_orig =np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03])
    
    ratio=np.interp(np.log(T),np.log(periods_orig),ratios_orig)
    sigma=np.interp(np.log(T),np.log(periods_orig),sigma_orig)
    phi=np.interp(np.log(T),np.log(periods_orig),phi_orig)
    tau=np.interp(np.log(T),np.log(periods_orig),tau_orig)
    
    return ratio, sigma, phi, tau 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gmpe_bssa_2014(M,T,Rjb,Fault_Type,region,z1,Vs30):
    
#     Input Variables
#     M = Moment Magnitude
#     T = Period (sec); Use Period = -1 for PGV computation
#                     Use 1000 to output the array of median with original period
#                     (no interpolation)
#     Rjb = Joyner-Boore distance (km)
#     Fault_Type    = 0 for unspecified fault
#                   = 1 for strike-slip fault
#                   = 2 for normal fault
#                   = 3 for reverse fault
#     region        = 0 for global (incl. Taiwan)
#                   = 1 for California
#                   = 2 for Japan
#                   = 3 for China or Turkey
#                   = 4 for Italy
#     z1            = Basin depth (km); depth from the groundsurface to the
#                       1km/s shear-wave horizon.
#                   = 999 if unknown
#     Vs30          = shear wave velocity averaged over top 30 m in m/s
#    
#     Output Variables
#     median        = Median amplitude prediction
#    
#     sigma         = NATURAL LOG standard deviation 
    
    period=np.array([-1,0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,7.5,10])
    
    if Fault_Type==0:
        U=1
        SS=0
        NS=0
        RS=0
    elif Fault_Type==1:
        U=0
        SS=1
        NS=0
        RS=0
    elif Fault_Type==2:
        U=0
        SS=0
        NS=1
        RS=0
    elif Fault_Type==3:
        U=0
        SS=0
        NS=0
        RS=1
    else:
        print('Fault type not included')
        
    if len(T)==1 and T==1000:
        median=np.zeros([len(period)-2])
        sigma=np.zeros([len(period)-2])
        period1=np.asarray([period[i] for i in range(2,len(period))])
        for ip in range(2,len(period)):
            [median[ip-2],sigma[ip-2]]=BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30)
    else:
        median=np.zeros([len(T)])
        sigma=np.zeros([len(T)])
        period1=T
        for i in range(len(T)):
            Ti=T[i]
            aux=abs(period-Ti)
            idx=[i for (i, val) in enumerate(aux) if val<1e-4]
            if len(idx)==0:
                T_low=max(period[period<Ti])
                T_high=min(period[period>Ti])
                ip_low=[i for (i, val) in enumerate(period) if val==T_low]
                ip_high=[i for (i, val) in enumerate(period) if val==T_high]
                
                [Sa_low,sigma_low]=BSSA_2014_sub(M, ip_low, Rjb, U, SS, NS, RS, region, z1, Vs30)
                [Sa_high,sigma_high]=BSSA_2014_sub(M, ip_high, Rjb, U, SS, NS, RS, region, z1, Vs30)
                
                x=np.array([math.log(T_low),math.log(T_high)])
                Y_sa=np.array([math.log(Sa_low),math.log(Sa_high)])
                Y_sigma=np.array([sigma_low,sigma_high])
                median[i]=math.exp(np.interp(math.log(Ti),x,Y_sa))
                sigma[i]=np.interp(math.log(Ti),x,Y_sigma)
                
            else:
                ip_T=idx
                [median[i],sigma[i]]=BSSA_2014_sub(M, ip_T, Rjb, U, SS, NS, RS, region, z1, Vs30)  
        
    return median, sigma
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30):
    
    # constants
    mref=4.5
    rref=1
    v_ref=760 
    f1=0
    f3=0.1
    v1=225
    v2=300
    period=np.array([-1,0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5,7.5,10])
    mh=np.array([6.2,5.5,5.5,5.5,5.5,5.5,5.5,5.54,5.74,5.92,6.05,6.14,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2])
    e0=np.array([5.037,0.4473,0.4534,0.48598,0.56916,0.75436,0.96447,1.1268,1.3095,1.3255,1.2766,1.2217,1.1046,0.96991,0.66903,0.3932,-0.14954,-0.58669,-1.1898,-1.6388,-1.966,-2.5865,-3.0702])
    e1=np.array([5.078,0.4856,0.4916,0.52359,0.6092,0.79905,1.0077,1.1669,1.3481,1.359,1.3017,1.2401,1.1214,0.99106,0.69737,0.4218,-0.11866,-0.55003,-1.142,-1.5748,-1.8882,-2.4874,-2.9537])
    e2=np.array([4.849,0.2459,0.2519,0.29707,0.40391,0.60652,0.77678,0.8871,1.0648,1.122,1.0828,1.0246,0.89765,0.7615,0.47523,0.207,-0.3138,-0.71466,-1.23,-1.6673,-2.0245,-2.8176,-3.3776])
    e3=np.array([5.033,0.4539,0.4599,0.48875,0.55783,0.72726,0.9563,1.1454,1.3324,1.3414,1.3052,1.2653,1.1552,1.012,0.69173,0.4124,-0.1437,-0.60658,-1.2664,-1.7516,-2.0928,-2.6854,-3.1726])
    e4=np.array([1.073,1.431,1.421,1.4331,1.4261,1.3974,1.4174,1.4293,1.2844,1.1349,1.0166,0.95676,0.96766,1.0384,1.2871,1.5004,1.7622,1.9152,2.1323,2.204,2.2299,2.1187,1.8837])
    e5=np.array([-0.1536,0.05053,0.04932,0.053388,0.061444,0.067357,0.073549,0.055231,-0.042065,-0.11096,-0.16213,-0.1959,-0.22608,-0.23522,-0.21591,-0.18983,-0.1467,-0.11237,-0.04332,-0.014642,-0.014855,-0.081606,-0.15096])
    e6=np.array([0.2252,-0.1662,-0.1659,-0.16561,-0.1669,-0.18082,-0.19665,-0.19838,-0.18234,-0.15852,-0.12784,-0.092855,-0.023189,0.029119,0.10829,0.17895,0.33896,0.44788,0.62694,0.76303,0.87314,1.0121,1.0651])
    c1=np.array([-1.24300,-1.13400,-1.13400,-1.13940,-1.14210,-1.11590,-1.08310,-1.06520,-1.05320,-1.06070,-1.07730,-1.09480,-1.12430,-1.14590,-1.17770,-1.19300,-1.20630,-1.21590,-1.21790,-1.21620,-1.21890,-1.25430,-1.32530])
    c2=np.array([0.14890,0.19170,0.19160,0.18962,0.18842,0.18709,0.18225,0.17203,0.15401,0.14489,0.13925,0.13388,0.12512,0.12015,0.11054,0.10248,0.09645,0.09636,0.09764,0.10218,0.10353,0.12507,0.15183])
    c3=np.array([-0.00344,-0.00809,-0.00809,-0.00807,-0.00834,-0.00982,-0.01058,-0.01020,-0.00898,-0.00772,-0.00652,-0.00548,-0.00405,-0.00322,-0.00193,-0.00121,-0.00037,0.00000,0.00000,-0.00005,0.00000,0.00000,0.00000])
    h=np.array([5.3,4.5,4.5,4.5,4.49,4.2,4.04,4.13,4.39,4.61,4.78,4.93,5.16,5.34,5.6,5.74,6.18,6.54,6.93,7.32,7.78,9.48,9.66])
    
    deltac3_gloCATW=np.array([0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000])
    deltac3_CHTU=np.array([0.004350,0.002860,0.002820,0.002780,0.002760,0.002960,0.002960,0.002880,0.002790,0.002610,0.002440,0.002200,0.002110,0.002350,0.002690,0.002920,0.003040,0.002920,0.002620,0.002610,0.002600,0.002600,0.003030])
    deltac3_ITJA=np.array([-0.000330,-0.002550,-0.002440,-0.002340,-0.002170,-0.001990,-0.002160,-0.002440,-0.002710,-0.002970,-0.003140,-0.003300,-0.003210,-0.002910,-0.002530,-0.002090,-0.001520,-0.001170,-0.001190,-0.001080,-0.000570,0.000380,0.001490])
    
    c=np.array([-0.8400,-0.6000,-0.6037,-0.5739,-0.5341,-0.4580,-0.4441,-0.4872,-0.5796,-0.6876,-0.7718,-0.8417,-0.9109,-0.9693,-1.0154,-1.0500,-1.0454,-1.0392,-1.0112,-0.9694,-0.9195,-0.7766,-0.6558])
    vc=np.array([1300.00,1500.00,1500.20,1500.36,1502.95,1501.42,1494.00,1479.12,1442.85,1392.61,1356.21,1308.47,1252.66,1203.91,1147.59,1109.95,1072.39,1009.49,922.43,844.48,793.13,771.01,775.00])
    f4=np.array([-0.1000,-0.1500,-0.1483,-0.1471,-0.1549,-0.1963,-0.2287,-0.2492,-0.2571,-0.2466,-0.2357,-0.2191,-0.1958,-0.1704,-0.1387,-0.1052,-0.0679,-0.0361,-0.0136,-0.0032,-0.0003,-0.0001,0.0000])
    f5=np.array([-0.00844,-0.00701,-0.00701,-0.00728,-0.00735,-0.00647,-0.00573,-0.00560,-0.00585,-0.00614,-0.00644,-0.00670,-0.00713,-0.00744,-0.00812,-0.00844,-0.00771,-0.00479,-0.00183,-0.00152,-0.00144,-0.00137,-0.00136])
    f6=np.array([-9.900,-9.900,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,0.092,0.367,0.638,0.871,1.135,1.271,1.329,1.329,1.183])
    f7=np.array([-9.900,-9.900,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,0.059,0.208,0.309,0.382,0.516,0.629,0.738,0.809,0.703])
    tau1=np.array([0.4010,0.3980,0.4020,0.4090,0.4450,0.5030,0.4740,0.4150,0.3540,0.3440,0.3500,0.3630,0.3810,0.4100,0.4570,0.4980,0.5250,0.5320,0.5370,0.5430,0.5320,0.5110,0.4870])
    tau2=np.array([0.3460,0.3480,0.3450,0.3460,0.3640,0.4260,0.4660,0.4580,0.3880,0.3090,0.2660,0.2290,0.2100,0.2240,0.2660,0.2980,0.3150,0.3290,0.3440,0.3490,0.3350,0.2700,0.2390])
    phi1=np.array([0.6440,0.6950,0.6980,0.7020,0.7210,0.7530,0.7450,0.7280,0.7200,0.7110,0.6980,0.6750,0.6430,0.6150,0.5810,0.5530,0.5320,0.5260,0.5340,0.5360,0.5280,0.5120,0.5100])
    phi2=np.array([0.5520,0.4950,0.4990,0.5020,0.5140,0.5320,0.5420,0.5410,0.5370,0.5390,0.5470,0.5610,0.5800,0.5990,0.6220,0.6250,0.6190,0.6180,0.6190,0.6160,0.6220,0.6340,0.6040])
    dphiR=np.array([0.082,0.100,0.096,0.092,0.081,0.063,0.064,0.087,0.120,0.136,0.141,0.138,0.122,0.109,0.100,0.098,0.104,0.105,0.088,0.070,0.061,0.058,0.060])
    dphiV=np.array([0.080,0.070,0.070,0.030,0.029,0.030,0.022,0.014,0.015,0.045,0.055,0.050,0.049,0.060,0.070,0.020,0.010,0.008,0.000,0.000,0.000,0.000,0.000])
    R1=np.array([105.000,110.000,111.670,113.100,112.130,97.930,85.990,79.590,81.330,90.910,97.040,103.150,106.020,105.540,108.390,116.390,125.380,130.370,130.360,129.490,130.220,130.720,130.000])
    R2=np.array([272.000,270.000,270.000,270.000,270.000,270.000,270.040,270.090,270.160,270.000,269.450,268.590,266.540,265.000,266.510,270.000,262.410,240.140,195.000,199.450,230.000,250.390,210.000])
    
    # the source event function
    if M<=mh[ip]:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e4[ip] * (M - mh[ip]) + e5[ip] * (M - mh[ip])**2
    else:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e6[ip] * (M - mh[ip])
    
    # the path function
    if region==0 or region==1:
        deltac3=deltac3_gloCATW
    elif region==3:
        deltac3=deltac3_CHTU
    elif region==2 or region==4:
        deltac3=deltac3_ITJA
        
    r=math.sqrt(Rjb**2+(h[ip])**2)
    F_P=(c1[ip]+c2[ip]*(M-mref))*math.log(r/rref)+(c3[ip]+deltac3[ip])*(r-rref)
    
    # find PGAr
    if Vs30!=v_ref or ip!=1:
        [PGA_r,sigma_r]=BSSA_2014_sub(M,1, Rjb, U, SS, NS, RS, region, z1, v_ref)
        
        # the site function    
        # -linear component-    
        if Vs30<=vc[ip]:
            ln_Flin=c[ip]*math.log(float(Vs30/v_ref))
        else:
            ln_Flin=c[ip]*math.log(float(vc[ip]/v_ref))
            
        # -nonlinear componenent-
        f2=f4[ip]*(math.exp(f5[ip]*(min([Vs30,760])-360))-math.exp(f5[ip]*(760-360)))
        
        ln_Fnlin=f1+f2*math.log((PGA_r+f3)/f3)
        
        # effect of basin depth
        
        if z1!=999:
            if region==1:
                mu_z1=math.exp(-7.15/4*math.log((Vs30**4+570.94**4)/(1360**4+570.94**4)))/1000
            elif region==2:
                mu_z1=math.exp(-5.23/2*math.log((Vs30**2+412.39**2)/(1360**2+412.39**2)))/1000
            else:
                mu_z1=math.exp(-7.15/4*math.log((Vs30**4+570.94**4)/(1360**4+570.94**4)))/1000
            dz1=z1-mu_z1
        else:
            dz1=0
            
        if z1!=999:
            if period[ip]<0.65:
                F_dz1=0
            elif period[ip]>=0.65 and abs(dz1)<=f7[ip]/f6[ip]:
                F_dz1=f6[ip]*dz1
            else:
                F_dz1=f7[ip]
        else:
            F_dz1=0
            
        F_S=ln_Flin+ln_Fnlin+F_dz1
        ln_Y=F_E+F_P+F_S
        median=math.exp(ln_Y)
    else:
        ln_Y=F_E+F_P
        median=math.exp(ln_Y)
        
    # aleatory uncertainty function
    if M<=4.5:
        tau=tau1[ip]
        phi_M=phi1[ip]
    elif 4.5<M and M<5.5:
        tau=tau1[ip]+(tau2[ip]-tau1[ip])*(M-4.5)
        phi_M=phi1[ip]+(phi2[ip]-phi1[ip])*(M-4.5)
    elif M>=5.5:
        tau=tau2[ip]
        phi_M=phi2[ip]
        
    if Rjb<R1[ip]:
        phi_MR=phi_M
    elif R1[ip]<Rjb and Rjb<=R2[ip]:
        phi_MR=phi_M+dphiR[ip]*(math.log(Rjb/R1[ip])/math.log(R2[ip]/R1[ip]))
    elif Rjb>R2[ip]:
        phi_MR=phi_M+dphiR[ip]
    
    if Vs30>=v2:
        phi_MRV=phi_MR
    elif v1<=Vs30 and Vs30<=v2:
        phi_MRV=phi_MR - dphiV[ip]*(math.log(v2/Vs30)/math.log(v2/v1))
    elif Vs30<=v1:
        phi_MRV=phi_MR-dphiV[ip]
        
    sigma=math.sqrt(phi_MRV**2+tau**2)
    
    return median,sigma
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_scale_factor(sampleBig,sampleSmall,meanReq,sigma,weights,maxScale):
    
    nGM=(sampleSmall.shape[0])+1
    scales=np.arange(0.1,maxScale,0.1)
    scaleFac=np.zeros([sampleBig.shape[0]])
    devTotal=np.zeros([len(scales)])
    minDev=np.zeros([len(scales)])
    sampleSmallNew =sampleSmall
    for i in range(sampleBig.shape[0]):
        for j in range(len(scales)):
            sampleSmallNew.append(float(sampleBig[i,:])+math.log(scales[j]))
            avg=np.sum(sampleSmallNew)/nGM 
            devMean=avg-meanReq
            devSig=math.sqrt((1/(nGM-1))*np.sum((sampleSmallNew-np.matlib.repmat(avg,nGM,1))**2))-sigma
            devTotal=weights[0]*np.sum(devMean**2)+weights(1)*np.sum(devSig**2)
        
        minDev[i]=min(devTotal)
        minID=devTotal.index(min(devTotal))
        scaleFac[i]=scales[minID]
    
    return scaleFac, minDev
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_ground_motions(selectionParams,simulatedSpectra,IMs):
    
    if selectionParams["cond"]==1:
        scaleFacIndex=selectionParams["indTcond"]
    else:
        scaleFacIndex=np.arange(1,len(selectionParams["TgtPer"]),1)
        
    IMs["recIDs"]=np.zeros([selectionParams["nGM"]])
    sampleSmall=np.empty([0,np.shape(selectionParams["TgtPer"])[0]])
    IMs["scaleFac"]=np.ones([selectionParams["nGM"]])
    
    for i in range(int(selectionParams["nGM"])):
        err=np.zeros(selectionParams["nBig"])
        scaleFac=np.ones(selectionParams["nBig"])
        for j in range(selectionParams["nBig"]):
            if selectionParams["isScaled"]==1:
                scaleFac[j]=np.sum(np.exp(IMs["sampleBig"][j,scaleFacIndex])*simulatedSpectra[i,scaleFacIndex])/np.sum(np.exp(IMs["sampleBig"][j,scaleFacIndex])**2)
            
            err[j] = np.sum((np.log(np.exp(IMs["sampleBig"][j,:])*scaleFac[j]) - np.log(simulatedSpectra[i,:]))**2)
            
        for k in range(0,i-1):
              err[int(IMs["recIDs"][k])]=1000000
              
        err[scaleFac>selectionParams["maxScale"]]=1000000
        
        temp=np.min(err)
        
        if temp>=1000000:
              print("Warning: problem with simulated spectrum. No good matches found")
        
        IMs["recIDs"][i]=np.argmin(err)
        recIDs=np.argmin(err)
        temp=np.zeros([1,np.shape(selectionParams["TgtPer"])[0]])
        for l in range(np.shape(selectionParams["TgtPer"])[0]):
              scale_factor=scaleFac[recIDs]
              sa_val=IMs["sampleBig"][recIDs,l]
              temp[0,l]=np.log(np.exp(sa_val)*scale_factor)
        
        sampleSmall=np.append(sampleSmall,temp,axis=0)
        IMs["scaleFac"][i]=scaleFac[recIDs]
    
    IMs["sampleSmall"]=sampleSmall
    
    return IMs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def simulate_spectra(targetSa,selectionParams,seedValue,nTrials):
    
    if seedValue!=0:
        np.random.seed(seedValue)
    else:
        np.random.seed(1000)
        
    devTotalSim=np.zeros([nTrials])
    spectraSample=np.empty([0,np.shape(targetSa["covReq"])[1]])
    for j in range(nTrials):
        temp=np.exp(np.random.multivariate_normal(targetSa["meanReq"],targetSa["covReq"],int(selectionParams["nGM"])))
        sampleMeanErr=np.mean(np.log(temp))-targetSa["meanReq"]
        sampleStdErr=np.std(np.log(temp))-np.sqrt(np.diag(targetSa["covReq"]))
        sampleSkewnessErr=scipy.stats.skew(np.log(temp),bias=False)
        spectraSample=np.append(spectraSample,temp,axis=0)
        
        devTotalSim[j]=selectionParams["weights"][0]*np.sum(np.square(sampleMeanErr))\
        +selectionParams["weights"][1]*np.sum(np.square(sampleStdErr))\
        +selectionParams["weights"][2]*np.sum(np.square(sampleSkewnessErr))
    
    bestSample=np.argmin(devTotalSim)
    simulatedSpectra=np.zeros([int(selectionParams["nGM"]),np.shape(targetSa["covReq"])[1]])
    for i in range(int(selectionParams["nGM"])):
          for j in range(np.shape(targetSa["covReq"])[1]):
                row_idx=bestSample*selectionParams["nGM"]+i
                col_idx=j
                simulatedSpectra[i][j]=spectraSample[row_idx,col_idx]
    
    return simulatedSpectra
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_spectrum_error(selectionParams,targetSa,sampleSmall):
    
    if selectionParams["optType"]==0:
        sampleMean=np.reshape(np.sum(sampleSmall,axis=0)/(np.shape(sampleSmall)[0]),
                              [1,np.shape(sampleSmall)[1]])
        sampleVar=np.sum((sampleSmall-np.matmul(np.ones([np.shape(sampleSmall)[0],1]),sampleMean))**2,axis=0)/(np.shape(sampleSmall)[1])
        devMean=sampleMean-targetSa["meanReq"]
        devSig=np.sqrt(sampleVar)-targetSa["stdevs"]
        devTotal=selectionParams["weights"][0]*np.sum(devMean**2)+selectionParams["weights"][1]*np.sum(devSig**2)
        
#        if selectionParams["penalty"]!=0:
#            for i in range(len(sampleSmall)):
#                devTotal=devTotal+np.sum(np.abs(np.exp(sampleSmall[:][i])>np.exp(targetSa["meanReq"][i]+3*targetSa["stdevs"][i])))*selectionParams["penalty"]
#        
#    elif selectionParams["optType"]== 1:
#        temp=[min(sampleSmall)]
#        temp.extend(sorted(sampleSmall))
#        sortedlnSa=np.array(temp)
#        norm_cdf=stats.norm.cdf(sortedlnSa,loc=np.matlib.repmat(targetSa["meanReq"],selectionParams["nGM"]+1,1),scale=np.matlib.repmat(targetSa["stdevs"],selectionParams["nGM"]+1,1))
#        emp_cdf=np.linspace(0,1,selectionParams["nGM"]+1)
#        Dn=max(abs(np.matlib.repmat(np.transpose(emp_cdf),len(selectionParams["TgtPer"]))-norm_cdf))
#        devTotal=np.sum(Dn)
        
    return devTotal
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def optimize_ground_motions(selectionParams,targetSa,IMs):
    
      sampleSmall=IMs["sampleSmall"]
    
      if selectionParams["isScaled"]==0:
        scaleFac=np.ones([selectionParams["nBig"],1])
        IMs["scaleFac"]=np.ones([selectionParams["nGM"],1])
      elif selectionParams["isScaled"]==1 and selectionParams["cond"]==1:
        scaleFac=np.zeros([selectionParams["nBig"]])
        for i in range(selectionParams["nBig"]):
              scaleFac[i]=np.exp(selectionParams["lnSa1"])/np.exp(IMs["sampleBig"][i][selectionParams["indTcond"]])
        
        idxAllow = [i for (i, val) in enumerate(scaleFac[:]) if val < selectionParams["maxScale"]]
        
    # initiation of ground motion optimization
      for k in range(selectionParams["nLoop"]):
        for i in range(selectionParams["nGM"]):
            
            sampleSmall=np.delete(sampleSmall,i,axis=0)
            IMs["recIDs"]=np.delete(IMs["recIDs"],i,axis=0)
            
            if selectionParams["isScaled"]==1 and selectionParams["cond"]==0:
                scaleFac=compute_scale_factor(IMs["sampleBig"],sampleSmall,targetSa["meanReq"],targetSa["stdevs"],selectionParams["weights"],selectionParams["maxScale"])
                idxAllow = [idx for (idx, val) in enumerate(scaleFac) if val < selectionParams["maxScale"]]
                
            devTotal=1000000*np.ones([selectionParams["nBig"],1])
            
            for j in range(len(idxAllow)):
                temp0=idxAllow[j]
                temp1=[idx for (idx, val) in enumerate(IMs["recIDs"][:]) if val ==temp0]
                
                if temp1==[]:
                    testSpectra=sampleSmall
                    row_to_append=np.zeros([1,np.shape(IMs["sampleBig"])[1]])
                    for idx in range(np.shape(IMs["sampleBig"])[1]):
                          row_to_append[0,idx]=IMs["sampleBig"][temp0,idx]+np.log(scaleFac[temp0])
                    
                    testSpectra=np.append(testSpectra,row_to_append,axis=0)
                    temp_dev=compute_spectrum_error(selectionParams,targetSa,testSpectra)
                    devTotal[temp0]=temp_dev
                    
            minID=np.argmin(devTotal)
            IMs["scaleFac"][i]=scaleFac[minID]
            IMs["recIDs"]=np.insert(IMs["recIDs"],i,np.array([minID]))
            row_to_add=np.reshape(IMs["sampleBig"][minID,:]+np.log(scaleFac[minID]),[1,np.shape(IMs["sampleBig"])[1]])
            sampleSmall=np.insert(sampleSmall,i,row_to_add,axis=0)
        if within_tolerance(sampleSmall,targetSa,selectionParams)==1:
            print('Max tolerance has been achieved')
            break
      
      IMs["sampleSmall"]=sampleSmall
    
      return IMs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def within_tolerance(selectedSa, targetSa, selectionParams):
    
    temp0=np.zeros([selectionParams["nGM"],1])
    for i in range(selectionParams["nGM"]):
          temp1=np.zeros([np.shape(targetSa["meanReq"])[0]])
          for j in range(np.shape(targetSa["meanReq"])[0]):
                sel_sa=selectedSa[i,j]
                tgt_sa=targetSa["meanReq"][j]
                temp1[j]=abs(np.exp(sel_sa)-np.exp(tgt_sa))/np.exp(tgt_sa)
          temp0[i]=np.max(temp1)
          
    medianErr=np.max(temp0)
    
    stdevs=np.zeros([np.shape(selectedSa)[1],1])
    for i in range(np.shape(selectedSa)[1]):
          stdevs[i]=np.std(selectedSa[:,i])
        
    temp0=[stdevs[idx] for idx in range(len(stdevs)) if idx!=selectionParams["indTcond"]]
    temp1=[targetSa["stdevs"][idx] for idx in range(len(targetSa["stdevs"][:])) if idx!=selectionParams["indTcond"]]
    
    temp0=np.array(temp0)
    temp1=np.array(temp1)
    
    stdErr=np.max(np.abs(temp0-temp1)/temp1)
    
    if medianErr<selectionParams["tol"] and stdErr<selectionParams["tol"]:
        withinTol=1
    else:
        withinTol=0
        
    return withinTol
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def screen_gmrs_folder(selectionParams):
      
      # searches for given folder or searches for the folder 'ground_motion_records' in 
      # working directory
      if "gmrs_folder" in selectionParams:
            gmrs_folder=selectionParams["gmrs_folder"]
      else:
            cwd = os.getcwd()
            gmrs_folder=os.path.join(cwd,'ground_motion_records')
            selectionParams["gmrs_folder"]=gmrs_folder
            
      SaKnown=np.empty([0,len(selectionParams["TgtPer"])]) # array to append spectra
      
      filenames = []
      print('Calculating response spectra for all gmrs')
      nGmrs=0
      for file in os.listdir(gmrs_folder):
           
           if file.endswith(".csv"):
                 nGmrs=nGmrs+1
                 itime,iacc=read_gmr(gmrs_folder,file)
                 filenames.append(file)
                 T=selectionParams["TgtPer"]
                 [sa,sd]=calculate_spectra(itime, iacc, T,0.05)
                 sa=np.reshape(sa,[1,len(selectionParams["TgtPer"])])
                 SaKnown=np.append(SaKnown,sa,axis=0)
      print('Response spectra calculated')
      
      metadata={}
      metadata["getTimeSeries"]=np.array(['The following records','from the folder','have met the selection requirements'])
      metadata["Filename"]=np.reshape(np.array(filenames),[nGmrs,1])
      knownPer=selectionParams["TgtPer"]
      
      idxPer=[idx for (idx, val) in enumerate(selectionParams["TgtPer"][:]) if val<=10]
      temp=[selectionParams["TgtPer"][idx] for (idx, val) in enumerate(selectionParams["TgtPer"][:]) if val<=selectionParams["Tcond"]]
      
      if selectionParams["cond"]==1 and temp!=[]:
             selectionParams["TgtPer"] = np.sort(np.append(selectionParams["TgtPer"],np.array(selectionParams["Tcond"])))
             
      indPer=np.zeros([len(selectionParams["TgtPer"])])      
      for i in range(len(selectionParams["TgtPer"])):
            temp=np.abs(knownPer-selectionParams["TgtPer"][i])
            indPer[i]=np.argmin(temp)
      
      indPer=np.array(np.unique(indPer),dtype=int)
      
      selectionParams["TgtPer"]=knownPer[indPer]
      
      temp=np.abs(selectionParams["TgtPer"]-selectionParams["Tcond"])
      selectionParams["indTcond"]=np.argmin(temp)
      
      alwIndex=np.arange(nGmrs)
      
      metadata["allowedIndex"]=np.arange(nGmrs)
      metadata["recNum"]=np.arange(nGmrs)
      metadata["compNum"]=np.ones([len(alwIndex)])
      selectionParams["TgtPer"]=knownPer[indPer]
      temp=np.zeros([len(alwIndex),len(idxPer)])
      
      for i in range(len(alwIndex)):
            for j in range(len(idxPer)):
                  row_ind=int(alwIndex[i])
                  col_ind=int(idxPer[j])
                  temp[i,j]=SaKnown[row_ind,col_ind]
      
      SaKnown=temp
      selectionParams["nBig"]=len(metadata["allowedIndex"][:])
      
      return SaKnown,selectionParams,indPer,knownPer,metadata
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def screen_database(selectionParams,allowedRecs):
      
      cwd = os.getcwd() #current directory
      databaseFile=os.path.join(cwd,'Databases',selectionParams["databaseFile"])
      
      #load mat file with database details 
      matdata=loadmat(databaseFile)
      metadata={}
      
      getTimeSeriesStr=np.array2string(matdata["getTimeSeries"])
      getTimeSeriesStr=getTimeSeriesStr.split("'")
      
      if len(getTimeSeriesStr)<8:
            matdata["getTimeSeries"]=np.array([getTimeSeriesStr[1],getTimeSeriesStr[5],getTimeSeriesStr[9]])
            metadata["getTimeSeries"]=np.array([getTimeSeriesStr[1],getTimeSeriesStr[5],getTimeSeriesStr[9]])
      else:
            matdata["getTimeSeries"]=np.array([getTimeSeriesStr[1],getTimeSeriesStr[5],getTimeSeriesStr[-1]])
            metadata["getTimeSeries"]=np.array([getTimeSeriesStr[1],getTimeSeriesStr[5],getTimeSeriesStr[-1]])
      
      dirLocationStr=[]
      for i in range(np.shape(matdata["dirLocation"])[0]):
            temp_str=np.array2string(matdata["dirLocation"][i])
            temp_str=temp_str.split("'")
            dirLocationStr.append(temp_str[1])
      matdata["dirLocation"]=np.array(dirLocationStr)
      metadata["dirLocation"]=np.array(dirLocationStr)
      
      magnitude=matdata["magnitude"]
      soil_Vs30=matdata["soil_Vs30"]
      closest_D=matdata["closest_D"]
      
      if np.shape(matdata["Sa_2"])==(0,0): #2nd component does not exist
            Filename1Str=[]
            for i in range(np.shape(matdata["Filename_1"])[0]):
                  temp_str=np.array2string(matdata["Filename_1"][i])
                  temp_str=temp_str.split("'")
                  Filename1Str.append(temp_str[1])
            matdata["Filename_1"]=np.array(Filename1Str)
      else:
            Filename1Str=[]
            Filename2Str=[]
            for i in range(np.shape(matdata["Filename_1"])[0]):
                  temp_str=np.array2string(matdata["Filename_1"][i])
                  temp_str=temp_str.split("'")
                  Filename1Str.append(temp_str[1])
                  
                  temp_str=np.array2string(matdata["Filename_2"][i])
                  temp_str=temp_str.split("'")
                  Filename2Str.append(temp_str[1])
            matdata["Filename_1"]=np.array(Filename1Str)
            matdata["Filename_2"]=np.array(Filename2Str)
            
      if selectionParams["arb"]==1:
            if np.shape(matdata["Sa_2"])==(0,0):
                  metadata["Filename"]=np.array(Filename1Str)
                  metadata["compNum"]=np.ones([len(matdata["magnitude"])])
                  metadata["recNum"]=np.arange(len(matdata["magnitude"]))
                  SaKnown=matdata["Sa_1"]
            else:
                  metadata["Filename"]=np.stack((np.array(Filename1Str),np.array(Filename2Str)),axis=0)
                  metadata["compNum"]=np.stack((np.ones([len(matdata["magnitude"])]),2*np.ones([len(matdata["magnitude"])])),axis=0)
                  metadata["recNum"]=np.stack((np.arange(len(matdata["magnitude"])),np.arange(len(matdata["magnitude"]))),axis=0)
                  SaKnown=np.stack((matdata["Sa_1"],matdata["Sa_2"]),axis=0)
                  soil_Vs30=np.stack((matdata["soil_Vs30"],matdata["soil_Vs30"]),axis=0)
                  magnitude=np.stack((matdata["magnitude"],matdata["magnitude"]),axis=0)
                  closest_D=np.stack((matdata["closest_D"],matdata["closest_D"]),axis=0)
                  metadata["dirLocation"]=np.stack((np.array(dirLocationStr),np.array(dirLocationStr)),axis=0)
      else: # 2 component selection
            metadata["Filename"]=np.stack((np.array(Filename1Str),np.array(Filename2Str)),axis=1)
            metadata["dirLocation"]=np.array(dirLocationStr)
            metadata["recNum"]=np.arange(len(matdata["magnitude"]))
            
            if selectionParams["RotD"]==50 and np.shape(matdata["Sa_RotD50"])!=(0,0):
                  SaKnown=matdata["Sa_RotD50"]
                  
            elif selectionParams["RotD"]==100 and np.shape(matdata["Sa_RotD100"])!=(0,0):#
                  SaKnown=matdata["Sa_RotD100"]
                  
            else:
                  SaKnown=np.sqrt(matdata["Sa_1"]*matdata["Sa_2"])
      
      knownPer=[matdata["Periods"][0,idx] for (idx, val) in enumerate(matdata["Periods"][0,:]) if val<=10]
      knownPer=np.array(knownPer)
      idxPer=[idx for (idx, val) in enumerate(matdata["Periods"][0,:]) if val<=10]
      
      temp=[selectionParams["TgtPer"][idx] for (idx, val) in enumerate(selectionParams["TgtPer"][:]) if val<=selectionParams["Tcond"]]
      if selectionParams["cond"]==1 and temp!=[]:
             selectionParams["TgtPer"] = np.sort(np.append(selectionParams["TgtPer"],np.array(selectionParams["Tcond"])))
             
      indPer=np.zeros([len(selectionParams["TgtPer"])])      
      for i in range(len(selectionParams["TgtPer"])):
            temp=np.abs(knownPer-selectionParams["TgtPer"][i])
            indPer[i]=np.argmin(temp)
            
      indPer=np.array(np.unique(indPer),dtype=int)
      
      selectionParams["TgtPer"]=knownPer[indPer]
      temp=np.abs(selectionParams["TgtPer"]-selectionParams["Tcond"])
      selectionParams["indTcond"]=np.argmin(temp)
      
      recValidSa=np.zeros([np.shape(SaKnown)[0]])
      recValidSoil=np.zeros([np.shape(soil_Vs30)[0]])
      recValidMag=np.zeros([len(magnitude)])
      recValidDist=np.zeros([len(closest_D)])
      for i in range(np.shape(SaKnown)[0]):
            temp=[SaKnown[i,idx] for idx in range(np.shape(SaKnown)[1]) if SaKnown[i,idx]==-999]
            if temp==[]:
                  recValidSa[i]=1

      for i in range(len(soil_Vs30)):
            if soil_Vs30[i]> allowedRecs["Vs30"][0] and soil_Vs30[i]< allowedRecs["Vs30"][1]:
                  recValidSoil[i]=1

      for i in range(len(magnitude)):
            if magnitude[i]> allowedRecs["Mag"][0] and magnitude[i]< allowedRecs["Mag"][1]:
                  recValidMag[i]=1

      for i in range(len(closest_D)):
            if closest_D[i]> allowedRecs["D"][0] and closest_D[i]< allowedRecs["D"][1]:
                  recValidDist[i]=1
                  
      alwIndex=[]
      
      for i in range(len(recValidSa)):
            if recValidSa[i]==1 and recValidSoil[i]==1 and recValidMag[i]==1 and recValidDist[i]==1:
                  alwIndex.append(i)
                  
      metadata["allowedIndex"]=np.array(alwIndex)
      temp=np.zeros([len(alwIndex),len(idxPer)])
      
      for i in range(len(alwIndex)):
            for j in range(len(idxPer)):
                  row_ind=alwIndex[i]
                  col_ind=idxPer[j]
                  temp[i,j]=SaKnown[row_ind,col_ind]
      
      SaKnown=temp
      selectionParams["nBig"]=len(metadata["allowedIndex"][:])

      return SaKnown,selectionParams,indPer,knownPer,metadata
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_output(recIdx, IMs, outputDir, outputFile, metadata):
      
      cwd=os.getcwd()
      outPutDir=os.path.join(cwd,outputDir)
      
      if os.path.exists(outPutDir)==False:
            os.mkdir(outPutDir)
            
      fid=open(os.path.join(outPutDir,outputFile),'w')
      fid.write("%s\t%s\t%s\n" 
                %(np.array2string(metadata["getTimeSeries"][0]),np.array2string(metadata["getTimeSeries"][1]),
                  np.array2string(metadata["getTimeSeries"][2])))
      fid.write("\n")
      
      if np.shape(metadata["Filename"])[1]==1:
            fid.write("%s\t%s\t%s\t%s\t%s\t\n" 
                      % ("Record Number","Record Sequence Number","Scale Factor","Component Number","File Name"))
      else:
            fid.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" 
                      % ("Record Number","Record Sequence Number","Scale Factor","File Name Dir1","File Name Dir2","URL 1","URL 2"))
      
      for i in range(len(recIdx)):
            if np.shape(metadata["Filename"])[1]==1:
                  fid.write("%d\t%d\t%6.2f\t%d\t%s\t\n" 
                            % (i,metadata["recNum"][int(recIdx[i])],IMs["scaleFac"][i],metadata["compNum"][int(recIdx[i])],
                            np.array2string(metadata["Filename"][int(recIdx[i])])))
            else:
                  fid.write("%d\t%d\t%6.2f\t%s\t%s\t\n" 
                            % (i,metadata["recNum"][int(recIdx[i])],
                               IMs["scaleFac"][i],np.array2string(metadata["Filename"][int(recIdx[i]),0]),
                               np.array2string(metadata["Filename"][int(recIdx[i]),1])))
                  
      fid.close()
      
      return True
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_out_gmrs(recIdx, IMs,inDir,outputDir, metadata):
      
      if os.path.exists(outputDir):
            shutil.rmtree(outputDir)
      os.mkdir(outputDir)
      
      for i in range(np.shape(recIdx)[0]):
            file=os.path.join(inDir,np.ndarray.tolist(metadata["Filename"][i])[0])
            time, acc = [], []
            scaleFac=IMs["scaleFac"][i]
            with open(file) as f:
              for line in f.readlines():
                  line = line.split(',')
                  time.append(float(line[0]))
                  acc.append(float(line[1])*float(scaleFac))
                  
            time_exp=np.reshape(np.array(time),[len(time),1])
            accel_exp=np.reshape(np.array(acc),[len(acc),1])
            gmr=np.hstack((time_exp,accel_exp))
            
            outfile=os.path.join(outputDir,np.ndarray.tolist(metadata["Filename"][i])[0])
            np.savetxt(outfile,gmr, delimiter=",")
            
      return True


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_results( selectionParams, targetSa, IMs, simulatedSpectra, SaKnown, knownPer ):
      
      plt.figure()
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]),linewidth=3,color='b')
      plt.hold(True)
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]+1.96*np.sqrt(np.diag(targetSa["covReq"]))),color='r',linewidth=3)
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]-1.96*np.sqrt(np.diag(targetSa["covReq"]))),color='r',linewidth=3)
      for i in range((selectionParams["nGM"])):
            plt.loglog(selectionParams["TgtPer"],simulatedSpectra[i,:],color='k')
      
      plt.axis([np.min(selectionParams["TgtPer"][:]),np.max(selectionParams["TgtPer"][:]),1e-2,5])
      plt.xlabel('Period [s]')
      plt.ylabel('Spectral acceleration [g]')
      plt.title('Response spectra of simulated ground motion spectra')
      plt.legend(['Median','2.5 percentile','97.5 percentile','Simulated ground motion'])
      
      plt.figure()
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]),linewidth=3,color='b')
      plt.hold(True)
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]+1.96*np.sqrt(np.diag(targetSa["covReq"]))),color='r',linewidth=3)
      plt.loglog(selectionParams["TgtPer"],
                 np.exp(targetSa["meanReq"]-1.96*np.sqrt(np.diag(targetSa["covReq"]))),color='r',linewidth=3)
      
      for i in range((selectionParams["nGM"])):
            row_idx=IMs["recIDs"][i]
            scale_fact=IMs["scaleFac"][i]
            plt.loglog(knownPer,SaKnown[int(row_idx),:]*scale_fact,color='k')
      plt.axis([np.min(selectionParams["TgtPer"][:]),np.max(selectionParams["TgtPer"][:]),1e-2,5])
      plt.xlabel('Period [s]')
      plt.ylabel('Spectral acceleration [g]')
      plt.title('Response spectra of selected ground motions')
      plt.legend(['Median','2.5 percentile','97.5 percentile','Selected ground motion'])
      
      return True
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def download_time_series(outputDir,rec,Filename,dirLocation):
      
      dirLocStr=np.array2string(dirLocation[0])
      
      if dirLocStr.find("nga_files")!=-1:
            if np.shape(Filename)[1]==1:
                  nComp=1
            else:
                  nComp=2
      elif dirLocStr.find("bbpvault")!=-1:
            nComp=1
      else:
            print("The selected database cannot be accessed by the software at the moment")
            return False
      for i in rec:
            for j in range(nComp):
                  url=np.array2string(dirLocation[rec[i]])+np.array2string(Filename[rec[i],j])
                  if nComp==2:
                        filename=os.path.join(outputDir,"GM"+str(i)+"_comp"+str(j)+".txt")
                  else:
                        filename=os.path.join(outputDir,"GM"+str(i)+".txt")
                        
                  page = urllib.request.urlopen(url)
                  fid = open(filename, "w")
                  content = page.read()
                  fid.write(content)
                  fid.close()
                  
      return True
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_gmrs(folder):

    #This function reads a set of ground motion records
    #and stores them in a dictionary
    time = []
    acc = []
    dt = []
    no_points = []
    name = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            itime, iacc = read_gmr(folder, f)
            time.append(itime)
            acc.append(iacc)
            dt.append(itime[1] - itime[0])
            no_points.append(len(iacc))
            name.append(f)

    gmrs = {'time': None, 'acc': None, 'dt': None,
        'no_points': None, 'name': None}
    gmrs['time'] = time
    gmrs['acc'] = acc
    gmrs['dt'] = dt
    gmrs['no_points'] = no_points
    gmrs['name'] = name

    return gmrs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_gmr(folder, gmr):

    time, acc = [], []
    with open(os.path.join(folder,gmr)) as f:
        for line in f.readlines():
            line = line.split(',')
            time.append(float(line[0]))
            acc.append(float(line[1])*9.81)

    return time, acc
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def calculate_spectra(time, acc, T, damping):

    u0 = 0
    v0 = 0
    dt = time[1]-time[0]
    no_acc = len(acc)
    no_T = len(T)
    M = 1
    Sd = np.zeros(no_T)
    Sa = np.zeros(no_T)
    u = np.zeros(no_acc)
    a = np.zeros(no_acc)
    v = np.zeros(no_acc)
    at = np.zeros(no_acc)

    for i in range(no_T):
        if T[i] == 0:
            Sd[i] = 0
            Sa[i] = max(abs(np.array(acc)))
        else:
            wn = 2*math.pi/T[i]
            C = damping*2*M*wn
            K = ((2*math.pi)/T[i])**2*M
            u[0] = u0
            v[0] = v0
            a[0] = -acc[0]-C*v[0]-K*u[0]
            at[0] = acc[0]+a[0]
            for j in range(no_acc-1):
                u[j+1] = u[j] + dt*v[j] + dt**2 / 2*a[j]
                a[j+1] = (1/(M+dt*0.5*C)) * (-M*acc[j+1] - K*u[j+1] - C*(v[j]+dt*0.5*a[j]))
                v[j+1] = v[j] + dt*(0.5*a[j] + 0.5*a[j+1])
                at[j+1] = acc[j+1] + a[j+1]

            Sd[i] = max(abs(u))
            Sa[i] = max(abs(at))/9.81

    return Sa, Sd
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
