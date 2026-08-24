# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:13:01 2026

@author: Bethany.Kilpatrick
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from dataclasses import dataclass
import scipy.signal as sig
from datetime import datetime 
import math
from functools import reduce
import re
import pandas as pd







fPath = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\2026_Performance_GiantPressure_Giant\\CSV\\'
fileExt = r".csv"

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]





save_on = 1
 
    
    
    
    
# Counter function for subsetting sections

def buildSection(dat, starts, step):
    """
    Automatically join slices from a DataFrame.
    
    Args:
        dat: pandas DataFrame
        starts: list of starting numbers for each RDO block
        step: how many columns each block spans (default 13)
    
    Returns:
        joined DataFrame of all slices
    """
    # Build column ranges
    column_ranges = [
        (f'elem{s} [psi.]', f'elem{s + step - 1} [psi.]') 
        for s in starts
    ]
    
    # Slice each block
    rdo_list = [dat.loc[:, start_col:end_col] for start_col, end_col in column_ranges]
    
    # Join all blocks
    joined = reduce(lambda left, right: left.join(right, how="outer"), rdo_list)
    
    return joined



# First number of each row
R_Dist_Occ_starts = [1472, 1503, 1534, 1565, 1596, 1627, 1658, 1689, 1720, 1751, 1162] 
# How long is the row
R_Dist_Occ_step= 13  

R_Dor_Occ_starts= [1193,1224,1255,1286,1317,1348,1379,1410,1441]
R_Dor_Occ_step= 13 

R_temporal_starts = [1782,1798,1814,1830,1846,1862,1878,1894,1910]
R_temporal_step = 16  

L_Dist_Occ_starts = [1485,1516,1547,1578,1609,1640,1671,1702,1733,1764,1175]
L_Dist_Occ_step = 18 

L_Dor_Occ_starts = [1206,1237,1268,1299,1330,1361,1392,1423,1454]
L_Dor_Occ_step = 18

L_Temporal_starts = [1926,1942,1958,1974,1990,2006,2022,2038,2054]
L_Temporal_step = 16


all_outcomes = []




## save configuration names from files
for fName in entries:
    try:
        
        #fName = entries[0] 
    
        
        helmet = fName.split(sep='_')[0] 
        config = fName.split(sep='_')[1]
        trial = fName.split(sep='_')[2].split(sep='.')[0]  
        
        dat = pd.read_csv(fPath+fName,sep=',',  header = 'infer')  
        dat[dat < 0.004] = 0 
        
        
        contact_area_cols = [
                'Forehead Contact Area [sq in.]',
                'Side UpRear Contact Area [sq in.]',
                'Side LowRear Contact Area [sq in.]',
                'Side UpFront R Contact Area [sq in.]',
                'Side UpFront L Contact Area [sq in.]', 
                
                                ]
        area_means = dat[contact_area_cols].mean()

        avg_all_ContArea    = float(area_means.sum())
        ContArea_Forehead   = float(area_means['Forehead Contact Area [sq in.]'])
        ContArea_SideUpRear = float(area_means['Side UpRear Contact Area [sq in.]'])
        ContArea_SideLowRear    = float(area_means['Side LowRear Contact Area [sq in.]'])
        ContArea_SideUpFrontR   = float(area_means['Side UpFront R Contact Area [sq in.]'])
        ContArea_SideUpFrontL   = float(area_means['Side UpFront L Contact Area [sq in.]'])
        
        
        
        # Drop all columns containing these measurement suffixes
        suffixes = ['Average Pressure', 'Minimum Pressure', 'Maximum Pressure', 
                    'Total Force', 'Contact Area [sq in.]', 'Contact Area [sq mm]' 'Centroid X', 'Centroid Y', 
                    'Centroid Z', 'Peak Location X', 'Peak Location Y', 'Peak Location Z']
        
        pattern = '|'.join(suffixes)
        dat.drop(columns=dat.filter(regex=pattern).columns, inplace=True) 
        
        
       
        
        allSides = dat.loc[:,'elem866 [psi.]': ] 
        allSidesna = allSides.replace(0, np.nan)
        avg_all_na = float(np.nanmean(allSidesna.values))*6.895
        
        sd_all_na = float(np.nanstd(allSidesna.values))*6.895
        cov_all_na = float(sd_all_na/avg_all_na)
        avg_all = float(np.mean(allSides.values))*6.895
        sd_all = float(np.std(allSides.values, axis=None))*6.895
        MaxPress_all = float(np.max(allSides.values)) *6.895
        cov_all = float(sd_all/avg_all)
        
        
        
        Frontal = dat.loc[:,'elem866 [psi.]':'elem1161 [psi.]']
        Frontal = np.mean(Frontal, axis = 0)
        Frontal_na = Frontal.replace(0, np.nan)
        Frontal_Tot = np.sum(Frontal_na)
        avg_Frontal = float(np.nanmean(Frontal_na.values)) * 6.895
        sd_Frontal = float(np.nanstd(Frontal_na.values)) * 6.895
        frontalMax = float(np.nanmax(Frontal_na.values)) * 6.895
        cov_Frontal = float(sd_Frontal / avg_Frontal)
        ppsFrontal = (Frontal_na.values * 6.895 > 10).sum()
        ppsFrontal15 = (Frontal_na.values * 6.895 > 15).sum()
        ppsFrontal20 = (Frontal_na.values * 6.895 > 20).sum()


        R_Distal_Occipital = buildSection(dat, R_Dist_Occ_starts, R_Dist_Occ_step)
        R_Distal_Occipital = np.mean(R_Distal_Occipital, axis = 0)
        R_Distal_Occipital_na = R_Distal_Occipital.replace(0, np.nan)
        tot_R_Distal_Occipital = np.sum(R_Distal_Occipital_na)
        avg_R_Distal_Occipital = float(np.nanmean(R_Distal_Occipital_na.values)) * 6.895
        sd_R_Distal_Occipital = float(np.nanstd(R_Distal_Occipital_na.values)) * 6.895
        MaxPress_R_Distal_Occipital = float(np.nanmax(R_Distal_Occipital_na.values)) * 6.895
        cov_R_Distal_Occipital = float(sd_R_Distal_Occipital / avg_R_Distal_Occipital) if avg_R_Distal_Occipital != 0 else 0.0
        ppsRDistOcc = (R_Distal_Occipital_na.values * 6.895 > 10).sum()
        ppsRDistOcc15 = (R_Distal_Occipital_na.values * 6.895 > 15).sum()
        ppsRDistOcc20 = (R_Distal_Occipital_na.values * 6.895 > 20).sum()


        R_Dorsal_Occipital = buildSection(dat, R_Dor_Occ_starts, R_Dor_Occ_step)
        R_Dorsal_Occipital = np.mean(R_Dorsal_Occipital, axis = 0)
        R_Dorsal_Occipital_na = R_Dorsal_Occipital.replace(0, np.nan)
        tot_R_Dorsal_Occipital = np.sum(R_Dorsal_Occipital_na)
        avg_R_Dorsal_Occipital = float(np.nanmean(R_Dorsal_Occipital_na.values)) * 6.895
        sd_R_Dorsal_Occipital = float(np.nanstd(R_Dorsal_Occipital_na.values)) * 6.895
        MaxPress_R_Dorsal_Occipital = float(np.nanmax(R_Dorsal_Occipital_na.values)) * 6.895
        cov_R_Dorsal_Occipital = float(sd_R_Dorsal_Occipital / avg_R_Dorsal_Occipital)
        ppsRDorsOcc = (R_Dorsal_Occipital_na.values * 6.895 > 10).sum()
        ppsRDorsOcc15 = (R_Dorsal_Occipital_na.values * 6.895 > 15).sum()
        ppsRDorsOcc20 = (R_Dorsal_Occipital_na.values * 6.895 > 20).sum()


        R_temporal = buildSection(dat, R_temporal_starts, R_temporal_step)
        R_temporal = np.mean(R_temporal, axis = 0)
        R_temporal_na = R_temporal.replace(0, np.nan)
        tot_R_temporal = np.sum(R_temporal_na)
        avg_R_temporal = float(np.nanmean(R_temporal_na.values)) * 6.895
        sd_R_temporal = float(np.nanstd(R_temporal_na.values)) * 6.895
        MaxPress_R_temporal = float(np.nanmax(R_temporal_na.values)) * 6.895
        cov_R_temporal = float(sd_R_temporal / avg_R_temporal)
        ppsRtemp = (R_temporal_na.values * 6.895 > 10).sum()
        ppsRtemp15 = (R_temporal_na.values * 6.895 > 15).sum()
        ppsRtemp20 = (R_temporal_na.values * 6.895 > 20).sum()


        L_Distal_Occipital = buildSection(dat, L_Dist_Occ_starts, L_Dist_Occ_step)
        L_Distal_Occipital = np.mean(L_Distal_Occipital, axis = 0)
        L_Distal_Occipital_na = L_Distal_Occipital.replace(0, np.nan)
        tot_L_Distal_Occipital = np.sum(L_Distal_Occipital_na)
        avg_L_Distal_Occipital = float(np.nanmean(L_Distal_Occipital_na.values)) * 6.895
        sd_L_Distal_Occipital = float(np.nanstd(L_Distal_Occipital_na.values)) * 6.895
        MaxPress_L_Distal_Occipital = float(np.nanmax(L_Distal_Occipital_na.values)) * 6.895
        cov_L_Distal_Occipital = float(sd_L_Distal_Occipital / avg_L_Distal_Occipital) if avg_L_Distal_Occipital != 0 else 0.0
        ppsLDistOcc = (L_Distal_Occipital_na.values * 6.895 > 10).sum()
        ppsLDistOcc15 = (L_Distal_Occipital_na.values * 6.895 > 15).sum()
        ppsLDistOcc20 = (L_Distal_Occipital_na.values * 6.895 > 20).sum()


        L_Dorsal_Occipital = buildSection(dat, L_Dor_Occ_starts, L_Dor_Occ_step)
        L_Dorsal_Occipital = np.mean(L_Dorsal_Occipital, axis = 0)
        L_Dorsal_Occipital_na = L_Dorsal_Occipital.replace(0, np.nan)
        tot_L_Dorsal_Occipital = np.sum(L_Dorsal_Occipital_na)
        avg_L_Dorsal_Occipital = float(np.nanmean(L_Dorsal_Occipital_na.values)) * 6.895
        sd_L_Dorsal_Occipital = float(np.nanstd(L_Dorsal_Occipital_na.values)) * 6.895
        MaxPress_L_Dorsal_Occipital = float(np.nanmax(L_Dorsal_Occipital_na.values)) * 6.895
        cov_L_Dorsal_Occipital = float(sd_L_Dorsal_Occipital / avg_L_Dorsal_Occipital)
        L_Dorsal_Occipital_kpa = L_Dorsal_Occipital_na.values * 6.895
        ppsLDorsOcc = (L_Dorsal_Occipital_na.values * 6.895 > 10).sum()
        ppsLDorsOcc15 = (L_Dorsal_Occipital_na.values * 6.895 > 15).sum()
        ppsLDorsOcc20 = (L_Dorsal_Occipital_na.values * 6.895 > 20).sum()


        L_Temporal = buildSection(dat, L_Temporal_starts, L_Temporal_step)
        L_Temporal = np.mean(L_Temporal, axis = 0)
        L_Temporal_na = L_Temporal.replace(0, np.nan)
        tot_L_Temporal = np.sum(L_Temporal_na)
        avg_L_Temporal = float(np.nanmean(L_Temporal_na.values)) * 6.895
        sd_L_Temporal = float(np.nanstd(L_Temporal_na.values)) * 6.895
        MaxPress_L_Temporal = float(np.nanmax(L_Temporal_na.values)) * 6.895
        cov_L_Temporal = float(sd_L_Temporal / avg_L_Temporal)
        ppsLtemp = (L_Temporal.values*6.895 > 10).sum()
        ppsLtemp15 = (L_Temporal.values*6.895 > 15).sum()
        ppsLtemp20 = (L_Temporal.values*6.895 > 20).sum()
        
        
         
        overallMax = np.max([frontalMax, MaxPress_R_Distal_Occipital, MaxPress_R_Dorsal_Occipital, MaxPress_R_temporal, MaxPress_L_Distal_Occipital, MaxPress_L_Dorsal_Occipital, MaxPress_L_Temporal]  )
        overallCA = ContArea_Forehead + ContArea_SideUpRear + ContArea_SideLowRear + ContArea_SideUpFrontR + ContArea_SideUpFrontL
        overallTotal = Frontal_Tot + tot_L_Distal_Occipital + tot_L_Dorsal_Occipital + tot_L_Temporal + tot_R_Distal_Occipital + tot_R_Dorsal_Occipital + tot_R_temporal
        overallAvg = overallTotal/overallCA
        ppsTotal = ppsFrontal + ppsLDistOcc + ppsLDorsOcc + ppsLtemp + ppsRDistOcc + ppsRDorsOcc + ppsRtemp
       
        ppsTotal15 = ppsFrontal15 + ppsLDistOcc15 + ppsLDorsOcc15 + ppsLtemp15 + ppsRDistOcc15 + ppsRDorsOcc15 + ppsRtemp15
        
        ppsTotal20 = ppsFrontal20 + ppsLDistOcc20 + ppsLDorsOcc20 + ppsLtemp20 + ppsRDistOcc20 + ppsRDorsOcc20 + ppsRtemp20
       
        
        

        
        
        
        

        all_outcomes.append([
    helmet, config, trial,
    overallCA, ContArea_Forehead, ContArea_SideUpRear, ContArea_SideLowRear,
    ContArea_SideUpFrontR,
    ContArea_SideUpFrontL,
    overallAvg, overallMax, 
    
    avg_Frontal, frontalMax, cov_Frontal,
    avg_R_Distal_Occipital, MaxPress_R_Distal_Occipital, cov_R_Distal_Occipital,
    avg_R_Dorsal_Occipital, MaxPress_R_Dorsal_Occipital, cov_R_Dorsal_Occipital,
    avg_R_temporal, MaxPress_R_temporal, cov_R_temporal,
    avg_L_Distal_Occipital, MaxPress_L_Distal_Occipital, cov_L_Distal_Occipital,
    avg_L_Dorsal_Occipital, MaxPress_L_Dorsal_Occipital, cov_L_Dorsal_Occipital,
    avg_L_Temporal, MaxPress_L_Temporal, cov_L_Temporal,
    ppsTotal, 
    ppsTotal15, 
    ppsTotal20, 
])

        
                       
    except: 
        print (fName) 
        
        
outcomes = pd.DataFrame(all_outcomes, columns=[
    "Helmet", "Config", "Order",
    'avg_all_ContArea', 'ContArea_Forehead', 'ContArea_SideUpRear', 'ContArea_SideLowRear',
    'ContArea_SideUpFrontR',
    'ContArea_SideUpFrontL',
    'avg_all', 'MaxPress_all', 
    
    "avg_Frontal", "frontalMax", "cov_Frontal",
    "avg_R_Distal_Occipital", "MaxPress_R_Distal_Occipital", "cov_R_Distal_Occipital",
    "avg_R_Dorsal_Occipital", "MaxPress_R_Dorsal_Occipital", "cov_R_Dorsal_Occipital",
    "avg_R_temporal", "MaxPress_R_temporal", "cov_R_temporal",
    "avg_L_Distal_Occipital", "MaxPress_L_Distal_Occipital", "cov_L_Distal_Occipital",
    "avg_L_Dorsal_Occipital", "MaxPress_L_Dorsal_Occipital", "cov_L_Dorsal_Occipital",
    "avg_L_Temporal", "MaxPress_L_Temporal", "cov_L_Temporal",
    "ppsTotal", 
    "ppsTotal15", 
    "ppsTotal20"
])
        


if save_on == 1:
    outfileName = fPath + '0_CompiledHelmetData.csv'
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, header=True, index = False)
 
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False)
    
    




        
