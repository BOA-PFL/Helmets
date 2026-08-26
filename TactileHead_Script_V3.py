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
from scipy import stats






fPath = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\2026_Performance_GiantPressure_Giant\\CSV\\'
fileExt = r".csv"

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

filePath2D = r"C:\Users\minori.iizuka\OneDrive - BOA Technology Inc\PFL Team - General\Helmets\TactileHead\3D_Model\sensel_display_positions.csv"
dat2D = pd.read_csv(filePath2D)
dat2D = dat2D.set_index('sensel_id', drop=False)




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

def extract_sensel_num(col_name):
    """Extract the integer sensel number from a column name like 'elem123 [psi.]'."""
    match = re.search(r'elem(\d+)', col_name)
    return int(match.group(1)) if match else np.nan



def pressureProminence(pressureArray):

    prominence = []

    for fr in range(len(pressureArray)):

        pressureArrayfr = pressureArray[fr]

        # Entire frame is NaN -> nothing to measure
        if np.all(np.isnan(pressureArrayfr)):
            prominence.append(np.nan)
            continue

        # NaN-safe top-3: push NaNs to -inf so they never outrank real values
        safeForSort = np.where(np.isnan(pressureArrayfr), -np.inf, pressureArrayfr)
        flatidx = np.argsort(safeForSort, axis=None)[-3:][::-1]
        top3press = pressureArrayfr.flatten()[flatidx]
        top3idx = [np.unravel_index(idx, pressureArrayfr.shape) for idx in flatidx]

        promtemp = []
        for i, maxpressidx in enumerate(top3idx):

            peakval = top3press[i]
            if np.isnan(peakval):
                # fewer than 3 valid (non-NaN) cells in this frame
                continue

            r0_1 = max(maxpressidx[0]-1, 0)
            r1_1 = maxpressidx[0]+2
            c0_1 = max(maxpressidx[1]-1, 0)
            c1_1 = maxpressidx[1]+2
            window = pressureArrayfr[r0_1:r1_1, c0_1:c1_1]

            r0_2 = max(maxpressidx[0]-2, 0)
            r1_2 = maxpressidx[0]+3
            c0_2 = max(maxpressidx[1]-2, 0)
            c1_2 = maxpressidx[1]+3
            window2 = pressureArrayfr[r0_2:r1_2, c0_2:c1_2]

            ring1count = np.sum(~np.isnan(window)) - 1                       # exclude center
            ring2count = np.sum(~np.isnan(window2)) - np.sum(~np.isnan(window))

            ring1sum = np.nansum(window) - peakval
            ring2sum = np.nansum(window2) - np.nansum(window)

            ring1avg = ring1sum / ring1count if ring1count > 0 else np.nan
            ring2avg = ring2sum / ring2count if ring2count > 0 else np.nan

            surroundgrad = ring1avg + ring2avg

            promtemp.append(peakval - surroundgrad)

        if promtemp and not np.all(np.isnan(promtemp)):
            prominence.append(np.nanmax(promtemp))
        else:
            prominence.append(np.nan)

    return np.nanmax(prominence)
     
def build_pressure_matrix(pressure_csv_path, dat2D):
    """
    Build a (frames, rows, cols) matrix of pressure values from one
    TactileHead pressure CSV, mapped onto the 2D sensel grid defined
    by dat2D (sensel_display_positions.csv).

    Each row of the pressure CSV is one frame; each 'elem{sensel_id} [psi.]'
    column holds that sensel's pressure. dat2D['x']/['y'] give the integer
    grid column/row for each sensel_id.

    Returns
    -------
    np.ndarray of shape (n_frames, n_rows, n_cols)
        Grid cells with no sensel mapped to them are NaN.
    """
    df = pd.read_csv(pressure_csv_path)

    n_cols = int(dat2D['x'].max()) + 1
    n_rows = int(dat2D['y'].max()) + 1
    n_frames = len(df)

    elem_cols = [f"elem{sid} [psi.]" for sid in dat2D['sensel_id']]
    missing = [c for c in elem_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} sensel columns from dat2D not found in "
            f"{os.path.basename(pressure_csv_path)}, e.g. {missing[:5]}"
        )

    pressure_values = df[elem_cols].to_numpy()  # (n_frames, n_sensels)
    rows = dat2D['y'].to_numpy()
    cols = dat2D['x'].to_numpy()

    matrix = np.full((n_frames, n_rows, n_cols), np.nan)
    matrix[:, rows, cols] = pressure_values

    return matrix

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
        
        # Build 2D map with pressure values for Prominence calculations
        pressureMap2D = build_pressure_matrix(os.path.join(fPath,fName), dat2D)
        # Separate into different regions (e.g., forehead,temples,occipital,crown)
        forehead2D = pressureMap2D[:,13:25,31:61]
        LTemple2D = pressureMap2D[:,11:20,63:79]
        LDorsalOcc2D = pressureMap2D[:,11:20,79:97]
        LDistalOcc2D = pressureMap2D[:,0:11,79:97]
        RTemple2D = pressureMap2D[:,11:20,13:29]
        RDorsalOcc2D = pressureMap2D[:,11:20,0:13]
        RDistalOcc2D = pressureMap2D[:,0:11,0:13]
        crown2D = pressureMap2D[:,26:59,26:66]
        # Combined Occipital (e.g., dorsal,distal,entire occipital)
        dorsalOccipital2D =  np.concat((LDorsalOcc2D,RDorsalOcc2D),axis=2)
        distalOccipital2D =  np.concat((LDistalOcc2D,RDistalOcc2D),axis=2)
        Occipital2D = np.concat((distalOccipital2D,dorsalOccipital2D),axis=1)
        
        
        colMean = dat.mean(axis=0, skipna=True)
        elem_num = colMean.index.str.extract(r'elem(\d+)', expand=False)
        valid = elem_num.notna()
        matches = pd.Series(False, index=colMean.index)
        matches[valid] = elem_num[valid].astype(int).isin(dat2D['sensel_id'])
        filtered_series = colMean[matches]
        # Center of Pressure
        x_mean = sum(filtered_series.values * dat2D.x.values) / sum(filtered_series.values)
        y_mean = sum(filtered_series.values * dat2D.y.values) / sum(filtered_series.values)
        
        contact_area_cols = [
                'Crown Contact Area [sq in.]',
                'Forehead Contact Area [sq in.]',
                'Side UpRear Contact Area [sq in.]',
                'Side LowRear Contact Area [sq in.]',
                'Side UpFront R Contact Area [sq in.]',
                'Side UpFront L Contact Area [sq in.]', 
                
                                ]
        area_means = dat[contact_area_cols].mean()

        avg_all_ContArea    = float(area_means.sum())
        ContArea_Crown      = float(area_means['Crown Contact Area [sq in.]'])
        ContArea_Forehead   = float(area_means['Forehead Contact Area [sq in.]'])
        ContArea_SideUpRear = float(area_means['Side UpRear Contact Area [sq in.]'])
        ContArea_SideLowRear    = float(area_means['Side LowRear Contact Area [sq in.]'])
        ContArea_SideUpFrontR   = float(area_means['Side UpFront R Contact Area [sq in.]'])
        ContArea_SideUpFrontL   = float(area_means['Side UpFront L Contact Area [sq in.]'])
        ContArea_Total = (ContArea_Crown + ContArea_Forehead + ContArea_SideUpRear 
                          + ContArea_SideLowRear + ContArea_SideUpFrontL 
                          + ContArea_SideUpFrontR)
        
        
        # Drop all columns containing these measurement suffixes
        suffixes = ['Average Pressure', 'Minimum Pressure', 'Maximum Pressure', 
                    'Total Force', 'Contact Area [sq in.]', 'Contact Area [sq mm]' 'Centroid X', 'Centroid Y', 
                    'Centroid Z', 'Peak Location X', 'Peak Location Y', 'Peak Location Z']
        
        pattern = '|'.join(suffixes)
        dat.drop(columns=dat.filter(regex=pattern).columns, inplace=True) 
        
        
       
        
        allSides = dat.loc[:, 'elem0 [psi.]':]
        allSidesna = allSides.replace(0, np.nan)
        avg_all_na = float(np.nanmean(allSidesna.values))*6.895
        
        sd_all_na = float(np.nanstd(allSidesna.values))*6.895
        cov_all_na = float(sd_all_na/avg_all_na)
        avg_all = float(np.mean(allSides.values))*6.895
        sd_all = float(np.std(allSides.values, axis=None))*6.895
        MaxPress_all = float(np.max(allSides.values)) *6.895
        cov_all = float(sd_all/avg_all)
        
        
        # --- Crown: contiguous elem0-865 ---
        crown = dat.loc[:, 'elem0 [psi.]':'elem865 [psi.]']
        crown_na = crown.replace(0, np.nan)
        crown = np.mean(crown, axis=0)
    
        Crown_Tot = np.sum(np.mean(crown_na, axis=0))
        avg_Crown = float(np.nanmean(crown_na.values)) * 6.895
        crownMax = float(np.nanmax(crown_na)) * 6.895
        crown_max_sensel_column = np.nanargmax(np.mean(crown_na, axis=0), axis=0)
        crown_max_sensel_col_name = crown_na.columns[crown_max_sensel_column]
        
        # --- Forehead: contiguous elem866-1161 ---
        Frontal = dat.loc[:,'elem866 [psi.]':'elem1161 [psi.]']
        Frontal = np.mean(Frontal, axis = 0)
        Frontal_na = Frontal.replace(0, np.nan)
        Frontal_Tot = np.sum(Frontal_na)
        avg_Frontal = float(np.nanmean(Frontal_na.values)) * 6.895
        sd_Frontal = float(np.nanstd(Frontal_na.values)) * 6.895
        frontalMax = float(np.nanmax(Frontal_na.values)) * 6.895
        cov_Frontal = float(sd_Frontal / avg_Frontal)
        forehead_max_sensel_column = np.nanargmax(np.mean(Frontal_na, axis=0), axis=0)
        forehead_max_sensel_col_name = Frontal_na.columns[forehead_max_sensel_column]
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
        R_Distal_Occipital_max_sensel_column = np.nanargmax(np.mean(R_Distal_Occipital_na, axis=0), axis=0)
        R_Distal_Occipital_max_sensel_col_name = R_Distal_Occipital_na.columns[R_Distal_Occipital_max_sensel_column]
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
        R_Dorsal_Occipital_max_sensel_column = np.nanargmax(np.mean(R_Dorsal_Occipital_na, axis=0), axis=0)
        R_Dorsal_Occipital_max_sensel_col_name = R_Dorsal_Occipital_na.columns[R_Dorsal_Occipital_max_sensel_column]
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
        R_Temporal_max_sensel_column = np.nanargmax(np.mean(R_temporal_na, axis=0), axis=0)
        R_Temporal_max_sensel_col_name = R_temporal_na.columns[R_Temporal_max_sensel_column]
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
        L_Distal_Occipital_max_sensel_column = np.nanargmax(np.mean(L_Distal_Occipital_na, axis=0), axis=0)
        L_Distal_Occipital_max_sensel_col_name = L_Distal_Occipital_na.columns[L_Distal_Occipital_max_sensel_column]
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
        L_Dorsal_Occipital_max_sensel_column = np.nanargmax(np.mean(L_Dorsal_Occipital_na, axis=0), axis=0)
        L_Dorsal_Occipital_max_sensel_col_name = L_Dorsal_Occipital_na.columns[L_Dorsal_Occipital_max_sensel_column]
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
        L_Temporal_max_sensel_col = np.nanargmax(np.mean(L_Temporal_na, axis=0))
        L_Temporal_max_sensel_col_name = L_Temporal_na.columns[L_Temporal_max_sensel_col]
        ppsLtemp = (L_Temporal.values*6.895 > 10).sum()
        ppsLtemp15 = (L_Temporal.values*6.895 > 15).sum()
        ppsLtemp20 = (L_Temporal.values*6.895 > 20).sum()
        
        
        # (kept for parity with the loop; not used downstream there either)
        leftSideNA = pd.concat((L_Distal_Occipital_na, L_Dorsal_Occipital_na), axis=1)
        rightSideNA = pd.concat((R_Distal_Occipital_na, R_Dorsal_Occipital_na), axis=1)
        # --- Left Side: All ---
        # leftSide_vals = pd.concat([l_dist,l_temp,l_dor],axis=1)
        left_max = float(np.nanmax(leftSideNA)) *  6.895 
        left_max_sensel_column = np.nanargmax(np.mean(leftSideNA,axis=0),axis=0)
        left_max_sensel = leftSideNA.columns[left_max_sensel_column]
    
        # --- Right Side: All ---
        # rightSide_vals = pd.concat([r_dist,r_temp,r_dor],axis=1)
        right_max = float(np.nanmax(rightSideNA)) *  6.895 
        right_max_sensel_column = np.nanargmax(np.mean(rightSideNA,axis=0),axis=0)
        right_max_sensel = rightSideNA.columns[right_max_sensel_column]
        
        
        overallMax = np.max([frontalMax, MaxPress_R_Distal_Occipital, MaxPress_R_Dorsal_Occipital, MaxPress_R_temporal, MaxPress_L_Distal_Occipital, MaxPress_L_Dorsal_Occipital, MaxPress_L_Temporal]  )
        overallCA = ContArea_Forehead + ContArea_SideUpRear + ContArea_SideLowRear + ContArea_SideUpFrontR + ContArea_SideUpFrontL
        overallTotal = Frontal_Tot + tot_L_Distal_Occipital + tot_L_Dorsal_Occipital + tot_L_Temporal + tot_R_Distal_Occipital + tot_R_Dorsal_Occipital + tot_R_temporal
        overallAvg = overallTotal/overallCA
        ppsTotal = ppsFrontal + ppsLDistOcc + ppsLDorsOcc + ppsLtemp + ppsRDistOcc + ppsRDorsOcc + ppsRtemp
       
        ppsTotal15 = ppsFrontal15 + ppsLDistOcc15 + ppsLDorsOcc15 + ppsLtemp15 + ppsRDistOcc15 + ppsRDorsOcc15 + ppsRtemp15
        
        ppsTotal20 = ppsFrontal20 + ppsLDistOcc20 + ppsLDorsOcc20 + ppsLtemp20 + ppsRDistOcc20 + ppsRDorsOcc20 + ppsRtemp20
       
        overallSenselTotal = np.sum(allSides, axis=1)
        amaxy, ind_amaxy = np.max(allSides, axis=1) * 6.895, stats.mode(np.argmax(allSides, axis=1))[0]
    
        crown_max_sensel = extract_sensel_num(crown_max_sensel_col_name)
        forehead_max_sensel = extract_sensel_num(forehead_max_sensel_col_name)
        R_Distal_Occipital_max_sensel = extract_sensel_num(R_Distal_Occipital_max_sensel_col_name)
        R_Dorsal_Occipital_max_sensel = extract_sensel_num(R_Dorsal_Occipital_max_sensel_col_name)
        R_Temporal_max_sensel = extract_sensel_num(R_Temporal_max_sensel_col_name)
        L_Distal_Occipital_max_sensel = extract_sensel_num(L_Distal_Occipital_max_sensel_col_name)
        L_Dorsal_Occipital_max_sensel = extract_sensel_num(L_Dorsal_Occipital_max_sensel_col_name)
        L_Temporal_max_sensel = extract_sensel_num(L_Temporal_max_sensel_col_name)
        
        # Prominence Calculations
        prominenceForehead = pressureProminence(forehead2D)
        prominenceLTemple = pressureProminence(LTemple2D)
        prominenceLDistalOcc = pressureProminence(LDistalOcc2D)
        prominenceLDorsalOcc = pressureProminence(LDorsalOcc2D)
        prominenceRTemple = pressureProminence(RTemple2D)
        prominenceRDistalOcc = pressureProminence(RDistalOcc2D)
        prominenceRDorsalOcc = pressureProminence(RDorsalOcc2D)
        prominencedorsalOccipital =  pressureProminence(dorsalOccipital2D)
        prominencedistalOccipital =  pressureProminence(distalOccipital2D)
        prominenceOccipital = pressureProminence(Occipital2D)
        

        
        
        
        

        all_outcomes.append([
    helmet, config, trial,
    overallCA, ContArea_Crown, ContArea_Forehead, ContArea_SideUpRear, ContArea_SideLowRear,
    ContArea_SideUpFrontR,
    ContArea_SideUpFrontL, ContArea_Total,
    overallAvg, overallMax,
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
    x_mean, y_mean,
    ind_amaxy,
    crown_max_sensel,
    forehead_max_sensel,
    R_Distal_Occipital_max_sensel,
    R_Dorsal_Occipital_max_sensel,
    R_Temporal_max_sensel,
    L_Distal_Occipital_max_sensel,
    L_Dorsal_Occipital_max_sensel,
    L_Temporal_max_sensel,
    prominenceForehead,
    prominenceLTemple,
    prominenceLDistalOcc,
    prominenceLDorsalOcc,
    prominenceRTemple,
    prominenceRDistalOcc,
    prominenceRDorsalOcc,
    prominencedorsalOccipital,
    prominencedistalOccipital,
    prominenceOccipital
])

        
                       
    except: 
        print (fName) 
        
        
outcomes = pd.DataFrame(all_outcomes, columns=[
    "Helmet", "Config", "Order",
    'avg_all_ContArea', 'ContArea_Crown','ContArea_Forehead', 'ContArea_SideUpRear', 'ContArea_SideLowRear',
    'ContArea_SideUpFrontR',
    'ContArea_SideUpFrontL','ContArea_Total',
    'overallAvg', 'overallMax',
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
    "ppsTotal20",
    "centroid_x","centroid_y",
    "sensel_#_peak",
    'Crown Max Sensel #',
    'Forehead Max Sensel #',
    'R_Distal_Occipital_max_sensel',
    'R_Dorsal_Occipital_max_sensel',
    'R_Temporal_max_sensel',
    'L_Distal_Occipital_max_sensel',
    'L_Dorsal_Occipital_max_sensel',
    'L_Temporal_max_sensel',
    'prominenceForehead',
    'prominenceLTemple',
    'prominenceLDistalOcc',
    'prominenceLDorsalOcc',
    'prominenceRTemple',
    'prominenceRDistalOcc',
    'prominenceRDorsalOcc',
    'prominencedorsalOccipital',
    'prominencedistalOccipital',
    'prominenceOccipital'
])
        


if save_on == 1:
    outfileName = fPath + '0_CompiledHelmetData.csv'
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, header=True, index = False)
 
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False)
    
    




        
