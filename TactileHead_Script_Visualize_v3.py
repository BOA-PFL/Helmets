# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:42:47 2026

@author: Bethany.Kilpatrick
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:00:58 2026

@author: Bethany.Kilpatrick

v3 - Updated to use All_Sensels_Combined.csv (global coordinates) to render
     all sections in a single unified head heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from functools import reduce
from dataclasses import dataclass, field


# ── Paths ─────────────────────────────────────────────────────────────────────
lookUpTables = 'C:\\Users\\bethany.kilpatrick\\BOA Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\TactileHead_LookupTables\\'

fPath    = 'C:\\Users\\bethany.kilpatrick\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Helmets\\2026_Performance_GiantPressure_Giant\\CSV\\'
fileExt  = r".csv"

entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and '0_' not in fName]

save_on    = 0
data_check = 0


# ── Section definitions (used only for slicing raw data columns) ──────────────
SECTIONS = {
    'R_Occipital':  {'starts': [1472,1503,1534,1565,1596,1627,1658,1689,1720,1751,1162], 'step': 13, 'lut': 'Right_DistalOccipital.csv'},
    'R_Parietal':   {'starts': [1193,1224,1255,1286,1317,1348,1379,1410,1441],           'step': 13, 'lut': 'Right_DorsalOccipital.csv'},
    'R_Temporal': {'starts': [1782,1798,1814,1830,1846,1862,1878,1894,1910],           'step': 16, 'lut': 'Right_Temporal.csv'},
    'L_Occipital':  {'starts': [1485,1516,1547,1578,1609,1640,1671,1702,1733,1764,1175], 'step': 18, 'lut': 'Left_DistalOccipital.csv'},
    'L_Parietal':   {'starts': [1206,1237,1268,1299,1330,1361,1392,1423,1454],           'step': 18, 'lut': 'Left_DorsalOccipital.csv'},
    'L_Temporal': {'starts': [1926,1942,1958,1974,1990,2006,2022,2038,2054],           'step': 16, 'lut': 'Left_Temporal.csv'},
    'Frontal':    {'start': 866, 'end': 1161,                                          'lut': 'Frontal.csv'},
}


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class headDat:
    fullDat:    pd.DataFrame
    sectionDat: dict = field(default_factory=dict)
    unifiedMat: np.ndarray = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_header(c):
    """Strip whitespace and other unwanted characters from column names."""
    return c.strip()

def buildSection(dat, cfg):
    """
    Subset sensor columns from the raw export.
    Handles both multi-block sections (starts + step) and
    single continuous blocks (start + end).
    """
    if 'start' in cfg:
        return dat.loc[:, f'elem{cfg["start"]} [psi.]' : f'elem{cfg["end"]} [psi.]']
    else:
        column_ranges = [
            (f'elem{s} [psi.]', f'elem{s + cfg["step"] - 1} [psi.]')
            for s in cfg['starts']
        ]
        slices = [dat.loc[:, start:end] for start, end in column_ranges]
        return reduce(lambda l, r: l.join(r, how='outer'), slices)

def buildUnifiedMatrix(sectionDat, combinedLutPath):
    """
    Map ALL sections onto a single 2D spatial matrix using the global
    X/Y coordinates from All_Sensels_Combined.csv.

    The LUT has one row per sensel with columns: sensel_id, x, y.
    sensel_id is used to match the raw data column 'elem{id} [psi.]'.
    """
    lut = pd.read_csv(combinedLutPath, sep=',', header=0)

    x_coords  = lut['x'].values.astype(int)
    y_coords  = lut['y'].values.astype(int)
    sensel_ids = lut['sensel_id'].values.astype(int)

    # Determine the number of timepoints from the first available section
    n_frames = next(iter(sectionDat.values())).shape[0]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Shift coords so matrix indices are always >= 0
    x_shift = -x_min if x_min < 0 else 0
    y_shift = -y_min if y_min < 0 else 0

    mat = np.full(
        (n_frames, x_max - x_min + 1, y_max - y_min + 1),
        np.nan
    )

    for ii, sid in enumerate(sensel_ids):
        col_name = f'elem{sid} [psi.]'
        # Search across all sections for this sensel's column
        for sec_df in sectionDat.values():
            if col_name in sec_df.columns:
                values = sec_df[col_name].values
                xi = x_coords[ii] + x_shift
                yi = y_coords[ii] + y_shift
                mat[:, xi, yi] = values
                break

    # Treat very low readings as no-contact
    mat[mat < 0.1] = 0

    return mat


# ── Main constructor ──────────────────────────────────────────────────────────
def createHead(inputName):
    """
    Load a raw export file, slice into anatomical sections,
    and build a single unified 2D matrix across all sections.
    """
    raw = pd.read_csv(fPath + inputName, sep=',', header='infer')
    raw.rename(columns=lambda c: clean_header(c), inplace=True)

    # Drop summary columns, keeping only raw sensel pressure data
    suffixes = ['Average Pressure', 'Minimum Pressure', 'Maximum Pressure',
                'Total Force', 'Contact Area [sq in.]', 'Contact Area [sq mm]',
                'Centroid X', 'Centroid Y', 'Centroid Z',
                'Peak Location X', 'Peak Location Y', 'Peak Location Z']
    pattern = '|'.join(suffixes)
    raw.drop(columns=raw.filter(regex=pattern).columns, inplace=True)

    sectionDat = {}
    for name, cfg in SECTIONS.items():
        sectionDat[name] = buildSection(raw, cfg)

    combinedLut = os.path.join(lookUpTables, 'All_Sensels_Combined_1.csv')
    if os.path.exists(combinedLut):
        unifiedMat = buildUnifiedMatrix(sectionDat, combinedLut)
    else:
        print(f"Warning: combined lookup table not found at {combinedLut}")
        unifiedMat = None

    return headDat(fullDat=raw, sectionDat=sectionDat, unifiedMat=unifiedMat)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plotUnifiedHead(result, title=''):
    """
    Plot a single unified heatmap of the entire head, averaged across
    all timepoints, using global sensel coordinates.
    NaN cells (no sensel) are shown as white.
    """
    mat   = result.unifiedMat
    frame = np.nanmean(mat, axis=0)

    # Mask cells that were never populated (remain NaN after mean)
    masked = np.ma.masked_invalid(frame)

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color='white')   # empty (no sensel) cells appear white

    im = ax.imshow(
        masked.T,                 # transpose so X=horizontal, Y=vertical
        cmap=cmap,
        origin='lower',
        aspect='equal',
        vmin=0,
        vmax=np.nanmax(frame)
    )

    fig.colorbar(im, ax=ax, label='Pressure (psi)')
    ax.set_title(f'Unified Head Map - Mean Pressure{" - " + title if title else ""}')
    ax.set_xlabel('X Position (global)')
    ax.set_ylabel('Y Position (global)')
    plt.tight_layout()
    plt.show()


# ── Run ───────────────────────────────────────────────────────────────────────
for fName in entries:

    # fName = entries[9]

    result = createHead(fName)

    plotUnifiedHead(result, title=fName)