# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:32:56 2026

@author: Bethany.Kilpatrick
"""

"""
Created on Tue Jun 16 15:52:25 2026

@author: Bethany.Kilpatrick

Notes:
    VT STAR score calculator + acceleration / rotational velocity plotter for
    the Mule Tightness project.

    Reads all Cadex export CSVs from fPath. Each CSV filename encodes a helmet
    MODEL (e.g. BOA, OEM) and a tightness CONDITION (e.g. High, AVG, Low):
        Tension_{MODEL}_{CONDITION}_{Date}-ir6.csv
        e.g. Tension_BOA_AVG_June2026-ir6.csv

    Checklists live in checklistDir, one workbook per MODEL:
        2026_ImpactChecklist_MuleTightness_{MODEL}.xlsx
        e.g. 2026_ImpactChecklist_MuleTightness_BOA.xlsx

    Each workbook has one sheet per CONDITION (e.g. FS2_High, FS2_Avg, FS2_Low
    - sheet-name prefixes and "Condition" spelling vary, e.g. "AVG" vs "Avg.",
    which normalize_condition handles) plus a "Mass" sheet (Sample -> mass).
    Other reference sheets (e.g. "OldRef") are ignored. MODEL selects the
    workbook, CONDITION selects the sheet, and that sheet maps
    ImpactOrder -> Location/Sample/Speed.

    STAR equation:
        R = 1 / (1 + exp(-(-10.2 + 0.0433*PLA + 0.19686*PRV - 0.0002075*PLA*PRV)))
    STAR Score = SUMPRODUCT(Exposure, Risk) over 12 conditions
    (6 locations x 2 speeds). Exposure: 4.8 m/s -> 6.33, 7.4 m/s -> 1.57

    Star rating (see star_rating):
        < 10.1 -> 5 | < 13 -> 4 | < 16 -> 3 | < 19 -> 2 | < 21.9 -> 1 | else 0

    PLA/PRV source:
        PLA and PRV come from the raw time-series block (Res.Trans. Acc{n} /
        Res Ang. Velo.{n}), NOT the Cadex summary columns, and are run through
        a peak safeguard (valid_peak) that rejects short, single-axis
        instrumentation glitches. This was added after a ~0.25 ms
        single-channel spike in one impact's Acc.Y trace was reported by the
        Cadex summary column as the peak (127.99 G instead of the real ~65.7 G
        pulse), which would have silently corrupted that impact's Risk/STAR
        contribution. If a CSV has no time-series block, PLA/PRV fall back to
        the Cadex summary columns.

    Plots: only impacts actually used in the STAR score (matched a checklist
    condition, present in the CSV, matched the expected speed) get a PNG.

    Updated Jul 08 2026 - multi-checklist / multi-condition support
    Updated Jul 20 2026 - time-series peak safeguard (glitch rejection for
                          PLA/PRV) + STAR-score-aware plot filtering
    Updated Oct 05 2026 - per-impact rotational velocity plots (X/Y/Z +
                          resultant) saved to 'Rotational Velocity Plots';
                          filename-safe plot names
    Updated Oct 14 2026 - restructured to PFL Python style; PLA peak marker
                          now a toggle (mark_pla_peak); save_on = 2 appends
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
import traceback
from io import StringIO
from openpyxl import load_workbook

# ______________________________________________________________________________
# Define constants and options
save_on = 1         # 0 = no save, 1 = write new file, 2 = append
debug = 1           # 1 = verbose console output
plot_acc_on = 1     # 1 = save per-impact linear acceleration plots
plot_rot_on = 1     # 1 = save per-impact rotational velocity plots (X/Y/Z + resultant)
mark_pla_peak = 0   # 1 = mark the PLA value used for STAR on the acceleration plots
mark_prv_peak = 0   # 1 = mark the PRV value used for STAR on the rotational plots

# Cadex exports
fPath = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_Impact_SystemTension_Mule\CadexExports'

# Folder containing ONE checklist workbook per helmet model, e.g.:
#   2026_ImpactChecklist_MuleTightness_BOA.xlsx
#   2026_ImpactChecklist_MuleTightness_OEM.xlsx
checklistDir = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_Impact_SystemTension_Mule\CheckLists'

outPath = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_Impact_SystemTension_Mule'
outfileName = os.path.join(outPath, 'STAR_outcomes_Updated_RotACC.csv')

# Impact speeds
HIGH_SPEED = 7.4        # [m/s]
LOW_SPEED = 4.8         # [m/s]
SPEED_TOLERANCE = 0.5   # measured speed must be within this of nominal [m/s]

EXPOSURE = {
    LOW_SPEED:  6.33,
    HIGH_SPEED: 1.57,
}

IMPACT_LOCATIONS = [1, 2, 3, 4, 5, 6]
SPEED_CONDITIONS = [HIGH_SPEED, LOW_SPEED]

LOCATION_NAMES = {
    1: 'VSTAR1',
    2: 'VSTAR2',
    3: 'VSTAR3',
    4: 'VSTAR4',
    5: 'VSTAR5',
    6: 'VSTAR6',
}

# Summary-table column names and their positions in the Cadex 6-port export
COL_PLA = 'PLA (g)'
COL_PRV = 'PRV (rad/s)'
COL_SPEED = 'Velocity IN (m/s)'
COL_POSITION = 'Position'

CADEX_COL_PLA = 1
CADEX_COL_VELOCITY = 2
CADEX_COL_POSITION = 11
CADEX_COL_PRV = 38

# Peak safeguard tuning: a real helmet-impact pulse for this protocol lasts
# several ms. Any excursion above PEAK_THRESH_FRAC * peak that is shorter than
# PEAK_MIN_DURATION_MS is treated as an instrumentation glitch and rejected.
PEAK_MIN_DURATION_MS = 1.0  # [ms]
PEAK_THRESH_FRAC = 0.5      # fraction of the candidate peak

# Checklist sheet columns (looked up by header, so column order and extra
# columns like "Torque" don't matter)
CHECKLIST_REQUIRED_COLS = ['Impact Order', 'Location', 'Helmet Model', 'Sample', 'Speed']
CHECKLIST_OPTIONAL_COLS = ['Condition', 'Completed', 'Notes', 'Torque']
IGNORE_SHEET_NAMES = {'mass', 'oldref'}   # not condition/data sheets

# Linear acceleration channels for plotting
ACC_TEMPLATES = {
    'Acc. X':    'Acc. X_{n}',
    'Acc. Y':    'Acc. Y_{n}',
    'Acc. Z':    'Acc. Z_{n}',
    'Resultant': 'Res.Trans. Acc{n}',
}
COLORS = {
    'Acc. X':    '#e74c3c',
    'Acc. Y':    '#2980b9',
    'Acc. Z':    '#27ae60',
    'Resultant': '#8e44ad',
}

ANG_VELO_TEMPLATE = 'Res Ang. Velo.{n}'   # resultant - used for PRV safeguard + plotted

# Rotational velocity channels for plotting. Cadex exports the X/Y/Z angular
# velocity as the angular rate sensor channels ARS 1/2/3. The filtered ARS
# channels are used because they rebuild Res Ang. Velo. exactly (checked on
# Tension_FS2_AVG_June2026-ir6, impact 1: max |sqrt(sum ARS^2) - Res| = 0);
# the raw "ARS 1_n" channels do not (off by up to ~1095).
# Note: the export has two spaces before "after filter"
# Note: ARS 1/2/3 -> X/Y/Z follows Cadex's channel order; not checked against
# the headform axes
# If an exact name isn't found, find_rot_column falls back to a fuzzy match.
ROT_TEMPLATES = {
    'ω X':       'ARS_1  after filter _{n}',
    'ω Y':       'ARS_2  after filter _{n}',
    'ω Z':       'ARS_3  after filter _{n}',
    'Resultant': ANG_VELO_TEMPLATE,
}
ROT_COLORS = {
    'ω X':       '#e74c3c',
    'ω Y':       '#2980b9',
    'ω Z':       '#27ae60',
    'Resultant': '#8e44ad',
}
ROT_UNITS = 'rad/s'

# Output column headers
OUTCOME_HEADERS = [
    'Model', 'Tightness Condition', 'Helmet Model (checklist)',
    'Impact Location', 'Location Name', 'Target Speed [m/s]',
    'Avg PLA (α) [g]', 'Avg PRV (ω) [rad/s]', 'Exposure (E)',
    'Risk (R)', 'n (impacts)', 'Notes', 'Sample Masses (g)',
    'STAR Score', 'STAR Rating',
]

# ______________________________________________________________________________
# Functions

def risk(pla, prv):
    """
    Compute the VT STAR concussion risk for one condition from the average
    peak linear acceleration and peak rotational velocity

    Parameters
    ----------
    pla : float
        peak linear acceleration [g]
    prv : float
        peak rotational velocity [rad/s]

    Returns
    -------
    float
        concussion risk (0-1)

    """
    exponent = -10.2 + 0.0433*pla + 0.19686*prv - 0.0002075*pla*prv
    return 1.0 / (1.0 + math.exp(-exponent))


def star_score(risks, exposures):
    """
    Compute the STAR score as the exposure-weighted sum of risks

    Parameters
    ----------
    risks : list
        risk for each location/speed condition
    exposures : list
        exposure weighting for each location/speed condition

    Returns
    -------
    score : float
        STAR score

    """
    score = 0
    for ii in range(len(risks)):
        score = score + exposures[ii]*risks[ii]
    return score


def star_rating(score):
    """
    Convert a STAR score into a star rating

    Parameters
    ----------
    score : float
        STAR score

    Returns
    -------
    int
        star rating (0-5)

    """
    if score < 10.1: return 5
    if score < 13: return 4
    if score < 16: return 3
    if score < 19: return 2
    if score < 21.9: return 1
    if score > 21.9: return 0


def match_speed(v):
    """
    Match a measured impact speed to the nominal test speed it belongs to

    Parameters
    ----------
    v : float
        measured impact velocity [m/s]

    Returns
    -------
    float or None
        nominal speed [m/s] within SPEED_TOLERANCE, or None if no match

    """
    for speed in SPEED_CONDITIONS:
        if abs(v - speed) <= SPEED_TOLERANCE:
            return speed
    return None


def normalize_condition(val):
    """
    Normalize a tightness condition label so checklist spellings match the
    CSV filename token: 'AVG' / 'Avg.' / 'Average' -> 'AVG',
    'High' / 'HIGH' -> 'HIGH', etc.

    Parameters
    ----------
    val : str
        condition label from a filename, sheet name, or "Condition" column

    Returns
    -------
    str
        normalized condition label ('' if val is None)

    """
    if val is None:
        return ''
    cond = str(val).strip()
    cond = cond.rstrip('.')
    cond = cond.upper()
    if cond in ('AVERAGE', 'AVE', 'AVG'):
        return 'AVG'
    return cond


def valid_peak(time_ms, signal, min_duration_ms=PEAK_MIN_DURATION_MS,
               thresh_frac=PEAK_THRESH_FRAC, max_iter=10):
    """
    Find the largest peak in a signal that is wide enough to be a real
    impact pulse.

    Cadex's summary columns (and a naive max() over a resultant channel) can
    be fooled by very short, single-axis instrumentation glitches that are not
    real impact kinematics. A candidate peak is accepted only if its
    excursion above thresh_frac * peak lasts at least min_duration_ms - real
    helmet-impact pulses last several ms, while a glitch lasting a fraction
    of a ms is not physically a head/helmet deceleration. If the largest peak
    fails, it is masked out and the search repeats, so a genuine (slightly
    smaller) impact peak underneath a glitch is still found.

    Parameters
    ----------
    time_ms : numpy array
        time (ms)
    signal : numpy array
        resultant acceleration [g] or angular velocity [rad/s]
    min_duration_ms : float
        minimum excursion width for a valid peak (ms)
    thresh_frac : float
        fraction of the candidate peak that defines the excursion
    max_iter : int
        maximum number of glitches to mask before giving up

    Returns
    -------
    peak_val : float or None
        value of the valid peak (None if nothing qualifies)
    peak_t : float or None
        time of the valid peak (ms)
    duration : float
        width of the excursion above threshold (ms)

    """
    sig_cp = np.asarray(signal, dtype=float).copy()
    t = np.asarray(time_ms, dtype=float)

    for ii in range(max_iter):
        if not np.any(np.isfinite(sig_cp)):
            return [None, None, 0.0]
        idx = int(np.nanargmax(sig_cp))
        peak_val = sig_cp[idx]
        if not np.isfinite(peak_val) or peak_val <= 0:
            return [None, None, 0.0]

        thresh = thresh_frac * peak_val
        above = sig_cp >= thresh

        # Walk out from the peak to the edges of the excursion
        i0 = idx
        while i0 > 0 and above[i0 - 1]:
            i0 -= 1
        i1 = idx
        while i1 < len(sig_cp) - 1 and above[i1 + 1]:
            i1 += 1

        duration = t[i1] - t[i0]
        if duration >= min_duration_ms:
            return [peak_val, t[idx], duration]

        # Too narrow to be a real impact pulse -> mask it out, try the next peak
        sig_cp[i0:i1 + 1] = -np.inf

    return [None, None, 0.0]


def compute_safeguarded_peaks(ts_df, n_impacts,
                              min_duration_ms=PEAK_MIN_DURATION_MS,
                              thresh_frac=PEAK_THRESH_FRAC):
    """
    Compute the safeguarded PLA (from Res.Trans. Acc{n}) and PRV (from
    Res Ang. Velo.{n}) for every impact using valid_peak. The naive
    (unsafeguarded) peak is also kept, and each impact is flagged when the
    safeguard changed the result (i.e. a glitch was excluded) so it can be
    logged.

    Parameters
    ----------
    ts_df : dataframe
        Cadex time-series block (from load_timeseries)
    n_impacts : int
        number of impacts in the time-series block
    min_duration_ms : float
        minimum excursion width for a valid peak (ms)
    thresh_frac : float
        fraction of the candidate peak that defines the excursion

    Returns
    -------
    peaks : dict
        {impact_num: {PLA, PLA_time_ms, PLA_duration_ms, PLA_naive,
        PLA_naive_time_ms, PLA_flagged, PRV, PRV_time_ms, PRV_duration_ms,
        PRV_naive, PRV_naive_time_ms, PRV_flagged}}

    """
    if ts_df is None or 'Time (s)' not in ts_df.columns:
        return {}

    time_ms = (ts_df['Time (s)'] * 1000).values
    peaks = {}

    for n in range(1, n_impacts + 1):
        acc_col = f'Res.Trans. Acc{n}'
        ang_col = ANG_VELO_TEMPLATE.format(n=n)
        entry = {}

        # Peak linear acceleration
        if acc_col in ts_df.columns and not ts_df[acc_col].isna().all():
            acc_sig = ts_df[acc_col].values
            naive_idx = int(np.nanargmax(acc_sig))
            naive_pla = float(acc_sig[naive_idx])
            naive_pla_t = float(time_ms[naive_idx])
            [pla, t_pla, dur_pla] = valid_peak(time_ms, acc_sig, min_duration_ms, thresh_frac)
            pla_flagged = pla is not None and abs(naive_pla - pla) > 1e-6
            entry.update({
                'PLA': pla, 'PLA_time_ms': t_pla, 'PLA_duration_ms': dur_pla,
                'PLA_naive': naive_pla, 'PLA_naive_time_ms': naive_pla_t,
                'PLA_flagged': pla_flagged,
            })
        else:
            entry.update({'PLA': None, 'PLA_time_ms': None, 'PLA_duration_ms': 0.0,
                          'PLA_naive': None, 'PLA_naive_time_ms': None, 'PLA_flagged': False})

        # Peak rotational velocity
        if ang_col in ts_df.columns and not ts_df[ang_col].isna().all():
            ang_sig = ts_df[ang_col].values
            naive_idx_w = int(np.nanargmax(ang_sig))
            naive_prv = float(ang_sig[naive_idx_w])
            naive_prv_t = float(time_ms[naive_idx_w])
            [prv, t_prv, dur_prv] = valid_peak(time_ms, ang_sig, min_duration_ms, thresh_frac)
            prv_flagged = prv is not None and abs(naive_prv - prv) > 1e-6
            entry.update({
                'PRV': prv, 'PRV_time_ms': t_prv, 'PRV_duration_ms': dur_prv,
                'PRV_naive': naive_prv, 'PRV_naive_time_ms': naive_prv_t,
                'PRV_flagged': prv_flagged,
            })
        else:
            entry.update({'PRV': None, 'PRV_time_ms': None, 'PRV_duration_ms': 0.0,
                          'PRV_naive': None, 'PRV_naive_time_ms': None, 'PRV_flagged': False})

        peaks[n] = entry

    return peaks


def discover_checklist_files(checklist_dir):
    """
    Map each helmet MODEL token to its checklist workbook. The MODEL token is
    the last underscore-separated segment of the filename, e.g.
    '2026_ImpactChecklist_MuleTightness_BOA.xlsx' -> 'BOA'.

    Parameters
    ----------
    checklist_dir : str
        folder holding the checklist workbooks

    Returns
    -------
    mapping : dict
        {MODEL token: workbook path}

    """
    mapping = {}
    for fn in os.listdir(checklist_dir):
        # Skip non-workbooks and Excel lock files
        if not fn.lower().endswith('.xlsx') or fn.startswith('~$'):
            continue
        stem = os.path.splitext(fn)[0]
        model_token = stem.split('_')[-1].upper()
        mapping[model_token] = os.path.join(checklist_dir, fn)
    return mapping


def sheet_header(ws):
    """
    Read row 1 of a worksheet as a list of stripped strings

    Parameters
    ----------
    ws : openpyxl worksheet
        checklist or mass sheet

    Returns
    -------
    header : list
        column names ('' for empty cells)

    """
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
    header = []
    for h in header_row:
        if h is None:
            header.append('')
        else:
            header.append(str(h).strip())
    return header


def sheet_header_index(ws):
    """
    Locate the recognized checklist columns in a sheet's header row

    Parameters
    ----------
    ws : openpyxl worksheet
        checklist condition sheet

    Returns
    -------
    idx : dict
        {column name: column index} for whichever recognized columns exist

    """
    header = sheet_header(ws)
    idx = {}
    for name in CHECKLIST_REQUIRED_COLS + CHECKLIST_OPTIONAL_COLS:
        if name in header:
            idx[name] = header.index(name)
    return idx


def find_condition_sheet(wb, condition_token):
    """
    Find the sheet in a checklist workbook that holds the given tightness
    condition. The sheet's own "Condition" column is checked first since it
    is authoritative regardless of how the tab is named; the sheet-name
    suffix (e.g. "FS2_High" -> "HIGH") is the fallback.

    Parameters
    ----------
    wb : openpyxl workbook
        checklist workbook for one helmet model
    condition_token : str
        condition from the CSV filename (e.g. 'AVG')

    Returns
    -------
    str or None
        matching sheet name, or None if no sheet matches

    """
    target = normalize_condition(condition_token)

    # Pass 1: match on the first value in the sheet's "Condition" column
    for sheet_name in wb.sheetnames:
        if sheet_name.strip().lower() in IGNORE_SHEET_NAMES:
            continue
        ws = wb[sheet_name]
        idx = sheet_header_index(ws)
        if 'Impact Order' not in idx or 'Condition' not in idx:
            continue
        cond_val = None
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[idx['Condition']] is not None:
                cond_val = row[idx['Condition']]
                break
        if normalize_condition(cond_val) == target:
            return sheet_name

    # Pass 2: fall back to matching the sheet-name suffix
    for sheet_name in wb.sheetnames:
        if sheet_name.strip().lower() in IGNORE_SHEET_NAMES:
            continue
        suffix = sheet_name.split('_')[-1]
        if normalize_condition(suffix) == target:
            return sheet_name

    return None


def load_checklist_from_sheet(ws, idx):
    """
    Read one checklist condition sheet, keeping only rows with a valid
    (positive integer) Impact Order

    Parameters
    ----------
    ws : openpyxl worksheet
        checklist condition sheet
    idx : dict
        column indices from sheet_header_index

    Returns
    -------
    df : dataframe
        ImpactOrder, Location, Helmet Model, Sample, Speed ('high'/'low'),
        Completed, Notes

    """
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        impact_order = row[idx['Impact Order']]
        location = row[idx['Location']]
        helmet_model = row[idx['Helmet Model']]
        sample = row[idx['Sample']]
        speed = row[idx['Speed']]
        notes = row[idx['Notes']] if 'Notes' in idx else None
        completed = row[idx['Completed']] if 'Completed' in idx else None

        try:
            impact_num = int(impact_order)
            if impact_num <= 0:
                continue
        except (TypeError, ValueError):
            if debug and impact_order is not None:
                print(f'    Checklist: skipping row (ImpactOrder={impact_order!r}, '
                      f'Model={helmet_model}, Sample={sample}, Loc={location}, Speed={speed})')
            continue

        if speed:
            speed_label = str(speed).strip()
            speed_label = speed_label.lower()
        else:
            speed_label = ''

        rows.append({
            'ImpactOrder':  impact_num,
            'Location':     location,
            'Helmet Model': helmet_model,   # Note: space in key - must match everywhere
            'Sample':       sample,
            'Speed':        speed_label,
            'Completed':    completed,
            'Notes':        notes,
        })
    df = pd.DataFrame(rows)

    if not df.empty:
        dupes = df[df.duplicated('ImpactOrder', keep=False)]
        if not dupes.empty:
            print(f'    WARNING: duplicate ImpactOrder values in checklist sheet:\n{dupes}')
    return df


def load_mass(wb):
    """
    Read the "Mass" sheet of a checklist workbook

    Parameters
    ----------
    wb : openpyxl workbook
        checklist workbook for one helmet model

    Returns
    -------
    mass_map : dict
        {sample: mass [g]}; empty if the sheet or its columns are missing

    """
    if 'Mass' not in wb.sheetnames:
        return {}
    ws = wb['Mass']
    header = sheet_header(ws)
    if 'Sample' not in header or 'Mass' not in header:
        return {}
    s_idx = header.index('Sample'); m_idx = header.index('Mass')

    mass_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sample = row[s_idx]; mass = row[m_idx]
        if sample is not None and mass is not None:
            mass_map[str(sample).strip()] = mass
    return mass_map


def read_lines(path):
    """
    Read a Cadex CSV as raw lines, trying common encodings in turn

    Parameters
    ----------
    path : str
        CSV file path

    Returns
    -------
    list
        lines of the file

    """
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(path, encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Could not decode {path}.')


def load_summary(lines, fname):
    """
    Extract the per-impact summary rows from a Cadex CSV. The index is
    1-based so checklist ImpactOrder numbers map directly onto rows.

    Parameters
    ----------
    lines : list
        raw lines of the CSV (from read_lines)
    fname : str
        file name, for error messages

    Returns
    -------
    df : dataframe
        PLA [g], velocity [m/s], position, and PRV [rad/s] per impact

    """
    # Summary rows start at the first line whose first field is "1"
    data_start = None
    for count, line in enumerate(lines):
        if line.split(',')[0].strip() == '1':
            data_start = count
            break
    if data_start is None:
        raise ValueError(f'No data rows found in {fname}.')

    # Keep only rows whose first field is a positive impact number
    valid = []
    for line in lines[data_start:]:
        first = line.split(',')[0].strip()
        if first.isdigit() and int(first) > 0:
            valid.append(line)
    if not valid:
        raise ValueError(f'No valid impact rows in {fname}.')

    raw = pd.read_csv(StringIO(''.join(valid)), header=None, on_bad_lines='skip')
    max_needed = max(CADEX_COL_PLA, CADEX_COL_VELOCITY, CADEX_COL_POSITION, CADEX_COL_PRV)
    if raw.shape[1] <= max_needed:
        raise ValueError(f'{fname} has too few columns for a Cadex 6-Port export.')

    df = raw[[CADEX_COL_PLA, CADEX_COL_VELOCITY, CADEX_COL_POSITION, CADEX_COL_PRV]].copy()
    df.columns = [COL_PLA, COL_SPEED, COL_POSITION, COL_PRV]
    for col in [COL_PLA, COL_SPEED, COL_PRV]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.index = range(1, len(df) + 1)
    return df


def load_timeseries(lines):
    """
    Extract the time-series waveform block from a Cadex CSV

    Parameters
    ----------
    lines : list
        raw lines of the CSV (from read_lines)

    Returns
    -------
    dataframe or None
        one column per channel per impact, plus "Time (s)"; None if the CSV
        has no time-series block

    """
    ts_start = None
    for count, line in enumerate(lines):
        if line.startswith('Time (s)'):
            ts_start = count
            break
    if ts_start is None:
        return None

    col_names = [c.strip() for c in lines[ts_start].rstrip('\n').split(',')]

    # Read until a blank line, an "end" marker, or a non-numeric time value
    data_lines = []
    for line in lines[ts_start + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('end'):
            break
        try:
            float(stripped.split(',')[0].strip())
            data_lines.append(stripped)
        except ValueError:
            break
    if not data_lines:
        return None

    df = pd.read_csv(
        StringIO('\n'.join(data_lines)),
        header=None, names=col_names,
        skipinitialspace=True, on_bad_lines='skip',
    )
    return df.apply(pd.to_numeric, errors='coerce')


def calculate_star(summary_df, checklist_df, helmet_model, safeguarded_peaks=None):
    """
    Map checklist ImpactOrder numbers onto CSV rows and compute the STAR
    score.

    For each of the 12 location/speed conditions, the checklist impacts are
    looked up in the CSV, rejected if missing or if the measured speed does
    not match the nominal speed, and the remaining PLA/PRV values averaged
    into a single risk. PLA/PRV come from safeguarded_peaks (time-series,
    glitch-rejected) when available for an impact, otherwise from the Cadex
    summary columns.

    Parameters
    ----------
    summary_df : dataframe
        per-impact summary (from load_summary)
    checklist_df : dataframe
        checklist condition sheet (from load_checklist_from_sheet)
    helmet_model : str
        helmet model label used in the checklist
    safeguarded_peaks : dict
        per-impact peaks (from compute_safeguarded_peaks)

    Returns
    -------
    conditions : list
        one dict per location/speed condition (averages, risk, notes, samples)
    score : float or None
        STAR score
    rating : int or None
        star rating
    good_impacts : set
        impact numbers actually used in the score
    flagged_impacts : set
        impact numbers where a glitch was excluded

    """
    model_rows = checklist_df[checklist_df['Helmet Model'] == helmet_model].copy()
    if model_rows.empty:
        # Sheet is already model-specific, so this should rarely trigger - fall
        # back to the whole sheet if the label doesn't match exactly
        model_rows = checklist_df.copy()

    # Preallocate
    conditions = []
    good_impacts = set()
    flagged_impacts = set()

    for loc in IMPACT_LOCATIONS:
        for speed_nominal in SPEED_CONDITIONS:

            speed_label = 'high' if speed_nominal == HIGH_SPEED else 'low'
            cond_rows = model_rows[
                (model_rows['Location'] == loc) &
                (model_rows['Speed'] == speed_label)
            ]

            pla_vals = []; prv_vals = []; sample_vals = []
            skipped = []; flagged_here = []

            for jj, crow in cond_rows.iterrows():
                impact_num = crow['ImpactOrder']

                if impact_num not in summary_df.index:
                    skipped.append(f'#{impact_num} not in CSV')
                    continue

                row = summary_df.loc[impact_num]
                measured_speed = match_speed(row[COL_SPEED])

                if measured_speed != speed_nominal:
                    skipped.append(
                        f'#{impact_num} speed mismatch '
                        f'(expected {speed_nominal}, got {row[COL_SPEED]:.2f})'
                    )
                    continue

                # Prefer the safeguarded time-series peak over the Cadex
                # summary column, which can be fooled by short single-axis
                # instrumentation glitches
                sg = None
                if safeguarded_peaks:
                    sg = safeguarded_peaks.get(impact_num)

                if sg and sg.get('PLA') is not None and sg.get('PRV') is not None:
                    pla_val = sg['PLA']
                    prv_val = sg['PRV']
                    if sg['PLA_flagged'] or sg['PRV_flagged']:
                        flagged_here.append(impact_num)
                        flagged_impacts.add(impact_num)
                        if debug:
                            if sg['PLA_flagged']:
                                print(f"      Impact #{impact_num}: PLA glitch excluded - "
                                      f"{sg['PLA_naive']:.2f}G at t={sg['PLA_naive_time_ms']:.2f}ms "
                                      f"-> using {pla_val:.2f}G at t={sg['PLA_time_ms']:.2f}ms")
                            if sg['PRV_flagged']:
                                print(f"      Impact #{impact_num}: PRV glitch excluded - "
                                      f"{sg['PRV_naive']:.3f} rad/s at t={sg['PRV_naive_time_ms']:.2f}ms "
                                      f"-> using {prv_val:.3f} rad/s at t={sg['PRV_time_ms']:.2f}ms")
                else:
                    # No time-series safeguard for this impact (e.g. CSV had no
                    # time-series block) -> fall back to the Cadex summary columns
                    pla_val = row[COL_PLA]
                    prv_val = row[COL_PRV]

                pla_vals.append(pla_val)
                prv_vals.append(prv_val)
                sample_vals.append(str(crow['Sample']))
                good_impacts.add(impact_num)

            if skipped and debug:
                print(f'      Loc {loc} {speed_label}: skipped {skipped}')

            if not pla_vals:
                conditions.append({
                    'Impact Location':     loc,
                    'Location Name':       LOCATION_NAMES[loc],
                    'Target Speed [m/s]':  speed_nominal,
                    'Avg PLA (α) [g]':     None,
                    'Avg PRV (ω) [rad/s]': None,
                    'Exposure (E)':        EXPOSURE[speed_nominal],
                    'Risk (R)':            None,
                    'n (impacts)':         0,
                    'Notes':               'No valid data',
                    'SampleMasses':        [],
                })
            else:
                avg_pla = sum(pla_vals) / len(pla_vals)
                avg_prv = sum(prv_vals) / len(prv_vals)
                cond_risk = risk(avg_pla, avg_prv)
                note = f'n={len(pla_vals)}'
                if flagged_here:
                    note += f'; glitch excluded on impact(s) {flagged_here}'
                conditions.append({
                    'Impact Location':     loc,
                    'Location Name':       LOCATION_NAMES[loc],
                    'Target Speed [m/s]':  speed_nominal,
                    'Avg PLA (α) [g]':     avg_pla,
                    'Avg PRV (ω) [rad/s]': avg_prv,
                    'Exposure (E)':        EXPOSURE[speed_nominal],
                    'Risk (R)':            cond_risk,
                    'n (impacts)':         len(pla_vals),
                    'Notes':               note,
                    'SampleMasses':        sample_vals,
                })

    # Score only the conditions that had valid data
    valid_risks = []; valid_exposures = []
    for c in conditions:
        if c['Risk (R)'] is not None:
            valid_risks.append(c['Risk (R)'])
            valid_exposures.append(c['Exposure (E)'])

    score = None; rating = None
    if valid_risks:
        score = star_score(valid_risks, valid_exposures)
        rating = star_rating(score)

    return [conditions, score, rating, good_impacts, flagged_impacts]


def build_outcome_rows(conditions, score, rating, model_token, condition_token,
                       helmet_model, mass_map):
    """
    Convert one CSV's STAR results into rows for the outcomes CSV. The STAR
    score and rating are written on the first row only.

    Parameters
    ----------
    conditions : list
        per-condition results (from calculate_star)
    score : float or None
        STAR score
    rating : int or None
        star rating
    model_token : str
        helmet model from the CSV filename
    condition_token : str
        tightness condition from the CSV filename
    helmet_model : str
        helmet model label from the checklist
    mass_map : dict
        {sample: mass [g]} (from load_mass)

    Returns
    -------
    rows : list
        one list per condition, ordered as OUTCOME_HEADERS

    """
    rows = []

    for count, c in enumerate(conditions):
        # List each sample used with its mass
        mass_parts = []
        for s in c.get('SampleMasses', []):
            if s in mass_map:
                mass_parts.append(f'{s}:{mass_map[s]}g')
            else:
                mass_parts.append(f'{s}:?')
        mass_str = ', '.join(mass_parts)

        avg_pla = c['Avg PLA (α) [g]']
        avg_prv = c['Avg PRV (ω) [rad/s]']
        cond_risk = c['Risk (R)']

        rows.append([
            model_token,
            condition_token,
            helmet_model,
            c['Impact Location'],
            c['Location Name'],
            c['Target Speed [m/s]'],
            round(avg_pla, 4) if avg_pla is not None else '',
            round(avg_prv, 4) if avg_prv is not None else '',
            c['Exposure (E)'],
            round(cond_risk, 6) if cond_risk is not None else '',
            c['n (impacts)'],
            c['Notes'],
            mass_str,
            round(score, 4) if (count == 0 and score is not None) else '',
            rating if (count == 0 and rating is not None) else '',
        ])
    return rows


def build_impact_info(checklist_df, helmet_model, label):
    """
    Build plot-title labels for each impact,
    e.g. {3: "BOA_AVG  |  Loc 2 VSTAR2  |  High 7.4 m/s"}

    Parameters
    ----------
    checklist_df : dataframe
        checklist condition sheet (from load_checklist_from_sheet)
    helmet_model : str
        helmet model label used in the checklist
    label : str
        MODEL_CONDITION label for the CSV

    Returns
    -------
    info : dict
        {impact number: title label}

    """
    model_rows = checklist_df[checklist_df['Helmet Model'] == helmet_model]
    if model_rows.empty:
        model_rows = checklist_df

    info = {}
    for jj, row in model_rows.iterrows():
        loc = row['Location']
        loc_name = LOCATION_NAMES.get(loc, f'Loc{loc}')
        if row['Speed'] == 'high':
            speed_str = f'High {HIGH_SPEED} m/s'
        else:
            speed_str = f'Low {LOW_SPEED} m/s'
        info[row['ImpactOrder']] = f'{label}  |  Loc {loc} {loc_name}  |  {speed_str}'
    return info


def plot_filename(csv_stem, impact_num, impact_info=None, suffix=''):
    """
    Build a Windows-safe PNG name for an impact plot. The plot titles contain
    '|' and 'm/s', which are illegal / path separators in filenames, so they
    are cleaned up here,
    e.g. Tension_BOA_AVG_June2026-ir6_impact03_Loc2_VSTAR2_High_7.4ms.png

    Parameters
    ----------
    csv_stem : str
        CSV filename without extension
    impact_num : int
        impact number (1-based)
    impact_info : dict
        {impact number: title label} (from build_impact_info)
    suffix : str
        appended before the extension (e.g. '_rotvel')

    Returns
    -------
    str
        PNG filename

    """
    tag = ''
    if impact_info and impact_num in impact_info:
        # Drop the leading "MODEL_COND" label (already in csv_stem)
        parts = [p.strip() for p in impact_info[impact_num].split('|')[1:]]
        tag = '_' + '_'.join(parts)
        tag = tag.replace('m/s', 'ms')
        tag = re.sub(r'[<>:"/\\|?*]', '', tag)
        tag = re.sub(r'\s+', '_', tag)
    return f'{csv_stem}_impact{impact_num:02d}{tag}{suffix}.png'


def plot_accelerations(ts_df, n_impacts, csv_path, impact_info=None,
                       good_impacts=None, safeguarded_peaks=None):
    """
    Save one linear acceleration PNG (X/Y/Z + resultant) per impact to an
    'Acceleration Plots' folder next to the CSV. Only impacts used in the
    STAR calculation are plotted; the rest were not part of the rating
    (checklist mismatch, wrong speed, missing data).

    Parameters
    ----------
    ts_df : dataframe
        Cadex time-series block (from load_timeseries)
    n_impacts : int
        number of impacts in the time-series block
    csv_path : str
        path of the source CSV
    impact_info : dict
        {impact number: title label}; missing keys fall back to the CSV name
    good_impacts : set
        impact numbers used in the STAR score; None plots every impact
    safeguarded_peaks : dict
        per-impact peaks, used to mark the PLA used when mark_pla_peak = 1

    Returns
    -------
    None

    """
    if ts_df is None or 'Time (s)' not in ts_df.columns:
        print('    No time-series data - skipping plots.')
        return

    time_ms = ts_df['Time (s)'] * 1000
    plot_dir = os.path.join(os.path.dirname(csv_path), 'Acceleration Plots')
    os.makedirs(plot_dir, exist_ok=True)

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    n_skipped = 0

    for impact_num in range(1, n_impacts + 1):
        if good_impacts is not None and impact_num not in good_impacts:
            n_skipped += 1
            continue

        if impact_info and impact_num in impact_info:
            title = f'Impact #{impact_num}  —  {impact_info[impact_num]}'
        else:
            title = f'Impact #{impact_num}  —  {csv_stem}'

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.suptitle(title, fontsize=11, fontweight='bold')

        for chan_label, template in ACC_TEMPLATES.items():
            col = template.replace('{n}', str(impact_num))
            if col not in ts_df.columns or ts_df[col].isna().all():
                continue
            ax.plot(
                time_ms, ts_df[col],
                label=chan_label,
                color=COLORS[chan_label],
                linewidth=1.6 if chan_label == 'Resultant' else 1.0,
                linestyle='-' if chan_label == 'Resultant' else '--',
                alpha=0.9,
            )

        # Mark the PLA peak actually used for STAR (excluded glitches are not
        # annotated)
        if mark_pla_peak == 1 and safeguarded_peaks:
            sg = safeguarded_peaks.get(impact_num)
            if sg and sg.get('PLA') is not None:
                ax.plot(sg['PLA_time_ms'], sg['PLA'], marker='x', markersize=16,
                        markeredgecolor='black', markerfacecolor='red', zorder=5,
                        linestyle='None', label=f"PLA used: {sg['PLA']:.1f}G")
                ax.annotate(f"{sg['PLA']:.1f} G", (sg['PLA_time_ms'], sg['PLA']),
                            textcoords='offset points', xytext=(6, 6), fontsize=8,
                            fontweight='bold')

        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Acceleration [G]')
        ax.set_xlabel('Time [ms]')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_name = plot_filename(csv_stem, impact_num, impact_info)
        out_path = os.path.join(plot_dir, out_name)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Plot saved -> {out_path}')

    if good_impacts is not None and n_skipped:
        print(f'    Skipped {n_skipped} impact(s) not used in the STAR calc '
              f'(no plot saved).')


def find_rot_column(ts_df, chan_label, template, impact_num):
    """
    Resolve the time-series column for one rotational velocity channel of
    one impact. The exact template name is tried first, then a fuzzy match so
    naming differences in the Cadex export don't drop the X/Y/Z traces.

    The fuzzy match compares column names on lower-case letters and digits
    only. A column matches when it ends in the impact number (not part of a
    larger number, so impact 1 never picks up impact 11), names a rotation
    ("ang" or "rot") and a velocity ("vel" or "rate"), and is not an
    acceleration channel. The resultant must start with "res"; an axis
    channel must not, and its axis letter may sit at either end of the name
    (e.g. "Ang. Velo. X_3", "AngVel X 3", "X Ang. Vel._3").

    Parameters
    ----------
    ts_df : dataframe
        Cadex time-series block (from load_timeseries)
    chan_label : str
        key in ROT_TEMPLATES ('ω X', 'ω Y', 'ω Z', 'Resultant')
    template : str
        expected column name with an {n} placeholder
    impact_num : int
        impact number (1-based)

    Returns
    -------
    str or None
        matching column name, or None if not found

    """
    exact = template.replace('{n}', str(impact_num))
    if exact in ts_df.columns:
        return exact

    if chan_label == 'Resultant':
        want_axis = None
    else:
        want_axis = chan_label[-1].lower()

    for col in ts_df.columns:
        # Compare on lower-case letters and digits only
        norm = re.sub(r'[^a-z0-9]', '', str(col).lower())

        # Split the trailing impact number off the channel name
        m = re.match(r'^(.*?[a-z])(\d+)$', norm)
        if not m:
            continue
        body = m.group(1)
        col_num = int(m.group(2))
        if col_num != impact_num:
            continue

        is_rot = 'ang' in body or 'rot' in body
        is_vel = 'vel' in body or 'rate' in body
        if not is_rot or not is_vel or 'acc' in body:
            continue

        if want_axis is None:
            if body.startswith('res'):
                return col
        elif not body.startswith('res'):
            # Note: "...velocity" ends in "y" without naming an axis, so the
            # trailing letter only counts as an axis when it isn't that word
            axis_at_end = body[-1] == want_axis and not body.endswith('velocity')
            axis_at_start = body[0] == want_axis
            if axis_at_end or axis_at_start:
                return col
    return None


def plot_rotational_velocity(ts_df, n_impacts, csv_path, impact_info=None,
                             good_impacts=None, safeguarded_peaks=None):
    """
    Save one angular velocity PNG (X/Y/Z + resultant) per impact to a
    'Rotational Velocity Plots' folder next to the CSV. Same filtering as
    plot_accelerations: only impacts used in the STAR calculation are
    plotted.

    Parameters
    ----------
    ts_df : dataframe
        Cadex time-series block (from load_timeseries)
    n_impacts : int
        number of impacts in the time-series block
    csv_path : str
        path of the source CSV
    impact_info : dict
        {impact number: title label}; missing keys fall back to the CSV name
    good_impacts : set
        impact numbers used in the STAR score; None plots every impact
    safeguarded_peaks : dict
        per-impact peaks, used to mark the PRV used when mark_prv_peak = 1

    Returns
    -------
    None

    """
    if ts_df is None or 'Time (s)' not in ts_df.columns:
        print('    No time-series data - skipping rotational plots.')
        return

    time_ms = ts_df['Time (s)'] * 1000
    plot_dir = os.path.join(os.path.dirname(csv_path), 'Rotational Velocity Plots')
    os.makedirs(plot_dir, exist_ok=True)

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    missing_warned = False

    for impact_num in range(1, n_impacts + 1):
        if good_impacts is not None and impact_num not in good_impacts:
            continue

        if impact_info and impact_num in impact_info:
            title = f'Impact #{impact_num}  —  {impact_info[impact_num]}'
        else:
            title = f'Impact #{impact_num}  —  {csv_stem}'

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.suptitle(title, fontsize=11, fontweight='bold')

        missing = []
        for chan_label, template in ROT_TEMPLATES.items():
            col = find_rot_column(ts_df, chan_label, template, impact_num)
            if col is None or ts_df[col].isna().all():
                missing.append(chan_label)
                continue
            ax.plot(
                time_ms, ts_df[col],
                label=chan_label,
                color=ROT_COLORS[chan_label],
                linewidth=1.6 if chan_label == 'Resultant' else 1.0,
                linestyle='-' if chan_label == 'Resultant' else '--',
                alpha=0.9,
            )

        if len(missing) == len(ROT_TEMPLATES):
            plt.close(fig)
            print(f'    Impact #{impact_num}: no rotational velocity channels found - skipped.')
            continue

        # Warn once per CSV, listing every channel the export has for this
        # impact (not only ones containing "ang") so the real X/Y/Z names can
        # be copied into ROT_TEMPLATES
        if missing and not missing_warned:
            impact_cols = []
            for c in ts_df.columns:
                trailing = re.search(r'(\d+)$', str(c))
                if trailing and int(trailing.group(1)) == impact_num:
                    impact_cols.append(c)
            print(f'    WARNING: rotational channel(s) {missing} not found for impact '
                  f'#{impact_num}. Channels present for this impact: {impact_cols}. '
                  f'Update ROT_TEMPLATES to match.')
            missing_warned = True

        # Mark the PRV peak actually used for STAR
        if mark_prv_peak == 1 and safeguarded_peaks:
            sg = safeguarded_peaks.get(impact_num)
            if sg and sg.get('PRV') is not None:
                ax.plot(sg['PRV_time_ms'], sg['PRV'], marker='x', markersize=12,
                        markeredgecolor='black', linestyle='None', zorder=5,
                        label=f"PRV used: {sg['PRV']:.2f} {ROT_UNITS}")

        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_ylabel(f'Angular Velocity [{ROT_UNITS}]')
        ax.set_xlabel('Time [ms]')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_name = plot_filename(csv_stem, impact_num, impact_info, suffix='_rotvel')
        out_path = os.path.join(plot_dir, out_name)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Rot. plot saved -> {out_path}')

# ______________________________________________________________________________
# Discover checklist workbooks (one per model) and Cadex CSVs
checklist_files = discover_checklist_files(checklistDir)
if not checklist_files:
    print(f'  ERROR: No checklist .xlsx files found in: {checklistDir}')
else:
    print(f'  Checklist files found -> {checklist_files}\n')

entries = [fName for fName in os.listdir(fPath) if fName.lower().endswith('.csv')]
entries.sort()
if not entries:
    print(f'No CSV files found in: {fPath}')
else:
    print(f'Found {len(entries)} CSV file(s) in {fPath}\n')

# Preallocate variables
outcome_rows = []
badFileList = []

# ______________________________________________________________________________
# Index through the CSVs
for ii in range(0, len(entries)):
    fName = entries[ii]
    csv_path = os.path.join(fPath, fName)
    stem = os.path.splitext(fName)[0]

    # Expect: Tension_{MODEL}_{CONDITION}_{Date}-ir6
    parts = stem.split('_')
    if len(parts) < 3:
        print(f"    NOTE: filename '{fName}' doesn't match the expected pattern "
              f"'Prefix_Model_Condition_Date-ir6' - skipping.")
        print()
        continue

    model_token = parts[1].upper()
    condition_token = parts[2].upper()

    checklist_path = checklist_files.get(model_token)
    if checklist_path is None:
        print(f"    NOTE: no checklist file found for model '{model_token}' "
              f"(from '{fName}'). Available models: {list(checklist_files.keys())}")
        print()
        continue

    try:
        #______________________________________________________________________
        # Load the checklist sheet for this model/condition
        wb = load_workbook(checklist_path, data_only=True)
        sheet_name = find_condition_sheet(wb, condition_token)
        if sheet_name is None:
            print(f"    NOTE: no sheet matching condition '{condition_token}' in "
                  f"{os.path.basename(checklist_path)}. Sheets: {wb.sheetnames}")
            print()
            continue

        ws = wb[sheet_name]
        idx = sheet_header_index(ws)
        checklist_df = load_checklist_from_sheet(ws, idx)
        mass_map = load_mass(wb)

        if checklist_df.empty:
            print(f"    ERROR: no valid ImpactOrder rows in sheet '{sheet_name}' "
                  f"of {os.path.basename(checklist_path)}.")
            badFileList.append(fName)
            print()
            continue

        # Use the checklist's helmet model label when present
        model_labels = checklist_df['Helmet Model'].dropna()
        if model_labels.empty:
            helmet_model = model_token
        else:
            helmet_model = model_labels.iloc[0]
        label = f'{model_token}_{condition_token}'

        print(f'    {fName}')
        print(f'      -> checklist: {os.path.basename(checklist_path)} '
              f'[{sheet_name}]  (Helmet Model = {helmet_model})')

        #______________________________________________________________________
        # Load the Cadex summary and time-series blocks
        lines = read_lines(csv_path)
        summary_df = load_summary(lines, fName)

        # Note: the time series and safeguarded peaks are computed BEFORE the
        # STAR calc so PLA/PRV come from the glitch-rejected time series
        # instead of blindly trusting the Cadex summary columns
        ts_df = load_timeseries(lines)
        if ts_df is not None:
            n_impacts = 0
            for jj in range(1, 50):
                if f'Res.Trans. Acc{jj}' in ts_df.columns:
                    n_impacts += 1
            print(f'      Time-series: {len(ts_df)} samples, {n_impacts} impact(s)')
            safeguarded_peaks = compute_safeguarded_peaks(ts_df, n_impacts)
        else:
            n_impacts = 0
            safeguarded_peaks = {}
            print('      No time-series block found - PLA/PRV will fall back '
                  'to Cadex summary columns.')

        #______________________________________________________________________
        # STAR score
        [conditions, score, rating, good_impacts, flagged_impacts] = calculate_star(
            summary_df, checklist_df, helmet_model, safeguarded_peaks)

        if score is not None:
            print(f'      STAR Score:  {score:.4f}')
            print(f'      STAR Rating: {rating}/5')
        else:
            print('      WARNING: insufficient data to compute STAR score.')

        if flagged_impacts:
            print(f'      NOTE: instrumentation glitch excluded from PLA/PRV peak '
                  f'for impact(s): {sorted(flagged_impacts)}')

        new_rows = build_outcome_rows(conditions, score, rating, model_token,
                                      condition_token, helmet_model, mass_map)
        outcome_rows.extend(new_rows)

        #______________________________________________________________________
        # Plots - only for impacts actually used in the STAR calc
        if ts_df is not None:
            impact_info = build_impact_info(checklist_df, helmet_model, label)
            if plot_acc_on == 1:
                plot_accelerations(ts_df, n_impacts, csv_path, impact_info,
                                   good_impacts=good_impacts,
                                   safeguarded_peaks=safeguarded_peaks)
            if plot_rot_on == 1:
                plot_rotational_velocity(ts_df, n_impacts, csv_path, impact_info,
                                         good_impacts=good_impacts,
                                         safeguarded_peaks=safeguarded_peaks)
        else:
            print('      No time-series block found - skipping plots.')

    except Exception as e:
        print(f'    ERROR processing {fName}: {e}')
        traceback.print_exc()
        badFileList.append(fName)

    print()

# ______________________________________________________________________________
# Save outcomes
outcomes = pd.DataFrame(outcome_rows, columns=OUTCOME_HEADERS)

if outcomes.empty:
    print("WARNING: No outcome rows generated - check that CSV filenames follow "
          "'Tension_{Model}_{Condition}_{Date}-ir6.csv' and that a matching "
          "checklist file/sheet exists for each Model/Condition.")
elif save_on == 1:
    outcomes.to_csv(outfileName, index=False)
    print(f'STAR outcomes saved -> {outfileName}')
elif save_on == 2:
    outcomes.to_csv(outfileName, mode='a', header=False, index=False)
    print(f'STAR outcomes appended -> {outfileName}')

if badFileList:
    print(f'\nFiles with errors or no results: {badFileList}')