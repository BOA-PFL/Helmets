# -*- coding: utf-8 -*-
"""
Created 2026-09-18
@author: Bethany.Kilpatrick

Linear Impact Peak Acceleration Calculator + Plotter
=======================================================
Companion script to star_calculator_V9 / TactileHead_Script_V3, built for the
uni-axial LINEAR impact rig (Cadex "Impact Uni-Axial" exports — a single
Z-axis accelerometer channel per impact, e.g. "Acc Z1", "Acc Z2", ...).

Checklist workbook (ONE file, MULTIPLE sheets — one sheet per Helmet Model):

    2026_HelmetTesting2026_LinImpact_Tension_Mule_Checklist.xlsx
        Sheet "FS2"      -> checklist rows for Model FS2
        Sheet "FS2Mass"  -> Sample -> Mass for FS2 (ignored as a data sheet)
        Sheet "OEM"      -> checklist rows for Model OEM
        Sheet "OEMMass"  -> Sample -> Mass for OEM (ignored as a data sheet)

    Each checklist sheet's columns (header-based lookup, order doesn't
    matter, extra columns are ignored):
        ImpactOrder, Location, Helmet Model, Anvil, Tension, Torque,
        Sample, Speed, Completed, Notes

    "ImpactOrder" on the checklist maps 1:1 to the "Impact #" column in the
    matching Cadex CSV (and to the Acc Z{n} time-series column for that
    impact) — this is the join key, same role ImpactOrder played in
    star_calculator_V9, just without the extra per-condition sheet lookup
    since Tension (AVG/HIGH/LOW) is a plain column here, not a separate tab.

Cadex CSV exports (one file per Helmet Model):

    {MODEL}_Tension_LinImpact_..._-imu.csv   e.g. FS2_Tension_LinImpact_Sept2026-imu.csv
    {MODEL}_Tension_LinImp_..._-imu.csv      e.g. OEM_Tension_LinImp_Sept2026-imu.csv

    MODEL = the first underscore-separated token of the filename, upper-cased.
    That token is matched directly against a checklist sheet name.

For every impact:
    - PLA (peak linear acceleration) is read off the raw Acc Z{n} time-series
      (NOT the Cadex summary column) and run through the same glitch-reject
      safeguard used in star_calculator_V9 (valid_peak()) so a short,
      single-sample instrumentation spike can't be mistaken for the real
      impact peak.
    - The start and end of the impact pulse are found by walking outward
      from that peak on absolute G thresholds: start-of-impact is where the
      signal first rises above IMPACT_START_THRESH_G, end-of-impact is where
      it falls back down to IMPACT_END_THRESH_G (or below) on the way out.
      Impact duration = end time - start time.
    - A plot is saved per impact with the peak, the start-of-impact and the
      end-of-impact all marked and annotated.

All impacts across both CSVs are combined into ONE exported CSV
(outfileName below).

Layout mimics TactileHead_Script_V3.py: one outer "for fName in entries"
loop per CSV/model, with an inner "for impact_num" loop where every value
that goes into the output row is computed as its own plain variable (not
buried inside a results dict), then appended to all_outcomes; the combined
DataFrame is built once at the end from that list, same as TactileHead.
"""

import os
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from openpyxl import load_workbook

# ===========================================================================
# OPTIONS
# ===========================================================================

save_on            = 1   # 1 = write the combined LinImpact_outcomes.csv
make_plots         = 1   # 1 = save an annotated acceleration plot per impact
make_overlay_plots = 1   # 1 = save one Acc Z overlay plot per (Location, Speed) group
compare_helmets    = 1   # 1 = write LinImpact_helmet_comparison.csv (FS2 vs OEM, difference nulled when the impacts overlap)
debug              = 0  # 1 = verbose console output


# Folder containing the Cadex linear-impact CSV exports (one per Helmet Model)
fPath = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_LinImpact_Tension_Mule\CadexExports'

# The ONE checklist workbook — one sheet per Helmet Model (e.g. "FS2", "OEM"),
# plus one "{Model}Mass" sheet per model.
checklistPath = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_LinImpact_Tension_Mule\2026_HelmetTesting2026_LinImpact_Tension_Mule_Checklist.xlsx'

outPath = r'C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_LinImpact_Tension_Mule'

outfileName = os.path.join(outPath, 'LinImpact_outcomes.csv')

comparisonFileName = os.path.join(outPath, 'LinImpact_helmet_comparison.csv')

# ===========================================================================
# CONSTANTS
# ===========================================================================

# ── Cadex summary-column indices (same 6-Port export layout as star_calculator) ──
CADEX_COL_PLA      = 1    # Peak Acc.(G)  (summary column — fallback only)
CADEX_COL_VELOCITY = 2    # Velocity IN (m/sec)
CADEX_COL_POSITION = 11   # Position (e.g. "SIDE R")

COL_PLA      = "PLA (g)"
COL_SPEED    = "Velocity IN (m/s)"
COL_POSITION = "Position"

# ── Time-series channel template — linear uni-axial rig = one Z channel per impact ──
ACC_CHANNEL_TEMPLATE = "Acc Z{n}"

# ── Peak-safeguard tuning (glitch rejection — same idea as star_calculator) ──
# A real linear-impact pulse lasts several ms. Any excursion above
# PEAK_THRESH_FRAC * peak that is shorter than PEAK_MIN_DURATION_MS is
# treated as an instrumentation glitch and rejected when locating the peak.
PEAK_MIN_DURATION_MS = 1.0
PEAK_THRESH_FRAC     = 0.5

# ── Impact-window tuning (start/end-of-impact markers) ──
# Once the real peak is located, the impact pulse is bounded on ABSOLUTE G
# thresholds (not a fraction of the peak):
#   - start of impact = walking backward from the peak, the first point
#     where the signal rises above IMPACT_START_THRESH_G
#   - end of impact   = walking forward from the peak, the first point
#     where the signal falls back to IMPACT_END_THRESH_G (or below)
IMPACT_START_THRESH_G = 5.0
IMPACT_END_THRESH_G   = 0.0

# ── Overlay-plot tuning (Location x Speed group overlays — no window/peak marks) ──
# Each impact in a group runs over its own recorded time span, so before they
# can be overlaid on one axis every trace is linearly resampled onto a shared,
# evenly-spaced time grid at this resolution (same approach used by
# Impact_plotting_V2.py's overlay plots for the STAR rig).
OVERLAY_INTERP_DT_MS = 0.05

# ── HIC (Head Injury Criterion) maximum window lengths ──
# HIC15 / HIC36 cap how long a window t2 - t1 may be when searching for the
# most severe stretch of the pulse (see compute_hic()).
HIC15_WINDOW_MS = 15.0
HIC36_WINDOW_MS = 36.0

# ── Helmet comparison (see compare_helmets()) ──
# For each Location x Anvil x Tension cell, the range (min to max) of the
# impacts for one helmet is compared with the range for the other. If the two
# ranges overlap, there is no clear difference between the helmets for that
# metric and the difference is left blank.
COMPARE_HELMET_A = "FS2"
COMPARE_HELMET_B = "OEM"
COMPARE_METRICS  = ["Peak Accel (G)", "HIC15", "Time to Peak (ms)"]

# ── Display-name overrides for plot titles/filenames only ───────────────────
# The checklist's raw Location/Speed values are kept as-is in the exported
# CSV (LinImpact_outcomes.csv) — these only relabel what shows up on plots.
# Unrecognized values fall back to the raw checklist value unchanged.
LOCATION_DISPLAY_NAMES = {1: "R Side", 2: "Front"}
SPEED_DISPLAY_NAMES    = {"low": "Hemi", "high": "Flat"}

# ── Checklist sheet column names (header-based lookup) ──────────────────────
CHECKLIST_REQUIRED_COLS = ["ImpactOrder", "Location", "Helmet Model", "Sample", "Speed"]
CHECKLIST_OPTIONAL_COLS = ["Anvil", "Tension", "Torque", "Completed", "Notes"]

# ── Output column headers ───────────────────────────────────────────────────
OUTCOME_HEADERS = [
    "CSV File", "Helmet Model", "Impact Order", "Location", "Anvil",
    "Tension Condition", "Torque", "Sample", "Sample Mass (g)",
    "Measured Velocity IN (m/s)", "Impact Position (Cadex)",
    
    "Peak Accel (G)", "Peak Accel Time (ms)",
    "Naive Peak Accel (G)", "Naive Peak Time (ms)", "Glitch Excluded",
    "Impact Start Time (ms)", "Impact End Time (ms)", "Impact Duration (ms)",
    "Time to Peak (ms)", "HIC15", "HIC36",
]

# ===========================================================================
# PEAK SAFEGUARD  (glitch rejection for PLA, sourced from the raw time series)
# ===========================================================================
#
# Same logic as star_calculator_V9's valid_peak(): reject any candidate peak
# whose excursion above thresh_frac*peak_value doesn't last at least
# min_duration_ms (a real impact pulse lasts several ms; a glitch lasting a
# fraction of a ms is not a physical event). If the largest peak fails that
# test, mask it out and search again, so a genuine smaller peak underneath a
# glitch is still found.

def valid_peak(time_ms, signal, min_duration_ms=PEAK_MIN_DURATION_MS,
               thresh_frac=PEAK_THRESH_FRAC, max_iter=10):
    """
    Return (peak_value, peak_idx, peak_time_ms, excursion_duration_ms) for
    the largest peak in `signal` whose excursion above thresh_frac*peak_value
    lasts at least min_duration_ms. Narrower excursions are masked out as
    suspected instrumentation glitches and the search repeats on what's left.
    Returns (None, None, None, 0.0) if nothing qualifies within max_iter tries.
    """
    sig = np.asarray(signal, dtype=float).copy()
    t   = np.asarray(time_ms, dtype=float)

    for _ in range(max_iter):
        idx = int(np.nanargmax(sig)) if np.any(np.isfinite(sig)) else None
        if idx is None:
            return None, None, None, 0.0
        peak_val = sig[idx]
        if not np.isfinite(peak_val) or peak_val <= 0:
            return None, None, None, 0.0

        thresh = thresh_frac * peak_val
        above  = sig >= thresh

        i0 = idx
        while i0 > 0 and above[i0 - 1]:
            i0 -= 1
        i1 = idx
        while i1 < len(sig) - 1 and above[i1 + 1]:
            i1 += 1

        duration = t[i1] - t[i0]
        if duration >= min_duration_ms:
            return peak_val, idx, t[idx], duration

        # Too narrow to be a real impact pulse -> mask it out, try the next peak.
        sig[i0:i1 + 1] = -np.inf

    return None, None, None, 0.0


def impact_window(time_ms, signal, peak_idx,
                   start_thresh_g=IMPACT_START_THRESH_G,
                   end_thresh_g=IMPACT_END_THRESH_G):
    """
    Given a known peak sample index, bound the impact pulse using two
    separate absolute-G thresholds:
        - start of impact: walking backward from the peak, the point where
          the signal first rises above `start_thresh_g`.
        - end of impact: walking forward from the peak, the point where the
          signal falls back down to `end_thresh_g` (or below).

    Returns (start_time_ms, end_time_ms, duration_ms, i_start, i_end), or
    (None, None, None, None, None) if the peak index isn't usable.
    """
    t = np.asarray(time_ms, dtype=float)
    sig = np.asarray(signal, dtype=float)

    if peak_idx is None or not np.isfinite(sig[peak_idx]):
        return None, None, None, None, None

    i0 = peak_idx
    while i0 > 0 and sig[i0 - 1] > start_thresh_g:
        i0 -= 1
    i1 = peak_idx
    while i1 < len(sig) - 1 and sig[i1 + 1] > end_thresh_g:
        i1 += 1

    return t[i0], t[i1], t[i1] - t[i0], i0, i1


def compute_hic(time_ms, signal, max_window_ms):
    """
    Head Injury Criterion: the maximum over every window (t1, t2) no longer
    than `max_window_ms` of

        (t2 - t1) * [ (1 / (t2 - t1)) * integral of a(t) dt from t1 to t2 ] ^ 2.5

    with a(t) in G and t in seconds. The acceleration is integrated once with
    the trapezoid rule into a running total, so the area of any window is a
    subtraction of two running-total values, and then every start sample is
    stepped through with all of its allowed end samples checked at once.

    Note: this uses the raw Acc Z trace, not a CFC 1000 (SAE J211) filtered
    one, so values can differ slightly from a filtered HIC.

    Parameters
    ----------
    time_ms : numpy array
        time (ms), increasing
    signal : numpy array
        acceleration (G), same length as time_ms
    max_window_ms : float
        longest allowed window length (ms), e.g. 15 for HIC15, 36 for HIC36

    Returns
    -------
    hic : float
        the HIC value (unitless); 0.0 if no window has a positive mean
        acceleration

    """
    t_s = np.asarray(time_ms, dtype=float) / 1000
    sig = np.nan_to_num(np.asarray(signal, dtype=float))
    max_window_s = max_window_ms / 1000

    # Running trapezoid integral of acceleration: area[jj] = integral from t[0] to t[jj]
    dt = np.diff(t_s)
    trap = 0.5 * (sig[1:] + sig[:-1]) * dt
    area = np.concatenate(([0.0], np.cumsum(trap)))

    hic = 0.0
    for ii in range(len(t_s) - 1):
        # Last sample that still keeps the window within the allowed length
        jj_last = np.searchsorted(t_s, t_s[ii] + max_window_s, side="right") - 1
        if jj_last <= ii:
            continue

        durations = t_s[ii + 1:jj_last + 1] - t_s[ii]
        areas = area[ii + 1:jj_last + 1] - area[ii]
        mean_acc = areas / durations

        # A negative mean acceleration can't be raised to 2.5, and isn't an injury window
        mean_acc = np.clip(mean_acc, 0, None)
        window_hic = durations * mean_acc ** 2.5

        window_max = np.max(window_hic)
        if window_max > hic:
            hic = window_max

    return float(hic)


def compare_helmets_by_cell(outcomes_df, helmet_a, helmet_b, metrics):
    """
    Compare two helmets in every Location x Anvil x Tension cell, one row per
    cell and metric.

    Each helmet has two impacts per cell, so the spread of those impacts is
    summarized by their range (min to max). If the two helmets' ranges
    overlap, the impacts of one helmet are not separated from the other's and
    there is no clear difference, so the difference is left blank (None). If
    the ranges do not overlap, the difference in means is reported and the
    helmet with the higher mean is named. A helmet with fewer than 2 impacts
    in a cell can't show a spread, so that comparison is marked
    "Insufficient n" and not judged.

    Parameters
    ----------
    outcomes_df : dataframe
        one row per impact, with the OUTCOME_HEADERS columns
    helmet_a : str
        first Helmet Model to compare (e.g. "FS2")
    helmet_b : str
        second Helmet Model to compare (e.g. "OEM")
    metrics : list
        outcome columns to compare (e.g. "Peak Accel (G)", "HIC15")

    Returns
    -------
    comparison : dataframe
        one row per cell and metric: the means and ranges for each helmet,
        whether the ranges overlap, the result, and the difference
        (helmet_a - helmet_b, blank when the ranges overlap)

    """
    cell_cols = ["Location", "Anvil", "Tension Condition"]

    headers = ["Location", "Location Name", "Anvil", "Tension Condition", "Metric"]
    for helmet in [helmet_a, helmet_b]:
        headers += [f"{helmet} n", f"{helmet} Mean", f"{helmet} Min", f"{helmet} Max"]
    headers += ["Ranges Overlap", "Result", f"Difference ({helmet_a} - {helmet_b})"]

    rows = []
    for cell_key, cell_df in outcomes_df.groupby(cell_cols, dropna=False):
        location, anvil, tension = cell_key

        rows_a = cell_df[cell_df["Helmet Model"] == helmet_a]
        rows_b = cell_df[cell_df["Helmet Model"] == helmet_b]
        if rows_a.empty or rows_b.empty:
            continue

        for metric in metrics:
            vals_a = rows_a[metric].dropna()
            vals_b = rows_b[metric].dropna()
            if vals_a.empty or vals_b.empty:
                continue

            n_a = len(vals_a); mean_a = vals_a.mean(); min_a = vals_a.min(); max_a = vals_a.max()
            n_b = len(vals_b); mean_b = vals_b.mean(); min_b = vals_b.min(); max_b = vals_b.max()

            if n_a < 2 or n_b < 2:
                overlap = None
                result = "Insufficient n"
                difference = None
            else:
                overlap = bool(min_a <= max_b and min_b <= max_a)
                if overlap:
                    # Spreads overlap -> can't tell the helmets apart, so null the difference
                    result = "No clear difference"
                    difference = None
                else:
                    difference = mean_a - mean_b
                    result = f"{helmet_a} higher" if difference > 0 else f"{helmet_b} higher"

            rows.append([
                location, display_location(location), anvil, tension, metric,
                n_a, round(mean_a, 3), round(min_a, 3), round(max_a, 3),
                n_b, round(mean_b, 3), round(min_b, 3), round(max_b, 3),
                overlap, result,
                round(difference, 3) if difference is not None else None,
            ])

    return pd.DataFrame(rows, columns=headers)

# ===========================================================================
# CHECKLIST DISCOVERY  (sheets within the ONE workbook, one sheet per Model)
# ===========================================================================

def sheet_header_index(ws):
    """Return {column_name: index} for whichever recognized columns exist in row 1."""
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
    header = [str(h).strip() if h is not None else "" for h in header_row]
    idx = {}
    for name in CHECKLIST_REQUIRED_COLS + CHECKLIST_OPTIONAL_COLS:
        if name in header:
            idx[name] = header.index(name)
    return idx


def discover_checklist_sheets(wb):
    """
    Map MODEL token -> sheet name for every sheet in the workbook that looks
    like a checklist/data sheet (has an "ImpactOrder" column). "{Model}Mass"
    sheets and any other reference tabs are skipped automatically because
    they won't have that column.
    """
    sheets = {}
    for sheet_name in wb.sheetnames:
        if sheet_name.strip().lower().endswith("mass"):
            continue
        ws = wb[sheet_name]
        idx = sheet_header_index(ws)
        if "ImpactOrder" not in idx or "Location" not in idx:
            continue
        sheets[sheet_name.strip().upper()] = sheet_name
    return sheets


def find_mass_sheet(wb, model_token):
    """Find the '{Model}Mass' sheet for a given model token, if present."""
    target = f"{model_token.strip().lower()}mass"
    for s in wb.sheetnames:
        if s.strip().lower() == target:
            return s
    for s in wb.sheetnames:
        if s.strip().lower().endswith("mass") and s.strip().lower().startswith(model_token.strip().lower()):
            return s
    return None


def load_checklist_from_sheet(ws, idx):
    """Read one Model's checklist sheet -> DataFrame with valid ImpactOrder rows only."""
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        impact_order = row[idx["ImpactOrder"]]
        try:
            impact_num = int(impact_order)
            if impact_num <= 0:
                continue
        except (TypeError, ValueError):
            continue

        rows.append({
            "ImpactOrder":   impact_num,
            "Location":      row[idx["Location"]],
            "Helmet Model":  row[idx["Helmet Model"]],
            "Anvil":         row[idx["Anvil"]]   if "Anvil"   in idx else None,
            "Tension":       row[idx["Tension"]] if "Tension" in idx else None,
            "Torque":        row[idx["Torque"]]  if "Torque"  in idx else None,
            "Sample":        row[idx["Sample"]],
            "Speed":         row[idx["Speed"]],
            "Completed":     row[idx["Completed"]] if "Completed" in idx else None,
            "Notes":         row[idx["Notes"]]     if "Notes"     in idx else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        dupes = df[df.duplicated("ImpactOrder", keep=False)]
        if not dupes.empty:
            print(f"    WARNING: duplicate ImpactOrder values in checklist sheet:\n{dupes}")
    return df


def load_mass(wb, sheet_name):
    """Read a 'Sample -> Mass' sheet via header lookup. Returns {} if absent/unusable."""
    if sheet_name is None or sheet_name not in wb.sheetnames:
        return {}
    ws = wb[sheet_name]
    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
    header = [str(h).strip() if h is not None else "" for h in header_row]
    if "Sample" not in header or "Mass" not in header:
        return {}
    s_idx, m_idx = header.index("Sample"), header.index("Mass")
    mass_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sample, mass = row[s_idx], row[m_idx]
        if sample is not None and mass is not None:
            mass_map[str(sample).strip()] = mass
    return mass_map

# ===========================================================================
# CADEX CSV LOADERS
# ===========================================================================

def read_lines(path):
    """Read file with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}.")


def load_summary(lines, fname):
    """Extract impact summary rows from Cadex CSV -> 1-based indexed DataFrame."""
    # Find the first row whose Impact # column reads "1" -> start of the data block
    data_start = None
    for ii, line in enumerate(lines):
        first_field = line.split(",")[0].strip()
        if first_field == "1":
            data_start = ii
            break
    if data_start is None:
        raise ValueError(f"No data rows found in {fname}.")

    # Keep only rows whose Impact # column is a positive integer
    valid = []
    for line in lines[data_start:]:
        first_field = line.split(",")[0].strip()
        if first_field.isdigit() and int(first_field) > 0:
            valid.append(line)
    if not valid:
        raise ValueError(f"No valid impact rows in {fname}.")
    raw = pd.read_csv(StringIO("".join(valid)), header=None, on_bad_lines="skip")
    max_needed = max(CADEX_COL_PLA, CADEX_COL_VELOCITY, CADEX_COL_POSITION)
    if raw.shape[1] <= max_needed:
        raise ValueError(f"{fname} has too few columns for a Cadex export.")
    df = raw[[CADEX_COL_PLA, CADEX_COL_VELOCITY, CADEX_COL_POSITION]].copy()
    df.columns = [COL_PLA, COL_SPEED, COL_POSITION]
    for col in [COL_PLA, COL_SPEED]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = range(1, len(df) + 1)   # 1-based so ImpactOrder maps directly
    return df


def load_timeseries(lines):
    """Extract the time-series waveform block (Time (s), Acc Z1, Acc Z2, ...) from a Cadex CSV."""
    ts_start = None
    for ii, line in enumerate(lines):
        if line.startswith("Time (s)"):
            ts_start = ii
            break
    if ts_start is None:
        return None
    col_names  = [c.strip() for c in lines[ts_start].rstrip("\n").split(",")]

    # Cadex pads the header row with empty cells (trailing commas); pandas
    # rejects duplicate names, so give each blank a unique placeholder.
    for ii, name in enumerate(col_names):
        if name == "":
            col_names[ii] = f"Unnamed_{ii}"

    data_lines = []
    for line in lines[ts_start + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("end"):
            break
        try:
            float(stripped.split(",")[0].strip())
            data_lines.append(stripped)
        except ValueError:
            break
    if not data_lines:
        return None
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None, names=col_names,
        skipinitialspace=True, on_bad_lines="skip",
    )
    return df.apply(pd.to_numeric, errors="coerce")

# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_impact_trace(time_ms, signal, peak_val, peak_idx, start_idx, end_idx,
                       title, out_path):
    """
    Save one PNG: the Acc Z time-series for a single impact, with the
    safeguarded peak marked, and the start/end of the impact pulse (per
    impact_window()) marked with separate vertical lines + a shaded span.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax.plot(time_ms, signal, color="#2980b9", linewidth=1.4, label="Acc Z")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    if peak_val is not None and peak_idx is not None:
        ax.plot(time_ms[peak_idx], peak_val, marker="x", markersize=14,
                 markeredgecolor="black", markerfacecolor="red", zorder=5,
                 linestyle="None", label=f"Peak: {peak_val:.1f} G")
        ax.annotate(f"{peak_val:.1f} G", (time_ms[peak_idx], peak_val),
                    textcoords="offset points", xytext=(6, 8), fontsize=8,
                    fontweight="bold")

    if start_idx is not None and end_idx is not None:
        ax.axvline(time_ms[start_idx], color="green", linestyle="--", linewidth=1.2,
                   label=f"Impact Start: {time_ms[start_idx]:.2f} ms")
        ax.axvline(time_ms[end_idx], color="darkorange", linestyle="--", linewidth=1.2,
                   label=f"Impact End: {time_ms[end_idx]:.2f} ms")
        ax.axvspan(time_ms[start_idx], time_ms[end_idx], color="gold", alpha=0.10)

    ax.set_ylabel("Acceleration (G)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interp_to_grid(time_ms, signal, grid_ms):
    """
    Linearly resample one impact's Acc Z trace onto a shared time grid so
    traces of slightly different length/sample spacing can be overlaid on
    one axis. Points outside this impact's own recorded time range come
    back as NaN (matplotlib skips NaN rather than extrapolating a flat
    line past where the impact actually ended).

    Parameters
    ----------
    time_ms : numpy array
        this impact's own recorded time (ms)
    signal : numpy array
        this impact's Acc Z signal (G), same length as time_ms
    grid_ms : numpy array
        shared evenly-spaced time grid (ms) to resample onto

    Returns
    -------
    signal_grid : numpy array
        `signal` resampled onto `grid_ms`, NaN outside
        [min(time_ms), max(time_ms)]

    """
    t   = np.asarray(time_ms, dtype=float)
    sig = np.asarray(signal, dtype=float)

    finite = np.isfinite(t) & np.isfinite(sig)
    if finite.sum() < 2:
        return np.full_like(grid_ms, np.nan, dtype=float)

    t = t[finite]; sig = sig[finite]
    order = np.argsort(t)
    t = t[order]; sig = sig[order]

    return np.interp(grid_ms, t, sig, left=np.nan, right=np.nan)


def display_location(location):
    """
    Map a checklist Location value to its human-readable plot name (e.g.
    1 -> "R Side"). Falls back to the raw value for any location not in
    LOCATION_DISPLAY_NAMES.

    Parameters
    ----------
    location : any
        raw checklist Location value

    Returns
    -------
    name : str
        human-readable location name for plot titles/filenames

    """
    return LOCATION_DISPLAY_NAMES.get(location, str(location))


def display_speed(speed_label):
    """
    Map a checklist Speed label to its human-readable plot name (e.g.
    "Low" -> "Hemi"), matched case-insensitively. Falls back to the raw
    value for any label not in SPEED_DISPLAY_NAMES.

    Parameters
    ----------
    speed_label : any
        raw checklist Speed value

    Returns
    -------
    name : str
        human-readable speed name for plot titles/filenames

    """
    key = str(speed_label).strip().lower()
    return SPEED_DISPLAY_NAMES.get(key, str(speed_label))


def sanitize_filename_part(value):
    """
    Turn a checklist value (Location, Speed label, ...) into something safe
    to use inside a filename. Speed labels in particular can contain "/"
    (e.g. "6.2 m/s"), which Windows reads as a path separator and silently
    turns one filename into a nested, non-existent subfolder — this strips
    that class of character instead of letting it reach os.path.join().

    Parameters
    ----------
    value : any
        the checklist value to sanitize (converted to str first)

    Returns
    -------
    safe_value : str
        `value` with path-separator and reserved filename characters
        replaced, and spaces removed; "Unknown" if that leaves nothing

    """
    text = str(value).strip()
    for ch in '\\/:*?"<>|':
        text = text.replace(ch, "-")
    text = text.replace(" ", "")
    return text or "Unknown"


def plot_overlay_group(curves, model_token, location, speed_label, out_dir):
    """
    Save one PNG overlaying every impact's Acc Z trace for one
    (Location, Speed) group on a single time axis, one color per impact.

    Unlike plot_impact_trace(), no start/end-of-impact window or peak
    marker is drawn here — this plot is for comparing overall pulse shape
    and spread across impacts at the same Location/Speed, not for reading
    off any single impact's peak or duration.

    Parameters
    ----------
    curves : list of dict
        one entry per impact in this group, each a dict with keys
        "time_ms" (numpy array), "signal" (numpy array), "label" (str)
    model_token : str
        Helmet Model token (e.g. "FS2"), used in the title and filename
    location : any
        checklist Location value shared by every impact in `curves`
    speed_label : any
        checklist Speed label shared by every impact in `curves`
    out_dir : str
        folder the PNG is saved into (created if it doesn't exist)

    Returns
    -------
    None

    """
    if not curves:
        return

    # Build one shared time grid spanning every impact's own recorded window
    t_min = min(np.nanmin(c["time_ms"]) for c in curves)
    t_max = max(np.nanmax(c["time_ms"]) for c in curves)
    grid_ms = np.arange(t_min, t_max + OVERLAY_INTERP_DT_MS, OVERLAY_INTERP_DT_MS)

    n_curves = len(curves)
    if n_curves > 1:
        color_list = cm.turbo(np.linspace(0.05, 0.95, n_curves))
    else:
        color_list = ["#2980b9"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for curve, color in zip(curves, color_list):
        signal_grid = interp_to_grid(curve["time_ms"], curve["signal"], grid_ms)
        ax.plot(grid_ms, signal_grid, color=color, linewidth=1.3, alpha=0.9,
                label=curve["label"])

    loc_display   = display_location(location)
    speed_display = display_speed(speed_label)

    title = (f"{model_token}  —  {loc_display}  —  {speed_display}  —  "
             f"Acc Z (impact overlay, n={n_curves})")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Acceleration (G)")
    ax.set_xlabel("Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=1 if n_curves <= 15 else 2, framealpha=0.8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_location = sanitize_filename_part(loc_display)
    safe_speed    = sanitize_filename_part(speed_display)
    out_path = os.path.join(out_dir, f"{model_token}_{safe_location}_{safe_speed}_AccZ.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Overlay plot saved -> {out_path}  ({n_curves} impact(s))")

# ===========================================================================
# MAIN
# ===========================================================================

# ── Load the checklist workbook once and discover its Model sheets ─────────

wb = load_workbook(checklistPath, data_only=True)
checklist_sheets = discover_checklist_sheets(wb)
if not checklist_sheets:
    print(f"  ERROR: No checklist sheets found in: {checklistPath}")
else:
    print(f"  Checklist sheets found -> {checklist_sheets}\n")

# ── Find CSVs ────────────────────────────────────────────────────────────────

entries = sorted([f for f in os.listdir(fPath) if f.lower().endswith(".csv")])
if not entries:
    print(f"No CSV files found in: {fPath}")
else:
    print(f"Found {len(entries)} CSV file(s) in {fPath}\n")

all_outcomes   = []
badFileList    = []
overlay_groups = {}   # (model_token, location, speed_label) -> list of curve dicts

# ── Main loop: one iteration per CSV/Model ───────────────────────────────────

for fName in entries:
    try:
        csv_path = os.path.join(fPath, fName)
        stem     = os.path.splitext(fName)[0]

        # Expect: {MODEL}_Tension_..._-imu
        parts = stem.split("_")
        model_token = parts[0].upper()

        sheet_name = checklist_sheets.get(model_token)
        if sheet_name is None:
            print(f"    NOTE: no checklist sheet found for model '{model_token}' "
                  f"(from '{fName}'). Available sheets: {list(checklist_sheets.keys())}")
            print()
            continue

        ws           = wb[sheet_name]
        idx          = sheet_header_index(ws)
        checklist_df = load_checklist_from_sheet(ws, idx)

        mass_sheet_name = find_mass_sheet(wb, model_token)
        mass_map        = load_mass(wb, mass_sheet_name)

        if checklist_df.empty:
            print(f"    ERROR: no valid ImpactOrder rows in sheet '{sheet_name}'.")
            badFileList.append(fName)
            print()
            continue

        print(f"    {fName}")
        print(f"      -> checklist sheet: '{sheet_name}'  "
              f"(mass sheet: {mass_sheet_name or 'none found'})")

        lines      = read_lines(csv_path)
        summary_df = load_summary(lines, fName)
        ts_df      = load_timeseries(lines)

        if ts_df is None or "Time (s)" not in ts_df.columns:
            print(f"      No time-series block found — skipping (no peak/plot data).")
            badFileList.append(fName)
            print()
            continue

        # Count how many "Acc Z{n}" impact channels are present in this CSV
        n_impacts = 0
        for ss in range(1, 60):
            acc_col_check = ACC_CHANNEL_TEMPLATE.format(n=ss)
            if acc_col_check in ts_df.columns:
                n_impacts += 1
        print(f"      Time-series: {len(ts_df)} samples, {n_impacts} impact(s)")

        time_ms  = (ts_df["Time (s)"] * 1000).values
        plot_dir = os.path.join(os.path.dirname(csv_path), "Acceleration Plots", model_token)
        if make_plots:
            os.makedirs(plot_dir, exist_ok=True)

        n_skipped_no_checklist = 0
        n_skipped_no_peak      = 0

        # ── Inner loop: one iteration per impact, variables computed in-line ──
        for impact_num in range(1, n_impacts + 1):

            checklist_match = checklist_df[checklist_df["ImpactOrder"] == impact_num]
            if checklist_match.empty:
                n_skipped_no_checklist += 1
                continue
            crow = checklist_match.iloc[0]

            location          = crow["Location"]
            helmet_model      = crow["Helmet Model"]
            anvil             = crow["Anvil"]
            tension_condition = crow["Tension"]
            torque            = crow["Torque"]
            sample            = crow["Sample"]
            
            sample_mass_g     = mass_map.get(str(sample).strip()) if sample is not None else None

            if impact_num not in summary_df.index:
                measured_velocity_in = None
                impact_position       = None
            else:
                srow = summary_df.loc[impact_num]
                measured_velocity_in = srow[COL_SPEED]
                impact_position       = srow[COL_POSITION]

            acc_col = ACC_CHANNEL_TEMPLATE.format(n=impact_num)
            if acc_col not in ts_df.columns or ts_df[acc_col].isna().all():
                n_skipped_no_peak += 1
                continue
            signal = ts_df[acc_col].values

            naive_idx  = int(np.nanargmax(signal))
            naive_peak = float(signal[naive_idx])
            naive_time = float(time_ms[naive_idx])

            peak_val, peak_idx, peak_time_ms, _glitch_dur = valid_peak(time_ms, signal)
            if peak_val is None:
                n_skipped_no_peak += 1
                continue
            glitch_excluded = abs(naive_peak - peak_val) > 1e-6

            start_time_ms, end_time_ms, duration_ms, start_idx, end_idx = \
                impact_window(time_ms, signal, peak_idx)

            # Time to peak: onset of impact -> peak G, i.e. the rise time of
            # the pulse. Distinct from "Peak Accel Time (ms)", which is
            # measured from the start of the whole recording, not the impact.
            time_to_peak_ms = peak_time_ms - start_time_ms if start_time_ms is not None else None

            hic15 = compute_hic(time_ms, signal, HIC15_WINDOW_MS)
            hic36 = compute_hic(time_ms, signal, HIC36_WINDOW_MS)

            if debug and glitch_excluded:
                print(f"      Impact #{impact_num}: glitch excluded — "
                      f"naive peak {naive_peak:.2f}G at t={naive_time:.2f}ms "
                      f"-> using {peak_val:.2f}G at t={peak_time_ms:.2f}ms")

            all_outcomes.append([
                fName, helmet_model, impact_num, location, anvil,
                tension_condition, torque, sample, sample_mass_g,
                 measured_velocity_in, impact_position,
                
                round(peak_val, 3), round(peak_time_ms, 4),
                round(naive_peak, 3), round(naive_time, 4), glitch_excluded,
                round(start_time_ms, 4) if start_time_ms is not None else None,
                round(end_time_ms, 4)   if end_time_ms   is not None else None,
                round(duration_ms, 4)   if duration_ms   is not None else None,
                round(time_to_peak_ms, 4) if time_to_peak_ms is not None else None,
                round(hic15, 2), round(hic36, 2),
            ])

            if make_plots:
                title = (f"Impact #{impact_num}  —  {model_token} {tension_condition}  |  "
                         f"{display_location(location)}  |  Sample {sample}  |  "
                         f"{display_speed(anvil)}")
                out_path = os.path.join(plot_dir, f"{model_token}_impact{impact_num:02d}_{sample}.png")
                plot_impact_trace(time_ms, signal, peak_val, peak_idx,
                                   start_idx, end_idx, title, out_path)

            if make_overlay_plots:
                overlay_key   = (model_token, location, anvil)
                overlay_label = f"{tension_condition} #{impact_num} ({sample})"
                overlay_groups.setdefault(overlay_key, []).append({
                    "time_ms": time_ms,
                    "signal":  signal,
                    "label":   overlay_label,
                })

        if n_skipped_no_checklist and debug:
            print(f"      {n_skipped_no_checklist} impact(s) in the CSV had no matching "
                  f"checklist ImpactOrder — skipped.")
        if n_skipped_no_peak and debug:
            print(f"      {n_skipped_no_peak} impact(s) had no usable time-series peak — skipped.")

    except Exception as e:
        print(f"    ERROR processing {fName}: {e}")
        import traceback; traceback.print_exc()
        badFileList.append(fName)

    print()

# ── Overlay plots: one PNG per (Model, Location, Speed) group ───────────────

if make_overlay_plots:
    print(f"Building overlay plots for {len(overlay_groups)} "
          f"(Model, Location, Speed) group(s)...")
    for (model_token, location, speed_label), curves in overlay_groups.items():
        overlay_dir = os.path.join(fPath, "Overlay Plots", model_token)
        plot_overlay_group(curves, model_token, location, speed_label, overlay_dir)
    print()

# ── Build combined DataFrame + save ──────────────────────────────────────────

outcomes = pd.DataFrame(all_outcomes, columns=OUTCOME_HEADERS)

if save_on == 1:
    if not outcomes.empty:
        if not os.path.exists(outfileName):
            outcomes.to_csv(outfileName, header=True, index=False)
        else:
            outcomes.to_csv(outfileName, mode='a', header=False, index=False)
        print(f"LinImpact outcomes saved -> {outfileName}  ({len(outcomes)} impact rows)")
    else:
        print("WARNING: No outcome rows generated — check that CSV filenames start with "
              "a Model token matching a checklist sheet name, and that ImpactOrder values "
              "on that sheet line up with impacts present in the CSV.")

# ── Helmet comparison: FS2 vs OEM, difference nulled when the ranges overlap ──
# Uses the impacts from this run only (not rows already in the outcomes CSV).

if compare_helmets and not outcomes.empty:
    comparison = compare_helmets_by_cell(outcomes, COMPARE_HELMET_A, COMPARE_HELMET_B,
                                         COMPARE_METRICS)
    n_clear = int((comparison["Result"].str.contains("higher")).sum())
    print(f"Helmet comparison: {len(comparison)} cell/metric comparisons, "
          f"{n_clear} with a clear difference (ranges do not overlap).")

    if not os.path.exists(comparisonFileName):
        comparison.to_csv(comparisonFileName, header=True, index=False)
        print(f"Helmet comparison saved -> {comparisonFileName}")
    else:
        print(f"Helmet comparison NOT saved: {comparisonFileName} already exists "
              f"(move or rename it to write a new one).")

if badFileList:
    print(f"\nFiles with errors or no results: {badFileList}")
