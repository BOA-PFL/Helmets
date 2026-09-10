# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:13:51 2026

@author: Minori.Iizuka
"""

#%%

"""
--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
    pip install cadquery-ocp trimesh pandas numpy scipy matplotlib plotly --break-system-packages
"""

import numpy as np
import pandas as pd
import trimesh
from scipy.interpolate import RBFInterpolator
import matplotlib
import matplotlib.pyplot as plt
original_backend = matplotlib.get_backend()
matplotlib.use("Agg")
import plotly.express as px
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io
import matplotlib
import os

# Load TactileHead Pressure Data
pressure_data_folder = r"C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Testing Segments\Helmets\2026_Performance_GiantPressure_Giant\CSV"
file_number = 0 #Trial specific index
entries = [fName for fName in os.listdir(pressure_data_folder)
           if fName.endswith('.csv') and '0_' not in fName]
filePath2D = r"C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Helmets\TactileHead\3D_Model\sensel_display_positions.csv"
dat2D = pd.read_csv(filePath2D)
model_3D_CSV = r"C:\Users\bethany.kilpatrick\BOA Technology Inc\PFL Team - General\Helmets\TactileHead\3D_Model\full_head_model_calibrated.csv"
model_3D = pd.read_csv(model_3D_CSV)

save_gif = 0
save_obj = 0
save_image = 1 
# %%

N_FRAMES = 60          # more = smoother rotation but a bigger file
ELEV = 120              # camera elevation angle (degrees)
DURATION_MS = 100       # ms per frame -- lower = faster spin
COLOR_COL = "y"        # column in RESULT_CSV to color points by (e.g. "y" = row index)
magnification = 1.0

# Which world axis to spin the points around (rotates the DATA, camera stays
# fixed -- matplotlib's view_init only ever orbits the camera around Z, so
# this is needed for X or Y rotation):
ROTATION_AXIS = "y"    # "x", "y", or "z"

AXIS_LINE_LENGTH = 1.15   # how far the drawn X/Y/Z axis lines extend, as a
                          # multiple of the point cloud's own radius


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"axis must be 'x','y', or 'z', got {axis!r}")


from typing import Union, Optional
def make_rotating_gif(model_3D: pd.DataFrame, out_path: str,
                       n_frames: int = 60, elev: float = 15, duration_ms: int = 60,
                       color_col: Optional[Union[str, pd.Series]] = "y",
                       rotation_axis: str = "y", axis_line_length: float = 1.15,
                       zoom: float = 1.5, vmin: float = None, vmax: float = None, #default is 1.5, smaller numbers will look more condensed. Start with 1 and go down from there
                       show_colorbar: bool = True,
                       figsize=(7, 7), dpi: int = 100):
    frames = []
    pts0 = model_3D[["x_3d", "y_3d", "z_3d"]].values.copy()
    center = (pts0.max(axis=0) + pts0.min(axis=0)) / 2
    radius = np.linalg.norm(pts0.max(axis=0) - pts0.min(axis=0)) / 2 * 1.05
    radius = radius / zoom   # zoom > 1 = tighter view = object appears bigger

    # resolve color_col the same way as before
    if color_col is None:
        colors = "black"
    elif isinstance(color_col, str):
        colors = model_3D[color_col] if color_col in model_3D.columns else "black"
    else:
        colors = color_col

    # fix the color scale ONCE from the real data range (or your own vmin/vmax),
    # so it's explicit and consistent rather than left to matplotlib's defaults
    if not isinstance(colors, str):
        vmin_ = np.nanmin(colors) if vmin is None else vmin
        vmax_ = np.nanmax(colors) if vmax is None else vmax
    else:
        vmin_ = vmax_ = None

    for i in range(n_frames):
        angle = 360 * i / n_frames
        R = rotation_matrix(rotation_axis, angle)
        pts = (pts0 - center) @ R.T + center

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d") 
        ax.set_facecolor('Black')
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=colors, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
        if show_colorbar and not isinstance(colors, str):
            fig.colorbar(sc, ax=ax, shrink=0.6,
                         label=color_col if isinstance(color_col, str) else "psi")

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect([1, 1, 1]) 
        ax.view_init(elev=elev, azim=-90)
        ax.grid(False) 
        ## Background script
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"Wrote {len(frames)}-frame GIF to {out_path}")
    

def visible_half(pts0: np.ndarray, center: np.ndarray, colors,
                 view: str, hide_far_half: bool = True,
                 top_cut_y: float = 75.0):
    """
    Select the sensels on the half of the shell that faces a view's camera,
    along with their colors.

    The head model is a hollow shell, so a 2D projection draws two surfaces on
    top of each other and the far one bleeds through the gaps in the near one
    -- a posterior hot spot then reads as an anterior one. Keeping only the
    near half fixes that.

    Which half is near was established by rendering two probe markers
    straddling the shell center and reading back from the raster which one
    survived, rather than from matplotlib's view_init documentation (note that
    all five views pass azim=-90, but elev=90 is matplotlib's degenerate pole,
    so azim does nothing for the four side views -- the rotation applied to the
    DATA is what selects the view). In the ORIGINAL unrotated model
    coordinates the camera-facing half is:

        front  +z          back   -z
        left   +x          right  -x
        top    +y

    where x is medial-lateral, y is superior-inferior and z is
    anterior-posterior. +z is anterior: the cap ends at the brow on the +z
    side and continues down the occiput to y = -56 on the -z side.

    The four side views cut at the bounding-box center, which is the correct
    silhouette for a roughly ellipsoidal shell. The top view does NOT, because
    the crown is nearly single-valued when seen from above -- its pile-up comes
    from the near-vertical side walls rather than from a far surface -- so it
    takes an explicit height instead.

    Note: top_cut_y is a coordinate in the model's own frame and is therefore
    specific to full_head_model_calibrated.csv. Revisit it if the head model is
    ever recalibrated.

    Parameters
    ----------
    pts0 : numpy array (Nx3)
        sensel positions, unrotated (mm)
    center : numpy array (3x1)
        bounding-box center of the full point cloud (mm)
    colors : pd.Series, numpy array (Nx1), or str
        per-sensel color values, or a single matplotlib color name
    view : str
        'front', 'back', 'left', 'right', or 'top'
    hide_far_half : bool
        False returns everything unchanged, for comparison against the old
        behaviour
    top_cut_y : float
        height above which the top view keeps sensels (mm)

    Returns
    -------
    [pts_vis, colors_vis] : list
        pts_vis : numpy array (Mx3) positions that face this view's camera
        colors_vis : numpy array (Mx1) or str, their matching colors

    """
    if not hide_far_half:
        return [pts0, colors]

    if view == "front":
        mask = pts0[:, 2] > center[2]
    elif view == "back":
        mask = pts0[:, 2] < center[2]
    elif view == "left":
        mask = pts0[:, 0] > center[0]
    elif view == "right":
        mask = pts0[:, 0] < center[0]
    elif view == "top":
        mask = pts0[:, 1] > top_cut_y
    else:
        raise ValueError(f"unknown view '{view}'")

    pts_vis = pts0[mask]

    # pts0 and colors are matched row-by-row, so BOTH have to be indexed with
    # the same mask. Masking one and not the other is the usual way this goes
    # wrong -- matplotlib raises a length mismatch on the c= argument. A
    # constant string color has nothing to index.
    if isinstance(colors, str):
        colors_vis = colors
    else:
        colors_arr = np.asarray(colors, dtype=float)
        colors_vis = colors_arr[mask]

    return [pts_vis, colors_vis]


def head_3D_png(model_3D: pd.DataFrame, top_out_path: str,
                       right_out_path: str, left_out_path: str,
                       back_out_path: str,front_out_path: str,
                       elev: float = 0,
                       color_col: Optional[Union[str, pd.Series]] = "y",
                       rotation_axis: str = "y", axis_line_length: float = 1.15,
                       zoom: float = 1.5, vmin: float = None, vmax: float = None, # default is 1.5 
                       hide_far_half: bool = True, top_cut_y: float = 75.0,
                       show_colorbar: bool = True,
                       figsize=(7, 7), dpi: int = 100):
    
    pts0 = model_3D[["x_3d", "y_3d", "z_3d"]].values.copy()
    center = (pts0.max(axis=0) + pts0.min(axis=0)) / 2
    radius = np.linalg.norm(pts0.max(axis=0) - pts0.min(axis=0)) / 2 * 1.05
    radius = radius / zoom   # zoom > 1 = tighter view = object appears bigger

    # resolve color_col the same way as before
    if color_col is None:
        colors = "black"
    elif isinstance(color_col, str):
        colors = model_3D[color_col] if color_col in model_3D.columns else "black"
    else:
        colors = color_col

    # fix the color scale ONCE from the real data range (or your own vmin/vmax),
    # so it's explicit and consistent rather than left to matplotlib's defaults
    if not isinstance(colors, str):
        vmin_ = np.nanmin(colors) if vmin is None else vmin
        vmax_ = np.nanmax(colors) if vmax is None else vmax
    else:
        vmin_ = vmax_ = None

    R_front = rotation_matrix(rotation_axis, 0)
    R_top = rotation_matrix(rotation_axis, 0)
    R_back = rotation_matrix(rotation_axis, 180)
    R_right = rotation_matrix(rotation_axis, 90)
    R_left = rotation_matrix(rotation_axis, 270)

    ### BACK ###
    [pts_vis, colors_back] = visible_half(pts0, center, colors, "back",
                                          hide_far_half, top_cut_y)
    pts_back = (pts_vis - center) @ R_back.T + center

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_back[:, 0], pts_back[:, 1], pts_back[:, 2],
                     c=colors_back, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
    if show_colorbar and not isinstance(colors, str):
        fig.colorbar(sc, ax=ax, shrink=0.6,
                     label=color_col if isinstance(color_col, str) else "psi")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=-90) 
    # ax.set_facecolor('#000000')
    ax.grid(False) 
    ax.axis('off')
    fig.savefig(back_out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    ### Front ###
    [pts_vis, colors_front] = visible_half(pts0, center, colors, "front",
                                          hide_far_half, top_cut_y)
    pts_front = (pts_vis - center) @ R_front.T + center
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_front[:, 0], pts_front[:, 1], pts_front[:, 2],
                     c=colors_front, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
    if show_colorbar and not isinstance(colors, str):
        fig.colorbar(sc, ax=ax, shrink=0.6,
                     label=color_col if isinstance(color_col, str) else "psi")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=-90)
    # ax.set_facecolor('#000000')
    ax.grid(False)
    ax.axis('off')

    fig.savefig(front_out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Left
    [pts_vis, colors_left] = visible_half(pts0, center, colors, "left",
                                          hide_far_half, top_cut_y)
    pts_left = (pts_vis - center) @ R_left.T + center

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_left[:, 0], pts_left[:, 1], pts_left[:, 2],
                     c=colors_left, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
    if show_colorbar and not isinstance(colors, str):
        fig.colorbar(sc, ax=ax, shrink=0.6,
                     label=color_col if isinstance(color_col, str) else "psi")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=-90)
    ax.grid(False)
    # ax.set_facecolor('#000000')
    ax.axis('off')

    fig.savefig(left_out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Right
    [pts_vis, colors_right] = visible_half(pts0, center, colors, "right",
                                          hide_far_half, top_cut_y)
    pts_right = (pts_vis - center) @ R_right.T + center

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_right[:, 0], pts_right[:, 1], pts_right[:, 2],
                     c=colors_right, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
    if show_colorbar and not isinstance(colors, str):
        fig.colorbar(sc, ax=ax, shrink=0.6,
                     label=color_col if isinstance(color_col, str) else "psi")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=-90)
    ax.grid(False)
    # ax.set_facecolor('#000000')
    ax.axis('off')

    # buf = io.BytesIO()
    fig.savefig(right_out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Top
    [pts_vis, colors_top] = visible_half(pts0, center, colors, "top",
                                          hide_far_half, top_cut_y)
    pts_top = (pts_vis - center) @ R_top.T + center

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_top[:, 0], pts_top[:, 1], pts_top[:, 2],
                     c=colors_top, cmap="gnuplot2", vmin=vmin_, vmax=vmax_, s=6)
    if show_colorbar and not isinstance(colors, str):
        fig.colorbar(sc, ax=ax, shrink=0.6,
                     label=color_col if isinstance(color_col, str) else "psi")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=180, azim=-90)
    ax.grid(False)
    # ax.set_facecolor('#000000')
    ax.axis('off')

    # buf = io.BytesIO()
    fig.savefig(top_out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_mean_pressure_series(fPath, dat2D, file_ext=".csv", file_index=0,
                               noise_floor=0.004, exclude_pattern='0_',
                               group_by=None):
    """
    Loads pressure-log CSV(s) from `fPath`, zeroes out noise below
    `noise_floor`, averages each sensel column over time, and returns
    only the columns that correspond to a real sensel_id present in
    `dat2D` (matching on the numeric part of column names like
    'elem123 [psi.]' -> 123).

    Parameters
    ----------
    ... (unchanged from before)
    group_by : callable or None
        If given, files are grouped by `group_by(filename)` -- e.g. a
        function that pulls (helmet, config) out of each filename -- and
        a separate mean-of-trials Series is computed for EACH group
        (every file in a group counts equally, same "mean of per-file
        means" logic as file_index=None, just done once per group
        instead of once overall). When group_by is given, file_index is
        ignored and a dict {group_key: pd.Series} is returned instead of
        a single Series.

    Returns
    -------
    pd.Series, or dict[group_key, pd.Series] if group_by is given
    """
    entries = [fName for fName in os.listdir(fPath)
               if fName.endswith(file_ext) and exclude_pattern not in fName]
    if not entries:
        raise FileNotFoundError(f"No files matching *{file_ext} (excluding '{exclude_pattern}') found in {fPath}")

    def _file_mean(fname):
        data = pd.read_csv(os.path.join(fPath, fname))
        data[data < noise_floor] = 0
        return data.mean(axis=0, skipna=True)

    def _filter_to_sensels(col_mean):
        elem_num = col_mean.index.str.extract(r'elem(\d+)', expand=False)
        valid = elem_num.notna()
        matches = pd.Series(False, index=col_mean.index)
        matches[valid] = elem_num[valid].astype(int).isin(dat2D['sensel_id'])
        return col_mean[matches]

    if group_by is not None:
        groups = {}
        for fname in entries:
            key = group_by(fname)
            groups.setdefault(key, []).append(fname)

        result = {}
        for key, fnames in groups.items():
            per_file_means = [_file_mean(fn) for fn in fnames]
            col_mean = pd.concat(per_file_means, axis=1).mean(axis=1, skipna=True)
            result[key] = _filter_to_sensels(col_mean)
        return result

    if file_index is None:
        per_file_means = [_file_mean(fname) for fname in entries]
        col_mean = pd.concat(per_file_means, axis=1).mean(axis=1, skipna=True)
    else:
        col_mean = _file_mean(entries[file_index])

    return _filter_to_sensels(col_mean)



MARKER_RADIUS = 1.5      # visual size of each sensel marker (mm)
COLOR_COL = "y"          # column in RESULT_CSV to color by (e.g. a pressure column, or "y" for row index)
CMAP_NAME = "gnuplot2"    # any matplotlib colormap name
VMIN = None              # None -> auto (min of COLOR_COL); set a number to fix the scale
VMAX = None              # None -> auto (max of COLOR_COL)


def make_colored_scene(model_3D: pd.DataFrame, values, marker_radius: float = 1.5,
                        marker_subdiv=(8, 8), cmap_name: str = "gnuplot2",
                        vmin: float = None, vmax: float = None) -> trimesh.Scene:
    """Builds a sphere at every sensel position, colored by `values` using
    a real per-object material (Kd diffuse color) -- NOT the inline
    vertex-color OBJ extension, which many viewers (including PowerPoint)
    don't respect. Returns a trimesh.Scene, which is what actually lets
    each sphere keep its own distinct material on export (concatenating
    into a single mesh first collapses everything down to one material).
    """
    pts = model_3D[["x_3d", "y_3d", "z_3d"]].values
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    scene = trimesh.Scene()
    for idx, (p, v) in enumerate(zip(pts, values)):
        s = trimesh.creation.uv_sphere(radius=marker_radius, count=marker_subdiv)
        s.apply_translation(p)
        rgba = np.array(cmap(norm(v)))
        rgb255 = tuple((rgba[:3] * 255).astype(np.uint8))
        s.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.SimpleMaterial(diffuse=rgb255)
        )
        scene.add_geometry(s, node_name=f"sensel_{idx}")

    return scene
def helmet_config_from_filename(fname):
    parts = fname.replace('.csv', '').split('_')
    return (parts[0], parts[1])   # (helmet, config)

def tactile_head_3D_html(model_3D: pd.DataFrame, out_path: str,
                       color_col: Optional[Union[str, pd.Series]] = "y",
                       ):
    if color_col is None:
        colors = "black"
    elif isinstance(color_col, str):
        colors = model_3D[color_col] if color_col in model_3D.columns else "black"
    else:
        colors = color_col
    
    # Save as standalone interactive HTML file
    fig = px.scatter_3d(model_3D,z = 'z_3d',x ='x_3d', y= 'y_3d',color=colors,hover_name=model_3D.sensel_id)
    # Customize layout
    fig.update_layout(
        title="3D Head Pressure Mapping",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        autosize=True
    )
    fig.update_coloraxes(colorbar_title = "Pressure (psi)")
    # Show in notebook or browser
    fig.show()
    fig.write_html(out_path)
#%%
fname_no_ext = os.path.splitext(entries[file_number])[0] #trial specific
os.makedirs(os.path.join(pressure_data_folder, "3D_Mapping",'gif'), exist_ok=True)
os.makedirs(os.path.join(pressure_data_folder, "3D_Mapping",'obj'), exist_ok=True)
OUT_PATH_GIF_Specific = pressure_data_folder + f'\\3D_Mapping\\gif\\Trial_{fname_no_ext}_rotation.gif' #Trial specific
OUT_PATH_OBJ = pressure_data_folder+f'\\3D_Mapping\\obj\\{fname_no_ext}_head_model_colored.obj'
OUT_PATH_GLB = pressure_data_folder+f'\\3D_Mapping\\obj\\{fname_no_ext}_head_model_colored.glb'
# Specific Trial
os.makedirs(os.path.join(pressure_data_folder, "3D_Mapping","TrialSpecificOutput"), exist_ok=True)
OUT_PATH_TOP = pressure_data_folder+f'\\3D_Mapping\\TrialSpecificOutput\\{fname_no_ext}_top.png'
OUT_PATH_RIGHT = pressure_data_folder+f'\\3D_Mapping\\TrialSpecificOutput\\{fname_no_ext}_right.png'
OUT_PATH_LEFT = pressure_data_folder+f'\\3D_Mapping\\TrialSpecificOutput\\{fname_no_ext}_left.png'
OUT_PATH_BACK = pressure_data_folder+f'\\3D_Mapping\\TrialSpecificOutput\\{fname_no_ext}_back.png'
OUT_PATH_FRONT = pressure_data_folder+f'\\3D_Mapping\\TrialSpecificOutput\\{fname_no_ext}_front.png'
OUT_PATH_HTML = pressure_data_folder+f'\\3D_Mapping\\html\\{fname_no_ext}_front.html'
# Average
os.makedirs(os.path.join(pressure_data_folder, "3D_Mapping","PressureAvg_AllConfigs"), exist_ok=True)
os.makedirs(os.path.join(pressure_data_folder, "3D_Mapping","html"), exist_ok=True)

filtered_series_specific = load_mean_pressure_series(
    pressure_data_folder,
    dat2D, file_index = file_number
)
grouped = load_mean_pressure_series(pressure_data_folder, dat2D, group_by=helmet_config_from_filename)

helmet_config_list = list(grouped.keys())
helmet_config_values_list = list(grouped.values())
if isinstance(COLOR_COL, str):
    values = model_3D[COLOR_COL].values
else:
    values = np.asarray(COLOR_COL)  # allows passing an actual Series/array too
values_specific = filtered_series_specific
if save_obj == 1:
    scene = make_colored_scene(model_3D, values_specific, marker_radius=MARKER_RADIUS,
                                cmap_name=CMAP_NAME, vmin=VMIN, vmax=VMAX)
    # .obj + .mtl -- try this first for PowerPoint (standard per-object material colors)
    scene.export(OUT_PATH_OBJ)
    print(f"Wrote colored .obj (+ .mtl alongside it) to {OUT_PATH_OBJ}")
    
    for (helmet, config), values_average in grouped.items():
        scene = make_colored_scene(model_3D, values_average, marker_radius=MARKER_RADIUS,
                                    cmap_name=CMAP_NAME, vmin=VMIN, vmax=VMAX)
        tag = f"{helmet}_{config}"
        out_path_obj = os.path.join(pressure_data_folder,"3D_Mapping",'obj', f"{fname_no_ext}_head_model.obj")
        out_path_glb = os.path.join(pressure_data_folder,"3D_Mapping",'obj', f"{fname_no_ext}_head_model.glb")
        scene.export(out_path_obj, mtl_name=f"head_model_{tag}.mtl")
        scene.export(out_path_glb)
    
    # .glb -- a single self-contained file (no separate .mtl to lose track of),
    # and generally the most reliably-colored format for PowerPoint's "Insert 3D
    # Models" -- try this if the .obj doesn't show colors correctly
    scene.export(OUT_PATH_GLB)
    print(f"Wrote colored .glb to {OUT_PATH_GLB}")

if save_image == 1:
    head_3D_png(model_3D, OUT_PATH_TOP,
                OUT_PATH_RIGHT, OUT_PATH_LEFT,
                OUT_PATH_BACK,OUT_PATH_FRONT,
                elev= 90,
                color_col=filtered_series_specific,
                zoom = magnification,
                rotation_axis=ROTATION_AXIS, axis_line_length=AXIS_LINE_LENGTH
                )
    for (helmet,config), values_average in grouped.items():
        tag = f"{helmet}_{config}"
        out_path_top = os.path.join(pressure_data_folder,"3D_Mapping",'PressureAvg_AllConfigs',f"{tag}_top.png")
        out_path_left = os.path.join(pressure_data_folder,"3D_Mapping",'PressureAvg_AllConfigs',f"{tag}_left.png")
        out_path_right = os.path.join(pressure_data_folder,"3D_Mapping",'PressureAvg_AllConfigs',f"{tag}_right.png")
        out_path_front= os.path.join(pressure_data_folder,"3D_Mapping",'PressureAvg_AllConfigs',f"{tag}_front.png")
        out_path_back = os.path.join(pressure_data_folder,"3D_Mapping",'PressureAvg_AllConfigs',f"{tag}_back.png")
        out_path_html = os.path.join(pressure_data_folder,"3D_Mapping",'html',f"{tag}_back.png")
        head_3D_png(model_3D, out_path_top,
                    out_path_right, out_path_left,
                    out_path_back,out_path_front,
                    elev= 90,
                    color_col= values_average,
                    zoom = magnification,
                    rotation_axis=ROTATION_AXIS, axis_line_length=AXIS_LINE_LENGTH
                    )
        
if save_gif == 1:
    # Create GIF of the 3D Pressure Head Data
    make_rotating_gif(model_3D, OUT_PATH_GIF_Specific, n_frames=N_FRAMES, elev=ELEV,
                       duration_ms=DURATION_MS, color_col=filtered_series_specific,
                      zoom = magnification,  rotation_axis=ROTATION_AXIS, axis_line_length=AXIS_LINE_LENGTH)
    for (helmet,config), values_average in grouped.items():
        tag = f"{helmet}_{config}"
        out_path_gif_average = os.path.join(pressure_data_folder,"3D_Mapping",'gif',f"{tag}_rotation.gif")
        make_rotating_gif(model_3D, out_path_gif_average, n_frames=N_FRAMES, elev=ELEV,
                           duration_ms=DURATION_MS, color_col=values_average,
                          zoom = magnification, rotation_axis=ROTATION_AXIS, axis_line_length=AXIS_LINE_LENGTH)
    
    
    for (helmet,config), values_average in grouped.items():
       tag = f"{helmet}_{config}"
       out_path_html = os.path.join(pressure_data_folder,"3D_Mapping",'html',f"{tag}.html")
       tactile_head_3D_html(model_3D, out_path_html,
                            color_col=values_average,
                            )

matplotlib.use(original_backend)  # switch back afterward