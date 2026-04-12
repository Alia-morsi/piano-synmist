"""
Visualization utilities for lowlvl.py

Plots the warping path (time_from → time_to) alongside piano roll
representations of src_na and tgt_na using note rectangles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import os


def _get_active_pitch_range(na_list, margin=2):
    """Find min/max active pitches across one or more note arrays."""
    all_pitches = []
    for na in na_list:
        if len(na) > 0:
            all_pitches.extend(na['pitch'].tolist())
    if not all_pitches:
        return 48, 84
    lo = max(0, min(all_pitches) - margin)
    hi = min(127, max(all_pitches) + margin)
    return int(lo), int(hi)


def _draw_piano_roll_horizontal(ax, na, t_start, t_end, lo_pitch, hi_pitch,
                                 cmap_name='Blues', alpha=0.85):
    """
    Draw a piano roll on ax with time on x-axis, pitch on y-axis.
    Each note is a rectangle: x=onset, width=duration, y=pitch, height=1.
    """
    patches = []
    colors = []
    for note in na:
        onset = float(note['onset_sec'])
        dur = float(note['duration_sec'])
        pitch = int(note['pitch'])
        if onset + dur < t_start or onset > t_end:
            continue
        if pitch < lo_pitch or pitch > hi_pitch:
            continue
        rect = Rectangle((onset, pitch - 0.4), max(dur, 0.02), 0.8)
        patches.append(rect)
        vel = int(note['velocity'])
        colors.append(vel / 127.0)

    if patches:
        pc = PatchCollection(patches, cmap=cmap_name, alpha=alpha,
                             edgecolors='#2c3e50', linewidths=0.3)
        pc.set_array(np.array(colors))
        pc.set_clim(0, 1)
        ax.add_collection(pc)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(lo_pitch - 1, hi_pitch + 1)


def _draw_piano_roll_vertical(ax, na, t_start, t_end, lo_pitch, hi_pitch,
                                cmap_name='Greens', alpha=0.85):
    """
    Draw a piano roll on ax with time on Y-axis, pitch on X-axis.
    Each note: x=pitch, width=1, y=onset, height=duration.
    """
    patches = []
    colors = []
    for note in na:
        onset = float(note['onset_sec'])
        dur = float(note['duration_sec'])
        pitch = int(note['pitch'])
        if onset + dur < t_start or onset > t_end:
            continue
        if pitch < lo_pitch or pitch > hi_pitch:
            continue
        rect = Rectangle((pitch - 0.4, onset), 0.8, max(dur, 0.02))
        patches.append(rect)
        vel = int(note['velocity'])
        colors.append(vel / 127.0)

    if patches:
        pc = PatchCollection(patches, cmap=cmap_name, alpha=alpha,
                             edgecolors='#2c3e50', linewidths=0.3)
        pc.set_array(np.array(colors))
        pc.set_clim(0, 1)
        ax.add_collection(pc)

    ax.set_xlim(lo_pitch - 1, hi_pitch + 1)
    ax.set_ylim(t_start, t_end)


def _pitch_ticks(ax, lo, hi, axis='y'):
    """Add readable pitch labels (note names) at every C and F."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    ticks = []
    labels = []
    for p in range(lo, hi + 1):
        name = note_names[p % 12]
        if name in ('C', 'F'):
            octave = (p // 12) - 1
            ticks.append(p)
            labels.append(f'{name}{octave}')

    if axis == 'y':
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')


def plot_lowlvl_state(ll_inst, title="", time_interval=None, save_path=None, figsize=None):
    """
    Plot the current state of a lowlvl instance:
      - Main panel: warping path (time_from on x → time_to on y) + repeat_tracker
      - Bottom panel: src piano roll (time on x, pitch on y)
      - Left panel: tgt piano roll (pitch on x, time on y — aligned with main y-axis)
      - Labels shown as colored spans on the warping path panel

    Parameters
    ----------
    ll_inst : lowlvl instance
    title : str
    time_interval : tuple (start, end) in src seconds. None = full range.
    save_path : str or None
    figsize : tuple or None

    Returns
    -------
    fig : matplotlib Figure
    """
    src_na = ll_inst.src_na
    tgt_na = ll_inst.tgt_na
    time_from = ll_inst.time_from
    time_to = ll_inst.time_to

    # ── Determine time windows ──────────────────────────────────────
    if time_interval is None:
        src_t_start = float(time_from[0])
        src_t_end = float(time_from[-1])
    else:
        src_t_start, src_t_end = time_interval

    # Crop warping path to src interval
    mask = (time_from >= src_t_start) & (time_from <= src_t_end)
    tf_crop = time_from[mask]
    tt_crop = time_to[mask]

    # Target time range
    if len(tt_crop) > 0:
        tgt_t_start = float(tt_crop.min())
        tgt_t_end = float(tt_crop.max())
    else:
        tgt_t_start, tgt_t_end = src_t_start, src_t_end

    # Extend tgt range with repeat_tracker
    for key, (rt_tt, rt_tf) in ll_inst.repeat_tracker.items():
        rt_mask = (rt_tf >= src_t_start) & (rt_tf <= src_t_end)
        if np.any(rt_mask):
            tgt_t_start = min(tgt_t_start, float(rt_tt[rt_mask].min()))
            tgt_t_end = max(tgt_t_end, float(rt_tt[rt_mask].max()))

    # Small padding
    src_pad = (src_t_end - src_t_start) * 0.02
    tgt_pad = (tgt_t_end - tgt_t_start) * 0.02
    src_t_start -= src_pad
    src_t_end += src_pad
    tgt_t_start -= tgt_pad
    tgt_t_end += tgt_pad

    # Pitch range from both arrays
    lo_pitch, hi_pitch = _get_active_pitch_range([src_na, tgt_na])

    # ── Figure and grid ─────────────────────────────────────────────
    if figsize is None:
        figsize = (14, 10)

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = GridSpec(2, 2,
                  width_ratios=[1, 3.5],
                  height_ratios=[3.5, 1],
                  wspace=0.05, hspace=0.05)

    # ── Main panel: warping path ────────────────────────────────────
    ax_warp = fig.add_subplot(gs[0, 1])

    # Identity reference
    id_lo = max(src_t_start, tgt_t_start)
    id_hi = min(src_t_end, tgt_t_end)
    if id_lo < id_hi:
        ax_warp.plot([id_lo, id_hi], [id_lo, id_hi],
                     '-', color='#dcdde1', linewidth=1.0, zorder=1, label='identity')

    # Main warping path
    ax_warp.plot(tf_crop, tt_crop, color='#2c3e50', linewidth=0.8, zorder=3, label='warping path')

    # Repeat tracker paths
    rt_colors = ['#e74c3c', '#8e44ad', '#f39c12', '#1abc9c', '#d35400']
    for idx, (key, (rt_tt, rt_tf)) in enumerate(ll_inst.repeat_tracker.items()):
        c = rt_colors[idx % len(rt_colors)]
        rt_mask = (rt_tf >= src_t_start) & (rt_tf <= src_t_end)
        if np.any(rt_mask):
            ax_warp.plot(rt_tf[rt_mask], rt_tt[rt_mask],
                         color=c, linewidth=1.5, zorder=4,
                         label=f'repeat {idx}')

    # Label spans
    label_colors = {
        'pitch_insert': '#3498db',
        'pitch_delete': '#e74c3c',
        'time_shift': '#f39c12',
        'change_offset': '#2ecc71',
        'change_onset': '#9b59b6',
    }
    shown = set()
    for lbl in ll_inst.label_na:
        low = str(lbl['lowlvl_label'])
        onset = float(lbl['onset_sec'])
        dur = float(lbl['duration_sec'])
        if onset > tgt_t_end or onset + dur < tgt_t_start:
            continue
        c = label_colors.get(low, '#95a5a6')
        ax_warp.axhspan(onset, onset + dur, alpha=0.15, color=c,
                         zorder=2, label=low if low not in shown else None)
        shown.add(low)

    ax_warp.set_xlim(src_t_start, src_t_end)
    ax_warp.set_ylim(tgt_t_start, tgt_t_end)
    ax_warp.set_xlabel('Source time (s)', fontsize=9)
    ax_warp.set_ylabel('Target time (s)', fontsize=9)
    ax_warp.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax_warp.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax_warp.tick_params(labelsize=8)
    ax_warp.grid(True, alpha=0.15)

    # ── Bottom panel: source piano roll ─────────────────────────────
    ax_src = fig.add_subplot(gs[1, 1])
    _draw_piano_roll_horizontal(ax_src, src_na, src_t_start, src_t_end,
                                 lo_pitch, hi_pitch, cmap_name='Blues')
    ax_src.set_xlabel('Source time (s)', fontsize=9)
    ax_src.set_ylabel('Pitch', fontsize=9)
    ax_src.set_xlim(src_t_start, src_t_end)
    _pitch_ticks(ax_src, lo_pitch, hi_pitch, axis='y')
    ax_src.tick_params(labelsize=8)
    ax_src.grid(True, alpha=0.1, axis='x')

    # ── Left panel: target piano roll (rotated) ─────────────────────
    ax_tgt = fig.add_subplot(gs[0, 0])
    _draw_piano_roll_vertical(ax_tgt, tgt_na, tgt_t_start, tgt_t_end,
                               lo_pitch, hi_pitch, cmap_name='Greens')
    ax_tgt.set_xlabel('Pitch', fontsize=9)
    ax_tgt.set_ylabel('Target time (s)', fontsize=9)
    ax_tgt.set_ylim(tgt_t_start, tgt_t_end)
    _pitch_ticks(ax_tgt, lo_pitch, hi_pitch, axis='x')
    ax_tgt.tick_params(labelsize=8)
    ax_tgt.grid(True, alpha=0.1, axis='y')

    # ── Empty corner ────────────────────────────────────────────────
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis('off')

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_cumulative_operations(ll_inst, operations, time_interval=None,
                                output_dir="plots", prefix="step"):
    """
    Apply a list of operations one by one, plotting after each.

    Parameters
    ----------
    ll_inst : lowlvl instance (will be mutated)
    operations : list of dicts with keys:
        'name': str
        'func': str — method name on ll_inst
        'args': dict — keyword arguments
    time_interval : tuple or None
    output_dir : str
    prefix : str

    Returns
    -------
    figs : list of matplotlib Figures
    """
    os.makedirs(output_dir, exist_ok=True)
    figs = []

    fig = plot_lowlvl_state(ll_inst,
                            title="Step 0: Initial state",
                            time_interval=time_interval,
                            save_path=os.path.join(output_dir, f"{prefix}_00_initial.png"))
    figs.append(fig)
    plt.close(fig)

    for i, op in enumerate(operations, start=1):
        method = getattr(ll_inst, op['func'])
        method(**op['args'])

        fig = plot_lowlvl_state(
            ll_inst,
            title=f"Step {i}: {op['name']}",
            time_interval=time_interval,
            save_path=os.path.join(output_dir, f"{prefix}_{i:02d}_{op['func']}.png"))
        figs.append(fig)
        plt.close(fig)

    return figs
