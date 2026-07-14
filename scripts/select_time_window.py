#!/usr/bin/env python3
"""
Interactive time window selector for OFT trajectory data.

Shows a split view:
  - Top panel: fused 2D trajectory for the selected time window
    - Bottom panel: movement-bout timeline with a fixed-width movable selector

Controls:
    - Drag the slider to move the center of the time window
    - Edit the start-time box to set window start time in seconds
    - Edit the duration box to change the fixed window width
  - Press 'S' or click "Save window" to write the selection to JSON
  - Press 'R' to reset to the full session
  - Press 'Q' / ESC to quit

Output:
  results/oft_analysis/<monkey>/<date>/selected_time_window.json
    {
      "start_sec": ...,
      "end_sec": ...,
      "start_abs": "...",
      "end_abs": "...",
      "duration_sec": ...
    }

Usage:
  conda activate video_tracking
  python scripts/select_time_window.py
  python scripts/select_time_window.py --monkey H1705011 --date 2026-05-25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, TextBox


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RESULTS_BASE  = PROJECT_ROOT / 'results' / 'oft_analysis'

MONKEY_ID     = 'H1705011'
SESSION_DATE  = '2026-05-25'

ARENA_SIZE    = 322   # cm
VELOCITY_THRESHOLD = 5.0
MIN_BOUT_DURATION = 0.5
MISSING_GAP_THRESHOLD_SEC = 1.0


# ── Arena background ───────────────────────────────────────────────────────────

def _draw_arena(ax):
    ax.set_xlim(-5, ARENA_SIZE + 5)
    ax.set_ylim(-5, ARENA_SIZE + 5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Outer boundary
    rect = mpatches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                               linewidth=2, edgecolor='black', facecolor='#f5f5f5')
    ax.add_patch(rect)

    # Quadrant dividers
    ax.axvline(ARENA_SIZE / 2, color='#cccccc', lw=1, ls='--')
    ax.axhline(ARENA_SIZE / 2, color='#cccccc', lw=1, ls='--')

    # Quadrant labels
    for (lx, ly, label) in [
        (40,  15,  'D1 (top-left)'),
        (200, 15,  'D2 (top-right)'),
        (40,  310, 'D3 (bottom-left)'),
        (200, 310, 'D4 (bottom-right)'),
    ]:
        ax.text(lx, ly, label, fontsize=8, color='#aaaaaa', va='top')

    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)


def _detect_bouts(df: pd.DataFrame,
                  vel_thresh: float = VELOCITY_THRESHOLD,
                  min_dur: float = MIN_BOUT_DURATION) -> pd.DataFrame:
    """Detect movement bouts using the same rule as the fusion summary."""
    if df.empty:
        return pd.DataFrame(columns=['start_sec', 'end_sec', 'duration_sec'])

    bout_df = df[['time_sec', 'velocity_cm_s']].copy()
    bout_df['is_moving'] = bout_df['velocity_cm_s'] > vel_thresh
    bout_df['bout_change'] = bout_df['is_moving'].astype(int).diff()

    bouts = []
    start = None
    for _, row in bout_df.iterrows():
        if row['bout_change'] == 1:
            start = row['time_sec']
        elif row['bout_change'] == -1 and start is not None:
            duration_sec = row['time_sec'] - start
            if duration_sec >= min_dur:
                bouts.append({
                    'start_sec': start,
                    'end_sec': row['time_sec'],
                    'duration_sec': duration_sec,
                })
            start = None
    return pd.DataFrame(bouts)


def _find_missing_spans(time_sec: pd.Series,
                        gap_threshold_sec: float = MISSING_GAP_THRESHOLD_SEC) -> list[tuple[float, float]]:
    """Return (start_sec, end_sec) spans where trajectory timestamps have long gaps."""
    if time_sec.empty:
        return []

    ts = pd.Series(time_sec).sort_values().reset_index(drop=True)
    diffs = ts.diff()
    spans: list[tuple[float, float]] = []
    for i in range(1, len(ts)):
        if float(diffs.iloc[i]) > gap_threshold_sec:
            spans.append((float(ts.iloc[i - 1]), float(ts.iloc[i])))
    return spans


# ── Main app ───────────────────────────────────────────────────────────────────

def run(monkey_id: str, session_date: str):
    data_dir = RESULTS_BASE / monkey_id / session_date
    traj_path = data_dir / 'trajectory_realworld.csv'
    out_path  = data_dir / 'selected_time_window.json'

    if not traj_path.exists():
        raise SystemExit(f"ERROR: trajectory not found at {traj_path}\n"
                         "Run fuse_trajectories.py first.")

    traj = pd.read_csv(traj_path)
    traj['absolute_time'] = pd.to_datetime(traj['absolute_time'])
    missing_spans_sec = _find_missing_spans(traj['time_sec'])

    t_min  = traj['time_sec'].min()
    t_max  = traj['time_sec'].max()
    t_span = t_max - t_min

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10.5))
    fig.canvas.manager.set_window_title(
        f"OFT Time Window Selector — {monkey_id} / {session_date}")

    # Axes arrangement: arena | stacked bout histograms + slider + start/duration + buttons
    ax_arena  = fig.add_axes([0.04, 0.14, 0.40, 0.78])   # trajectory map
    ax_bouts  = fig.add_axes([0.50, 0.70, 0.47, 0.20])   # movement bout count histogram
    ax_speed  = fig.add_axes([0.50, 0.48, 0.47, 0.20])   # moving velocity histogram
    ax_slider = fig.add_axes([0.50, 0.41, 0.47, 0.05])   # center slider
    ax_start  = fig.add_axes([0.50, 0.31, 0.12, 0.055])  # start-sec box
    ax_dur    = fig.add_axes([0.64, 0.31, 0.12, 0.055])  # duration box
    ax_save   = fig.add_axes([0.78, 0.31, 0.09, 0.055])  # save button
    ax_reset  = fig.add_axes([0.88, 0.31, 0.09, 0.055])  # reset button
    ax_info   = fig.add_axes([0.50, 0.06, 0.47, 0.20])   # stats text
    ax_info.axis('off')

    movement_bouts = _detect_bouts(traj)

    # ── 1-minute histograms over the full session ─────────────────────────────
    bin_sec = 60.0
    bin_edges_sec = np.arange(t_min, t_max + bin_sec, bin_sec)
    if len(bin_edges_sec) < 2:
        bin_edges_sec = np.array([t_min, t_max + bin_sec])
    bin_centers_min = (bin_edges_sec[:-1] + bin_edges_sec[1:]) / 120.0
    bar_width_min = (bin_edges_sec[1] - bin_edges_sec[0]) / 60.0 * 0.9

    # Bout count per minute: count bout starts that fall in each minute bin.
    bout_counts, _ = np.histogram(movement_bouts['start_sec'].values if not movement_bouts.empty else [], bins=bin_edges_sec)

    moving_mask = traj['velocity_cm_s'] > VELOCITY_THRESHOLD
    moving_sub = traj.loc[moving_mask, ['time_sec', 'velocity_cm_s']].copy()
    if moving_sub.empty:
        moving_means = np.zeros(len(bin_centers_min), dtype=float)
    else:
        moving_sub['bin'] = pd.cut(moving_sub['time_sec'], bins=bin_edges_sec, right=False, include_lowest=True)
        moving_means = (
            moving_sub.groupby('bin', observed=False)['velocity_cm_s']
            .mean()
            .reindex(pd.IntervalIndex.from_breaks(bin_edges_sec, closed='left'), fill_value=np.nan)
            .to_numpy()
        )
        moving_means = np.nan_to_num(moving_means, nan=0.0)

    ax_bouts.bar(bin_centers_min, bout_counts, width=bar_width_min, color='seagreen', alpha=0.55, edgecolor='none')
    ax_bouts.set_ylabel('Bout count', fontsize=9)
    ax_bouts.set_title('Movement bouts and moving velocity (1-min bins)', fontsize=10)
    ax_bouts.set_xlim(t_min / 60, t_max / 60)
    ax_bouts.set_xticklabels([])
    ax_bouts.grid(axis='y', alpha=0.2)

    ax_speed.bar(bin_centers_min, moving_means, width=bar_width_min, color='steelblue', alpha=0.55, edgecolor='none')
    ax_speed.axhline(VELOCITY_THRESHOLD, color='salmon', lw=1, ls='--', alpha=0.6, label='5 cm/s threshold')
    ax_speed.set_xlabel('Time (min)', fontsize=9)
    ax_speed.set_ylabel('Mean moving velocity\n(cm/s)', fontsize=9)
    ax_speed.set_xlim(t_min / 60, t_max / 60)
    ax_speed.grid(axis='y', alpha=0.2)

    # Mark long timeline gaps as likely missing video periods on both histograms.
    missing_label_used = False
    for start_sec, end_sec in missing_spans_sec:
        for ax in (ax_bouts, ax_speed):
            ax.axvspan(
                start_sec / 60,
                end_sec / 60,
                color='red',
                alpha=0.12,
                zorder=0,
                label='missing video' if not missing_label_used else None,
            )
        missing_label_used = True

    ax_speed.legend(handles=[
        mpatches.Patch(color='seagreen', alpha=0.55, label='movement bout count'),
        mpatches.Patch(color='steelblue', alpha=0.55, label='mean moving velocity'),
        mpatches.Patch(color='red', alpha=0.12, label='missing video'),
    ], fontsize=8, loc='upper right')

    # Highlighted window overlay on the histograms
    default_duration_min = max(5.0, min(30.0, t_span / 60))
    default_center_min = (t_min / 60 + t_max / 60) / 2

    win_patch_bouts = ax_bouts.axvspan(default_center_min - default_duration_min / 2,
                                       default_center_min + default_duration_min / 2,
                                       color='steelblue', alpha=0.15, zorder=2)
    win_patch_speed = ax_speed.axvspan(default_center_min - default_duration_min / 2,
                                       default_center_min + default_duration_min / 2,
                                       color='steelblue', alpha=0.15, zorder=2)

    # ── Slider ─────────────────────────────────────────────────────────────────
    slider = Slider(ax_slider, 'Center (min)',
                    t_min / 60, t_max / 60,
                    valinit=default_center_min,
                    color='steelblue')
    start_box = TextBox(ax_start, 'Start (s)',
                        initial=f'{max(t_min, default_center_min * 60 - default_duration_min * 30):.1f}')
    duration_box = TextBox(ax_dur, 'Duration (min)',
                           initial=f'{default_duration_min:.1f}')

    # ── Arena scatter (will be updated) ────────────────────────────────────────
    _draw_arena(ax_arena)
    sc = ax_arena.scatter([], [], s=3, c=[], cmap='viridis',
                          vmin=t_min, vmax=t_max, alpha=0.5, zorder=5)
    start_pt, = ax_arena.plot([], [], 'go', ms=10,
                               markeredgecolor='white', markeredgewidth=1.5,
                               zorder=10, label='Start')
    end_pt,   = ax_arena.plot([], [], 'rs', ms=10,
                               markeredgecolor='white', markeredgewidth=1.5,
                               zorder=10, label='End')
    ax_arena.legend(loc='upper right', fontsize=8)
    ax_arena.set_title('Trajectory in selected window', fontsize=11)
    cbar_ax = fig.add_axes([0.45, 0.12, 0.01, 0.82])
    fig.colorbar(sc, cax=cbar_ax, label='time_sec')

    # Info text
    info_text = ax_info.text(0.0, 1.0, '', transform=ax_info.transAxes,
                              fontsize=8.5, va='top', linespacing=1.5)

    # ── State ──────────────────────────────────────────────────────────────────
    state = {'lo': t_min, 'hi': t_max, 'duration_min': default_duration_min}

    def _refresh(center_min, duration_min):
        half_width = duration_min / 2
        lo_min = max(t_min / 60, center_min - half_width)
        hi_min = min(t_max / 60, center_min + half_width)

        if hi_min - lo_min < duration_min:
            if lo_min <= t_min / 60:
                hi_min = min(t_max / 60, lo_min + duration_min)
            elif hi_min >= t_max / 60:
                lo_min = max(t_min / 60, hi_min - duration_min)

        lo = lo_min * 60
        hi = hi_min * 60
        state['lo'] = lo
        state['hi'] = hi
        state['duration_min'] = hi_min - lo_min
        if start_box.text != f"{lo:.1f}":
            start_box.set_val(f"{lo:.1f}")

        sub = traj[(traj['time_sec'] >= lo) & (traj['time_sec'] <= hi)]
        if sub.empty:
            return

        # Arena scatter
        sc.set_offsets(np.c_[sub['x_cm'].values, sub['y_cm'].values])
        sc.set_array(sub['time_sec'].values)
        sc.set_clim(lo, hi)

        # Start / end markers
        start_pt.set_data([sub['x_cm'].iloc[0]], [sub['y_cm'].iloc[0]])
        end_pt.set_data([sub['x_cm'].iloc[-1]], [sub['y_cm'].iloc[-1]])

        # Window highlight on both histograms
        win_patch_bouts.set_x(lo_min)
        win_patch_bouts.set_width(hi_min - lo_min)
        win_patch_speed.set_x(lo_min)
        win_patch_speed.set_width(hi_min - lo_min)

        # Stats
        dur   = hi - lo
        dist  = sub['distance_cm'].sum()
        immob = (sub['velocity_cm_s'] <= VELOCITY_THRESHOLD).mean() * 100
        moving = sub['velocity_cm_s'] > VELOCITY_THRESHOLD
        bouts = _detect_bouts(sub)
        max_vel = sub['velocity_cm_s'].max()
        active_time = bouts['duration_sec'].sum() if not bouts.empty else 0.0
        mean_bout_duration = bouts['duration_sec'].mean() if not bouts.empty else 0.0
        mean_vel_moving = sub.loc[moving, 'velocity_cm_s'].mean() if moving.any() else 0.0
        t0    = sub['absolute_time'].iloc[0].strftime('%H:%M:%S')
        t1    = sub['absolute_time'].iloc[-1].strftime('%H:%M:%S')
        info_text.set_text(
            f"Window:  {t0}  →  {t1}\n"
            f"Duration:  {dur:.0f} s  ({dur/60:.1f} min)\n"
            f"Distance:  {dist/100:.1f} m ({dist:.0f} cm)\n"
            f"Max velocity:  {max_vel:.2f} cm/s\n"
            f"Movement bouts:  {len(bouts)}\n"
            f"Mean bout duration:  {mean_bout_duration:.2f} s\n"
            f"Active time:  {active_time:.1f} s\n"
            f"Immobility:  {immob:.1f}%\n"
            f"Mean velocity:  {sub['velocity_cm_s'].mean():.2f} cm/s\n"
            f"Mean velocity (moving):  {mean_vel_moving:.2f} cm/s"
        )
        fig.canvas.draw_idle()

    # Initial draw
    _refresh(default_center_min, default_duration_min)

    def _on_slider(val):
        _refresh(val, state['duration_min'])

    slider.on_changed(_on_slider)

    def _on_duration_submit(text):
        try:
            duration_min = float(text)
        except ValueError:
            duration_box.set_val(f"{state['duration_min']:.1f}")
            return

        max_duration = max((t_max - t_min) / 60, 0.1)
        duration_min = max(0.1, min(duration_min, max_duration))
        if abs(duration_min - state['duration_min']) > 1e-9:
            duration_box.set_val(f"{duration_min:.1f}")
        _refresh(slider.val, duration_min)

    def _on_start_submit(text):
        try:
            start_sec = float(text)
        except ValueError:
            start_box.set_val(f"{state['lo']:.1f}")
            return

        duration_sec = state['duration_min'] * 60
        max_start = max(t_max - duration_sec, t_min)
        start_sec = max(t_min, min(start_sec, max_start))
        center_min = (start_sec + duration_sec / 2) / 60

        if abs(start_sec - state['lo']) > 1e-9:
            start_box.set_val(f"{start_sec:.1f}")
        slider.set_val(center_min)

    duration_box.on_submit(_on_duration_submit)
    start_box.on_submit(_on_start_submit)

    # ── Buttons ────────────────────────────────────────────────────────────────
    btn_save  = Button(ax_save,  'Save window', color='#c8e6c9', hovercolor='#a5d6a7')
    btn_reset = Button(ax_reset, 'Reset',       color='#ffe0b2', hovercolor='#ffcc80')

    def _save(_):
        lo, hi = state['lo'], state['hi']
        window_duration_sec = float(max(0.0, hi - lo))
        sub = traj[(traj['time_sec'] >= lo) & (traj['time_sec'] <= hi)]
        bouts = _detect_bouts(sub) if not sub.empty else pd.DataFrame(columns=['duration_sec'])
        moving = sub['velocity_cm_s'] > VELOCITY_THRESHOLD if not sub.empty else pd.Series(dtype=bool)
        mean_velocity_moving = float(sub.loc[moving, 'velocity_cm_s'].mean()) if not sub.empty and moving.any() else 0.0
        total_active_time = float(bouts['duration_sec'].sum()) if len(bouts) > 0 else 0.0
        total_active_time = min(total_active_time, window_duration_sec)
        immobility_time = max(0.0, window_duration_sec - total_active_time)
        t0  = sub['absolute_time'].iloc[0].isoformat() if not sub.empty else ''
        t1  = sub['absolute_time'].iloc[-1].isoformat() if not sub.empty else ''
        result = {
            'monkey_id':    monkey_id,
            'session_date': session_date,
            'start_sec':    round(lo, 3),
            'end_sec':      round(hi, 3),
            'start_abs':    t0,
            'end_abs':      t1,
            'duration_sec': round(hi - lo, 3),
            'mean_velocity_moving_cm_s': round(mean_velocity_moving, 3),
            'distance_cm':  round(float(sub['distance_cm'].sum()), 2) if not sub.empty else 0.0,
            'max_velocity_cm_s': round(float(sub['velocity_cm_s'].max()), 3) if not sub.empty else 0.0,
            'n_movement_bouts': int(len(bouts)),
            'mean_bout_duration_sec': round(float(bouts['duration_sec'].mean()), 3) if len(bouts) > 0 else 0.0,
            'total_active_time_sec': round(total_active_time, 2),
            'immobility_time_sec': round(immobility_time, 2),
            'velocity_threshold_cm_s': VELOCITY_THRESHOLD,
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅  Window saved → {out_path}")
        print(f"   {t0}  →  {t1}  ({(hi - lo)/60:.1f} min)")
        ax_save.set_facecolor('#81c784')
        fig.canvas.draw_idle()

    def _reset(_):
        state['duration_min'] = default_duration_min
        duration_box.set_val(f'{default_duration_min:.1f}')
        slider.set_val(default_center_min)
        start_box.set_val(f'{state["lo"]:.1f}')
        ax_save.set_facecolor('#c8e6c9')
        fig.canvas.draw_idle()

    btn_save.on_clicked(_save)
    btn_reset.on_clicked(_reset)

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────
    def _on_key(event):
        if event.key in ('s', 'S'):
            _save(None)
        elif event.key in ('r', 'R'):
            _reset(None)
        elif event.key in ('q', 'Q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)

    fig.suptitle(f'OFT Time Window Selector   {monkey_id} / {session_date}',
                 fontsize=13, fontweight='bold')
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Interactive OFT time window selector')
    ap.add_argument('--monkey', default=MONKEY_ID,    help='Monkey ID')
    ap.add_argument('--date',   default=SESSION_DATE, help='Session date (YYYY-MM-DD)')
    args = ap.parse_args()
    run(args.monkey, args.date)


if __name__ == '__main__':
    main()
