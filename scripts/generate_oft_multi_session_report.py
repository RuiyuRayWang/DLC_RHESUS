#!/usr/bin/env python3
"""
Generate a multi-session HTML report for OFT trajectory summaries.

The report scans an oft_analysis/<monkey>/ tree and combines multiple session
folders into one HTML page with:
  - a summary table across sessions
  - per-session 1-minute histograms for movement-bout counts and mean moving velocity
  - key summary metrics from trajectory_summary.json

Expected session artifacts inside each date folder:
  - trajectory_summary.json
  - trajectory_realworld.csv
  - movement_bouts.csv

The output is a single self-contained HTML file with inline SVG charts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = PROJECT_ROOT / "results" / "oft_analysis"
DEFAULT_BOUT_MIN_DURATION_SEC = 0.5
DEFAULT_VELOCITY_THRESHOLD_CM_S = 5.0
MASK_FILENAME = "trajectory_ignore_mask.csv"


@dataclass
class SessionReport:
    monkey_id: str
    session_date: str
    session_dir: Path
    summary: dict
    time_min_bins: np.ndarray
    bout_counts: np.ndarray
    moving_velocity_mean: np.ndarray
    missing_spans: list[tuple[float, float]]
    bout_rows: pd.DataFrame


def recompute_summary_from_traj(traj: pd.DataFrame, vel_thresh: float, min_bout_dur: float) -> tuple[dict, pd.DataFrame]:
    traj = traj.sort_values("time_sec").reset_index(drop=True).copy()
    if traj.empty:
        return {}, pd.DataFrame(columns=["start_sec", "end_sec", "duration_sec"])

    dx = traj["x_cm"].diff() if "x_cm" in traj.columns else pd.Series(np.nan, index=traj.index)
    dy = traj["y_cm"].diff() if "y_cm" in traj.columns else pd.Series(np.nan, index=traj.index)
    dt = traj["time_sec"].diff()
    if "x_cm" in traj.columns and "y_cm" in traj.columns:
        dist = np.sqrt(dx**2 + dy**2)
        vel = (dist / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        traj["velocity_cm_s"] = vel
        traj["distance_cm"] = dist.fillna(0.0)
    elif "velocity_cm_s" not in traj.columns:
        traj["velocity_cm_s"] = 0.0

    bouts = detect_bouts(traj, vel_thresh=vel_thresh, min_dur=min_bout_dur)

    total_dur = float(traj["time_sec"].max() - traj["time_sec"].min()) if len(traj) > 1 else 0.0
    if "distance_cm" in traj.columns:
        total_dist = float(traj["distance_cm"].fillna(0.0).sum())
    else:
        total_dist = 0.0
    mean_vel = float(traj["velocity_cm_s"].mean()) if len(traj) > 0 else 0.0
    moving_mask = traj["velocity_cm_s"] > vel_thresh
    mean_vel_moving = float(traj.loc[moving_mask, "velocity_cm_s"].mean()) if moving_mask.any() else 0.0
    immob_frames = int((traj["velocity_cm_s"] <= vel_thresh).sum())
    immob_pct = 100.0 * immob_frames / len(traj) if len(traj) > 0 else 0.0

    summary = {
        "total_duration_sec": round(total_dur, 2),
        "total_distance_cm": round(total_dist, 2),
        "mean_velocity_cm_s": round(mean_vel, 3),
        "mean_velocity_moving_cm_s": round(mean_vel_moving, 3),
        "max_velocity_cm_s": round(float(traj["velocity_cm_s"].max()) if len(traj) > 0 else 0.0, 3),
        "immobility_pct": round(float(immob_pct), 2),
        "n_movement_bouts": int(len(bouts)),
        "mean_bout_duration_sec": round(float(bouts["duration_sec"].mean()), 3) if len(bouts) > 0 else 0.0,
        "total_active_time_sec": round(float(bouts["duration_sec"].sum()), 2) if len(bouts) > 0 else 0.0,
        "velocity_threshold_cm_s": float(vel_thresh),
        "start_sec": float(traj["time_sec"].min()) if len(traj) else 0.0,
        "end_sec": float(traj["time_sec"].max()) if len(traj) else 0.0,
    }
    return summary, bouts


def detect_bouts(df: pd.DataFrame, vel_thresh: float, min_dur: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["start_sec", "end_sec", "duration_sec"])

    ts = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(dtype=float)
    vel = pd.to_numeric(df["velocity_cm_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_moving = vel > float(vel_thresh)

    # Match prior behavior: bouts start on False->True transitions and end on True->False.
    changes = np.diff(is_moving.astype(np.int8), prepend=is_moving[:1].astype(np.int8))
    start_idx = np.flatnonzero(changes == 1)
    end_idx = np.flatnonzero(changes == -1)

    bouts: list[dict[str, float]] = []
    if start_idx.size == 0 or end_idx.size == 0:
        return pd.DataFrame(columns=["start_sec", "end_sec", "duration_sec"])

    end_ptr = 0
    for s_idx in start_idx:
        while end_ptr < end_idx.size and end_idx[end_ptr] <= s_idx:
            end_ptr += 1
        if end_ptr >= end_idx.size:
            break
        e_idx = end_idx[end_ptr]
        start = float(ts[s_idx])
        end = float(ts[e_idx])
        duration = end - start
        if duration >= float(min_dur):
            bouts.append({"start_sec": start, "end_sec": end, "duration_sec": duration})

    return pd.DataFrame(bouts, columns=["start_sec", "end_sec", "duration_sec"])


def find_missing_spans(time_sec: pd.Series, gap_threshold_sec: float = 1.0) -> list[tuple[float, float]]:
    if time_sec.empty:
        return []
    ts = pd.Series(time_sec).sort_values().reset_index(drop=True)
    diffs = ts.diff()
    spans: list[tuple[float, float]] = []
    for i in range(1, len(ts)):
        if float(diffs.iloc[i]) > gap_threshold_sec:
            spans.append((float(ts.iloc[i - 1]), float(ts.iloc[i])))
    return spans


def load_session(session_dir: Path, monkey_id: str) -> SessionReport | None:
    summary_path = session_dir / "trajectory_summary.json"
    traj_path = session_dir / "trajectory_realworld.csv"
    bouts_path = session_dir / "movement_bouts.csv"

    if not (summary_path.exists() and traj_path.exists() and bouts_path.exists()):
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    traj = pd.read_csv(traj_path)
    mask_path = session_dir / MASK_FILENAME
    if mask_path.exists() and not traj.empty:
        try:
            mask_df = pd.read_csv(mask_path)
        except Exception:
            mask_df = pd.DataFrame()

        if not mask_df.empty:
            ignore = np.zeros(len(traj), dtype=bool)
            if "row_index" in mask_df.columns:
                idx = pd.to_numeric(mask_df["row_index"], errors="coerce").dropna().astype(int)
                idx = idx[(idx >= 0) & (idx < len(traj))]
                if len(idx) > 0:
                    ignore[idx.unique()] = True

            if "absolute_time" in mask_df.columns and "absolute_time" in traj.columns:
                # Fast path: exact string matching avoids expensive datetime parsing.
                m_abs_raw = mask_df["absolute_time"].astype("string").str.strip()
                m_abs_raw = m_abs_raw[m_abs_raw.notna() & (m_abs_raw != "")]
                if len(m_abs_raw) > 0:
                    mask_abs_set = set(m_abs_raw.unique().tolist())
                    if mask_abs_set:
                        t_abs_raw = traj["absolute_time"].astype("string").str.strip()
                        raw_hit = t_abs_raw.isin(mask_abs_set).to_numpy(dtype=bool)
                        ignore |= raw_hit

                        # Fallback parse only unresolved mask times (format mismatches).
                        unresolved = [x for x in mask_abs_set if x not in set(t_abs_raw.dropna().unique().tolist())]
                        if unresolved:
                            m_ts = pd.to_datetime(pd.Series(unresolved), errors="coerce", utc=True).dropna()
                            if len(m_ts) > 0:
                                t_ts = pd.to_datetime(traj["absolute_time"], errors="coerce", utc=True)
                                t_ns = t_ts.astype("int64").to_numpy()
                                m_ns = m_ts.astype("int64").to_numpy()
                                ignore |= np.isin(t_ns, m_ns)

            traj = traj.loc[~ignore].copy()

    needed = ["time_sec", "velocity_cm_s"]
    for col in needed:
        if col not in traj.columns:
            traj[col] = np.nan

    traj = traj.dropna(subset=["time_sec"])
    if traj.empty:
        return None

    vel_thresh = float(summary.get("velocity_threshold_cm_s", DEFAULT_VELOCITY_THRESHOLD_CM_S))
    summary_recomputed, bouts = recompute_summary_from_traj(traj, vel_thresh=vel_thresh, min_bout_dur=DEFAULT_BOUT_MIN_DURATION_SEC)
    summary.update(summary_recomputed)

    t_min = float(traj["time_sec"].min())
    t_max = float(traj["time_sec"].max())
    bin_edges = np.arange(math.floor(t_min / 60) * 60, math.ceil(t_max / 60) * 60 + 60, 60.0)
    if len(bin_edges) < 2:
        bin_edges = np.array([t_min, t_max + 60.0])
    time_min_bins = (bin_edges[:-1] + bin_edges[1:]) / 120.0

    bout_counts = np.histogram(bouts["start_sec"].to_numpy(dtype=float) if not bouts.empty else [], bins=bin_edges)[0]

    moving = traj.loc[traj["velocity_cm_s"] > DEFAULT_VELOCITY_THRESHOLD_CM_S, ["time_sec", "velocity_cm_s"]].copy()
    if moving.empty:
        moving_velocity_mean = np.zeros(len(time_min_bins), dtype=float)
    else:
        moving["bin"] = pd.cut(moving["time_sec"], bins=bin_edges, right=False, include_lowest=True)
        interval_index = pd.IntervalIndex.from_breaks(bin_edges, closed="left")
        moving_velocity_mean = (
            moving.groupby("bin", observed=False)["velocity_cm_s"]
            .mean()
            .reindex(interval_index, fill_value=np.nan)
            .to_numpy()
        )
        moving_velocity_mean = np.nan_to_num(moving_velocity_mean, nan=0.0)

    missing_spans = find_missing_spans(pd.to_numeric(traj["time_sec"], errors="coerce").dropna())

    return SessionReport(
        monkey_id=monkey_id,
        session_date=session_dir.name,
        session_dir=session_dir,
        summary=summary,
        time_min_bins=time_min_bins,
        bout_counts=bout_counts,
        moving_velocity_mean=moving_velocity_mean,
        missing_spans=missing_spans,
        bout_rows=bouts,
    )


def _load_session_task(args: tuple[Path, str]) -> SessionReport | None:
    session_dir, monkey_id = args
    return load_session(session_dir, monkey_id=monkey_id)


def collect_sessions(monkey_dir: Path, workers: int = 1) -> list[SessionReport]:
    session_dirs = sorted(p for p in monkey_dir.iterdir() if p.is_dir())
    if not session_dirs:
        return []

    if workers <= 1:
        sessions: list[SessionReport] = []
        for session_dir in session_dirs:
            report = load_session(session_dir, monkey_id=monkey_dir.name)
            if report is not None:
                sessions.append(report)
        return sessions

    tasks = [(session_dir, monkey_dir.name) for session_dir in session_dirs]
    sessions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for report in ex.map(_load_session_task, tasks):
            if report is not None:
                sessions.append(report)
    return sessions


def svg_histogram(
    x_min: float,
    x_max: float,
    bins_min: np.ndarray,
    values: np.ndarray,
    *,
    width: int = 900,
    height: int = 160,
    fill: str = "#2e7d32",
    stroke: str = "none",
    y_label: str = "",
    title: str = "",
    show_missing: list[tuple[float, float]] | None = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
    selected_range: tuple[float, float] | None = None,
) -> str:
    margin_left = 58
    margin_right = 18
    margin_top = 24
    margin_bottom = 28
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    if len(values) == 0:
        max_val = 1.0
    else:
        max_val = max(1.0, float(np.nanmax(values)))
        if threshold is not None:
            max_val = max(max_val, float(threshold))

    def x_to_px(x_minute: float) -> float:
        return margin_left + (x_minute - x_min) / (x_max - x_min + 1e-9) * plot_w

    def y_to_px(y_val: float) -> float:
        return margin_top + plot_h - (y_val / max_val) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" rx="10" ry="10"/>',
        f'<text x="{margin_left}" y="18" font-size="13" font-weight="600" fill="#1f1f1f">{html.escape(title)}</text>',
    ]

    # Background missing spans.
    if show_missing:
        for start_sec, end_sec in show_missing:
            x0 = x_to_px(start_sec / 60.0)
            x1 = x_to_px(end_sec / 60.0)
            parts.append(
                f'<rect x="{x0:.2f}" y="{margin_top}" width="{max(0.0, x1 - x0):.2f}" height="{plot_h}" fill="#d32f2f" opacity="0.12"/>'
            )

    # Grid + axes.
    for tick in np.linspace(0, max_val, 4):
        y = y_to_px(float(tick))
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="10" fill="#666">{tick:.0f}</text>')

    # Minute ticks.
    tick_minutes = np.arange(math.floor(x_min), math.ceil(x_max) + 1, 1)
    for tick in tick_minutes:
        x = x_to_px(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#efefef" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#666">{int(tick)}</text>')

    # Bars.
    for left_min, val in zip(bins_min, values):
        x0 = x_to_px(float(left_min))
        x1 = x_to_px(float(left_min) + (60.0 / 60.0))
        bar_w = max(1.0, x1 - x0 - 1.0)
        bar_h = (float(val) / max_val) * plot_h
        y0 = margin_top + plot_h - bar_h
        parts.append(f'<rect x="{x0 + 0.5:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{fill}" stroke="{stroke}" opacity="0.72"/>')

    if threshold is not None:
        ty = y_to_px(float(threshold))
        parts.append(f'<line x1="{margin_left}" y1="{ty:.2f}" x2="{width - margin_right}" y2="{ty:.2f}" stroke="#ef5350" stroke-dasharray="4,4" stroke-width="1.5"/>')
        if threshold_label:
            parts.append(f'<text x="{width - margin_right}" y="{ty - 4:.2f}" text-anchor="end" font-size="10" fill="#b71c1c">{html.escape(threshold_label)}</text>')

    if selected_range is not None:
        lo_min, hi_min = selected_range
        x0 = x_to_px(lo_min)
        x1 = x_to_px(hi_min)
        parts.append(f'<rect x="{x0:.2f}" y="{margin_top}" width="{max(0.0, x1 - x0):.2f}" height="{plot_h}" fill="#1565c0" opacity="0.10"/>')
        parts.append(f'<rect x="{x0:.2f}" y="{margin_top}" width="{max(0.0, x1 - x0):.2f}" height="{plot_h}" fill="none" stroke="#1565c0" stroke-width="1.4"/>')

    if y_label:
        parts.append(f'<text x="16" y="{margin_top + plot_h / 2:.2f}" font-size="10" fill="#666" transform="rotate(-90 16 {margin_top + plot_h / 2:.2f})">{html.escape(y_label)}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def session_card(report: SessionReport) -> str:
    summary = report.summary
    session_dir = report.session_dir
    lo = float(summary.get("start_sec", 0.0))
    hi = float(summary.get("end_sec", summary.get("total_duration_sec", 0.0)))
    selected_range_min = (lo / 60.0, hi / 60.0)
    x_min = float(report.time_min_bins[0]) if len(report.time_min_bins) else 0.0
    x_max = float(report.time_min_bins[-1] + 1.0) if len(report.time_min_bins) else 1.0

    bout_svg = svg_histogram(
        x_min=x_min,
        x_max=x_max,
        bins_min=report.time_min_bins,
        values=report.bout_counts,
        height=150,
        fill="#2e7d32",
        y_label="bout count",
        title="Movement bouts per minute",
        show_missing=report.missing_spans,
        selected_range=selected_range_min,
    )
    speed_svg = svg_histogram(
        x_min=x_min,
        x_max=x_max,
        bins_min=report.time_min_bins,
        values=report.moving_velocity_mean,
        height=150,
        fill="#1565c0",
        y_label="cm/s",
        title="Mean moving velocity per minute",
        show_missing=report.missing_spans,
        threshold=float(summary.get("velocity_threshold_cm_s", DEFAULT_VELOCITY_THRESHOLD_CM_S)),
        threshold_label=f"threshold {float(summary.get('velocity_threshold_cm_s', DEFAULT_VELOCITY_THRESHOLD_CM_S)):.1f} cm/s",
        selected_range=selected_range_min,
    )

    meta = (
        f"<div class='meta'>"
        f"<div><b>Date</b>: {html.escape(report.session_date)}</div>"
        f"<div><b>Duration</b>: {float(summary.get('total_duration_sec', 0.0)) / 60.0:.1f} min</div>"
        f"<div><b>Distance</b>: {float(summary.get('total_distance_cm', 0.0)) / 100.0:.1f} m</div>"
        f"<div><b>Immobility</b>: {float(summary.get('immobility_pct', 0.0)):.1f}%</div>"
        f"<div><b>Bouts</b>: {int(summary.get('n_movement_bouts', 0))}</div>"
        f"<div><b>Mean bout</b>: {float(summary.get('mean_bout_duration_sec', 0.0)):.2f} s</div>"
        f"<div><b>Active time</b>: {float(summary.get('total_active_time_sec', 0.0)):.1f} s</div>"
        f"<div><b>Max speed</b>: {float(summary.get('max_velocity_cm_s', 0.0)):.1f} cm/s</div>"
        f"<div><b>Window</b>: {float(summary.get('start_sec', 0.0)):.1f} → {float(summary.get('end_sec', 0.0)):.1f} s</div>"
        f"</div>"
    )

    return (
        f"<section class='card'>"
        f"<div class='card-head'>"
        f"<div>"
        f"<h2>{html.escape(report.session_date)}</h2>"
        f"<div class='path'>{html.escape(str(session_dir))}</div>"
        f"</div>"
        f"{meta}"
        f"</div>"
        f"<div class='charts'>"
        f"<div class='chart'>{bout_svg}</div>"
        f"<div class='chart'>{speed_svg}</div>"
        f"</div>"
        f"</section>"
    )


def build_html(monkey_id: str, sessions: list[SessionReport]) -> str:
    total_sessions = len(sessions)
    total_duration = sum(float(s.summary.get("total_duration_sec", 0.0)) for s in sessions)
    total_distance = sum(float(s.summary.get("total_distance_cm", 0.0)) for s in sessions)
    total_bouts = sum(int(s.summary.get("n_movement_bouts", 0)) for s in sessions)
    mean_immobility = float(np.mean([float(s.summary.get("immobility_pct", 0.0)) for s in sessions])) if sessions else 0.0

    summary_rows = []
    for s in sessions:
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(s.session_date)}</td>"
            f"<td>{float(s.summary.get('total_duration_sec', 0.0)) / 60.0:.1f}</td>"
            f"<td>{float(s.summary.get('total_distance_cm', 0.0)) / 100.0:.1f}</td>"
            f"<td>{int(s.summary.get('n_movement_bouts', 0))}</td>"
            f"<td>{float(s.summary.get('mean_bout_duration_sec', 0.0)):.2f}</td>"
            f"<td>{float(s.summary.get('total_active_time_sec', 0.0)):.1f}</td>"
            f"<td>{float(s.summary.get('immobility_pct', 0.0)):.1f}</td>"
            f"<td>{html.escape(s.session_dir.name)}</td>"
            "</tr>"
        )

    cards = "\n".join(session_card(s) for s in sessions)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(monkey_id)} OFT Session Comparison</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #ffffff;
      --ink: #1f1f1f;
      --muted: #666;
      --line: #d9d2c3;
      --accent: #1565c0;
      --green: #2e7d32;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f8f5ef 0%, #efebe3 100%);
    }}
    .wrap {{ max-width: 1560px; margin: 0 auto; padding: 24px; }}
    header {{
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 24px;
      backdrop-filter: blur(8px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    }}
    h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; line-height: 1.5; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .kpi {{ background: white; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .kpi .value {{ font-size: 20px; font-weight: 700; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; background: white; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee6d8; font-size: 13px; }}
    th {{ background: #fbfaf7; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    tr:last-child td {{ border-bottom: none; }}
    .cards {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 22px; }}
    .card {{ background: rgba(255,255,255,0.9); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 6px 20px rgba(0,0,0,0.04); }}
    .card-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 12px; flex-wrap: wrap; }}
    .card h2 {{ margin: 0; font-size: 22px; }}
    .path {{ color: var(--muted); font-size: 12px; margin-top: 4px; word-break: break-all; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 16px; font-size: 13px; min-width: 420px; }}
    .meta b {{ color: var(--ink); }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 14px; }}
    .chart svg {{ display: block; width: 100%; height: auto; }}
    footer {{ color: var(--muted); font-size: 12px; margin: 18px 4px 6px; }}
    @media (min-width: 1200px) {{
      .charts {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 900px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .meta {{ grid-template-columns: repeat(2, minmax(0, 1fr)); min-width: 0; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{html.escape(monkey_id)} OFT Session Comparison</h1>
      <div class="subtitle">
        Multi-session overview generated from <code>results/oft_analysis/{html.escape(monkey_id)}</code>.
        Each card shows 1-minute histograms of movement-bout counts and mean moving velocity, so you can compare days directly.
      </div>
      <div class="kpis">
        <div class="kpi"><div class="label">Sessions</div><div class="value">{total_sessions}</div></div>
        <div class="kpi"><div class="label">Total duration</div><div class="value">{total_duration / 60.0:.1f} min</div></div>
        <div class="kpi"><div class="label">Total distance</div><div class="value">{total_distance / 100.0:.1f} m</div></div>
        <div class="kpi"><div class="label">Total bouts</div><div class="value">{total_bouts}</div></div>
        <div class="kpi"><div class="label">Mean immobility</div><div class="value">{mean_immobility:.1f}%</div></div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Duration (min)</th>
            <th>Distance (m)</th>
            <th>Bouts</th>
            <th>Mean bout (s)</th>
            <th>Active time (s)</th>
            <th>Immobility (%)</th>
            <th>Folder</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </header>

    <main class="cards">
      {cards}
    </main>

    <footer>
      Generated from trajectory_summary.json, movement_bouts.csv, and trajectory_realworld.csv.
    </footer>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-session OFT HTML report")
    parser.add_argument("--monkey", required=True, help="Monkey ID folder under results/oft_analysis")
    parser.add_argument("--base-dir", type=Path, default=RESULTS_BASE, help="Base oft_analysis directory")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (defaults to <monkey>_session_report.html)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for session loading (0=auto, 1=serial)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monkey_dir = args.base_dir.expanduser().resolve() / args.monkey
    if not monkey_dir.exists():
        raise SystemExit(f"ERROR: monkey directory not found: {monkey_dir}")

    workers = int(args.workers)
    if workers == 0:
        cpu = os.cpu_count() or 2
        workers = max(1, min(8, cpu - 1))

    sessions = collect_sessions(monkey_dir, workers=workers)
    if not sessions:
        raise SystemExit(f"ERROR: no complete sessions found under {monkey_dir}")

    out_path = args.out.expanduser().resolve() if args.out else (monkey_dir / f"{args.monkey}_session_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(args.monkey, sessions), encoding="utf-8")
    print(f"[DONE] wrote {out_path}")
    print(f"[DONE] sessions included: {len(sessions)}")


if __name__ == "__main__":
    main()
