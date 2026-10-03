#!/usr/bin/env python3
"""
Generate trial-level realworld trajectory heatmaps from selected_time_window_*.json.

For each session folder:
- reads trajectory_realworld.csv
- reads selected_time_window_1/2/3.json (or any selected_time_window_*.json)
- slices trajectory rows by [start_sec, end_sec]
- saves a heatmap PNG into the same folder

Example:
    python scripts/make_trial_heatmaps.py
    python scripts/make_trial_heatmaps.py --monkey H1803085
    python scripts/make_trial_heatmaps.py --monkey H1803085 --date 2026-09-03
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "oft_analysis"
WINDOW_RE = re.compile(r"selected_time_window_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate trial-level trajectory heatmaps.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root: results/oft_analysis")
    p.add_argument("--monkey", type=str, default=None, help="Optional monkey ID filter, e.g. H1803085")
    p.add_argument("--date", type=str, default=None, help="Optional date filter YYYY-MM-DD")
    p.add_argument("--gridsize", type=int, default=70, help="Hexbin grid size")
    p.add_argument("--dpi", type=int, default=220, help="Output DPI")
    p.add_argument(
        "--output-format",
        type=str,
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output image format (default: png)",
    )
    p.add_argument("--arena-min", type=float, default=0.0, help="Axis min in cm")
    p.add_argument("--arena-max", type=float, default=322.0, help="Axis max in cm")
    p.add_argument("--use-arena-limits", action="store_true", help="Force x/y limits to arena-min/arena-max")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name (default: viridis)")
    p.add_argument(
        "--density-scale",
        type=str,
        choices=["linear", "log"],
        default="log",
        help="Hexbin density scaling (default: log)",
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Clip upper hexbin intensities at this percentile (default: 99.5)",
    )
    p.add_argument(
        "--color-scale-mode",
        type=str,
        choices=["per-plot", "global"],
        default="per-plot",
        help="Use per-plot or shared global occupancy color scale (default: per-plot)",
    )
    return p.parse_args()


def list_session_dirs(root: Path, monkey: str | None, date: str | None) -> list[Path]:
    if monkey and date:
        d = root / monkey / date
        return [d] if d.exists() else []
    if monkey:
        mdir = root / monkey
        if not mdir.exists():
            return []
        return sorted([p for p in mdir.iterdir() if p.is_dir()])
    # all monkey/date folders
    out: list[Path] = []
    for mdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        out.extend(sorted([p for p in mdir.iterdir() if p.is_dir()]))
    return out


def trial_index_from_name(path: Path) -> int | None:
    m = WINDOW_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def load_windows(session_dir: Path) -> list[tuple[Path, dict]]:
    files = sorted(session_dir.glob("selected_time_window_*.json"), key=lambda p: (trial_index_from_name(p) or 999, p.name))
    out: list[tuple[Path, dict]] = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                d = json.load(f)
            out.append((fp, d))
        except Exception as e:
            print(f"[WARN] Failed to parse {fp}: {e}")
    return out


def load_traj(session_dir: Path) -> pd.DataFrame | None:
    traj_path = session_dir / "trajectory_realworld.csv"
    if not traj_path.exists():
        print(f"[WARN] Missing trajectory file: {traj_path}")
        return None
    try:
        df = pd.read_csv(traj_path)
    except Exception as e:
        print(f"[WARN] Failed reading {traj_path}: {e}")
        return None

    required = {"time_sec", "x_cm", "y_cm"}
    if not required.issubset(df.columns):
        print(f"[WARN] Missing required columns in {traj_path}. Need: {sorted(required)}")
        return None

    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["x_cm"] = pd.to_numeric(df["x_cm"], errors="coerce")
    df["y_cm"] = pd.to_numeric(df["y_cm"], errors="coerce")
    df = df.dropna(subset=["time_sec", "x_cm", "y_cm"]).copy()
    return df


def make_one_heatmap(
    traj: pd.DataFrame,
    win: dict,
    win_json_path: Path,
    session_dir: Path,
    gridsize: int,
    dpi: int,
    arena_min: float,
    arena_max: float,
    use_arena_limits: bool,
    cmap_name: str,
    density_scale: str,
    vmax_percentile: float,
    global_clim: tuple[float, float] | None,
    output_format: str,
) -> Path | None:
    start_sec = win.get("start_sec", None)
    end_sec = win.get("end_sec", None)

    if start_sec is None or end_sec is None:
        print(f"[WARN] start_sec/end_sec missing in {win_json_path.name}")
        return None

    try:
        start_sec = float(start_sec)
        end_sec = float(end_sec)
    except Exception:
        print(f"[WARN] Non-numeric start_sec/end_sec in {win_json_path.name}")
        return None

    if end_sec <= start_sec:
        print(f"[WARN] Invalid window in {win_json_path.name}: end <= start")
        return None

    sub = traj[(traj["time_sec"] >= start_sec) & (traj["time_sec"] <= end_sec)].copy()
    if sub.empty:
        print(f"[WARN] Empty trajectory slice for {win_json_path.name}")
        return None

    trial_idx = trial_index_from_name(win_json_path)
    trial_text = f"trial {trial_idx}" if trial_idx is not None else win_json_path.stem

    monkey_id = win.get("monkey_id", session_dir.parent.name)
    sess_date = win.get("session_date", session_dir.name)
    distance_cm = win.get("distance_cm", np.nan)
    immobile_sec = win.get("immobility_time_sec", np.nan)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.set_facecolor("#f4f4f4")

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("#f2f2f2")

    hb = ax.hexbin(
        sub["x_cm"].to_numpy(),
        sub["y_cm"].to_numpy(),
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        bins="log" if density_scale == "log" else None,
    )

    if global_clim is not None:
        hb.set_clim(vmin=float(global_clim[0]), vmax=float(global_clim[1]))
    else:
        # Robust clipping reduces domination by a few dense bins and improves contrast.
        counts = hb.get_array()
        if counts is not None and len(counts) > 0 and np.isfinite(vmax_percentile):
            vp = float(np.clip(vmax_percentile, 90.0, 100.0))
            vmax = float(np.percentile(counts, vp))
            vmin = float(np.nanmin(counts))
            if np.isfinite(vmax) and np.isfinite(vmin) and vmax > vmin:
                hb.set_clim(vmin=vmin, vmax=vmax)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Occupancy (log count)" if density_scale == "log" else "Occupancy count")

    if use_arena_limits:
        ax.set_xlim(arena_min, arena_max)
        ax.set_ylim(arena_min, arena_max)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    title = (
        f"{monkey_id}  {sess_date}  {trial_text}\n"
        f"{start_sec:.0f}-{end_sec:.0f}s | "
        f"distance={distance_cm:.1f} cm | immobility={immobile_sec:.1f} s"
    )
    ax.set_title(title, fontsize=10)

    out_name = f"trajectory_heatmap_{win_json_path.stem}.{output_format}"
    out_path = session_dir / out_name
    fig.tight_layout()
    save_kwargs = {}
    if output_format == "png":
        save_kwargs["dpi"] = dpi
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"ERROR: root not found: {root}")

    sessions = list_session_dirs(root, args.monkey, args.date)
    if not sessions:
        raise SystemExit("ERROR: no matching session folders found")

    total = 0
    made = 0
    tasks: list[tuple[Path, pd.DataFrame, Path, dict, pd.DataFrame]] = []

    for sdir in sessions:
        windows = load_windows(sdir)
        if not windows:
            continue

        traj = load_traj(sdir)
        if traj is None:
            continue

        for win_json_path, win in windows:
            start_sec = win.get("start_sec")
            end_sec = win.get("end_sec")
            try:
                start_sec = float(start_sec)
                end_sec = float(end_sec)
            except Exception:
                print(f"[WARN] Non-numeric start_sec/end_sec in {win_json_path.name}")
                continue
            if end_sec <= start_sec:
                print(f"[WARN] Invalid window in {win_json_path.name}: end <= start")
                continue
            sub = traj[(traj["time_sec"] >= start_sec) & (traj["time_sec"] <= end_sec)].copy()
            if sub.empty:
                print(f"[WARN] Empty trajectory slice for {win_json_path.name}")
                continue
            tasks.append((sdir, traj, win_json_path, win, sub))

    if not tasks:
        raise SystemExit("ERROR: no valid selected windows with trajectory samples were found")

    global_clim: tuple[float, float] | None = None
    if args.color_scale_mode == "global":
        all_vals: list[np.ndarray] = []
        cmap = plt.get_cmap(args.cmap).copy()
        cmap.set_under("#f2f2f2")
        for _, _, _, _, sub in tasks:
            fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
            hb_tmp = ax_tmp.hexbin(
                sub["x_cm"].to_numpy(),
                sub["y_cm"].to_numpy(),
                gridsize=args.gridsize,
                cmap=cmap,
                mincnt=1,
                bins="log" if args.density_scale == "log" else None,
            )
            arr = hb_tmp.get_array()
            if arr is not None and len(arr) > 0:
                all_vals.append(np.asarray(arr, dtype=float))
            plt.close(fig_tmp)

        if all_vals:
            merged = np.concatenate(all_vals)
            vp = float(np.clip(float(args.vmax_percentile), 90.0, 100.0))
            vmin = float(np.nanmin(merged))
            vmax = float(np.percentile(merged, vp))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                global_clim = (vmin, vmax)
                print(f"[INFO] Global color scale enabled: vmin={vmin:.4g}, vmax={vmax:.4g}")
            else:
                print("[WARN] Failed to compute valid global color scale; falling back to per-plot scaling")

    for sdir, traj, win_json_path, win, _ in tasks:
            total += 1
            out = make_one_heatmap(
                traj=traj,
                win=win,
                win_json_path=win_json_path,
                session_dir=sdir,
                gridsize=args.gridsize,
                dpi=args.dpi,
                arena_min=args.arena_min,
                arena_max=args.arena_max,
                use_arena_limits=args.use_arena_limits,
                cmap_name=args.cmap,
                density_scale=args.density_scale,
                vmax_percentile=float(args.vmax_percentile),
                global_clim=global_clim,
                output_format=args.output_format,
            )
            if out is not None:
                made += 1
                print(f"[OK] {out}")

    print(f"\nDone. Generated {made}/{total} heatmaps.")


if __name__ == "__main__":
    main()