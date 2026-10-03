#!/usr/bin/env python3
"""Collect selected OFT time window summaries into a tab-delimited table.

Example usage:
    python scripts/gather_oft_trials.py --monkey H1803085
    python scripts/gather_oft_trials.py --monkey H1803085 --date 2026-07-20
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = PROJECT_ROOT / "results" / "oft_analysis"

OUTPUT_COLUMNS = [
    ("session_date", "date"),
    ("start_sec", "start_sec"),
    ("end_sec", "end_sec"),
    ("mean_velocity_moving_cm_s", "Mean velocity (moving; cm/s)"),
    ("avg_distance_cm_per_hour", "Session avg dist per hr (cm/h)"),
    ("distance_cm", "Distance traveled (cm)"),
    ("max_velocity_cm_s", "Max velocity (cm/s)"),
    ("n_movement_bouts", "Movement bouts"),
    ("mean_bout_duration_sec", "Mean bout duration (s)"),
    ("immobility_time_sec", "Immobility time (s)"),
    ("total_active_time_sec", "Total active time (s)"),
]

TRIAL_COLUMN_NAME = "trial"


def iter_window_files(monkey_id: str, session_date: str | None = None):
    base_dir = RESULTS_BASE / monkey_id
    if not base_dir.exists():
        raise FileNotFoundError(f"No folder found at: {base_dir}")

    date_dirs = [base_dir] if session_date is None else [base_dir / session_date]
    if session_date is None:
        date_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())

    for date_dir in date_dirs:
        if not date_dir.exists():
            continue
        for file_path in sorted(date_dir.glob("selected_time_window_[123].json")):
            yield file_path


def load_row(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row = {}
    for key, label in OUTPUT_COLUMNS:
        value = data.get(key)
        if key == "session_date":
            value = datetime.strptime(str(value), "%Y-%m-%d").strftime("%d-%b-%Y")
        row[label] = value

    raw_date = data.get("session_date")
    row["_sort_date"] = datetime.strptime(str(raw_date), "%Y-%m-%d")

    match = re.search(r"selected_time_window_(\d+)\.json$", json_path.name)
    if match:
        trial_value = int(match.group(1))
        row[TRIAL_COLUMN_NAME] = trial_value
        row["_sort_trial"] = trial_value
    else:
        row[TRIAL_COLUMN_NAME] = None
        row["_sort_trial"] = 0

    return row


def main():
    parser = argparse.ArgumentParser(description="Collect selected OFT time-window JSON summaries.")
    parser.add_argument("--monkey", default="H1803085", help="Monkey ID, e.g. H1803085")
    parser.add_argument("--date", default=None, help="Optional session date in YYYY-MM-DD format")
    parser.add_argument("--format", choices=["tsv", "csv"], default="tsv", help="Output delimiter format")
    parser.add_argument("--output", default=None, help="Optional output file path (CSV or TSV). If omitted, prints to stdout.")
    args = parser.parse_args()

    try:
        files = list(iter_window_files(args.monkey, args.date))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not files:
        target = f"{RESULTS_BASE / args.monkey}"
        if args.date:
            target = f"{target / args.date}"
        print(f"ERROR: no selected_time_window_[123].json files found under {target}", file=sys.stderr)
        raise SystemExit(1)

    rows = [load_row(path) for path in files]
    rows.sort(key=lambda r: (r["_sort_date"], r["_sort_trial"]))

    fieldnames = [label for _, label in OUTPUT_COLUMNS] + [TRIAL_COLUMN_NAME]
    delimiter = "," if args.format == "csv" else "\t"

    out = sys.stdout
    if args.output:
        out = open(args.output, "w", newline="", encoding="utf-8")

    try:
        writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(col, "") for col in fieldnames])
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
