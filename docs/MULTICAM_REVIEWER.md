# MULTICAM Reviewer Guide

Last updated: 2026-07-18

This guide documents the interactive review workflow for synchronized multi-camera predictions and fused trajectory QC.

## Purpose

Use the reviewer to:

- inspect 4-camera detections in time sync,
- correct missing or bad boxes,
- mark false-positive detections as no-detection,
- inspect and mask trajectory artifacts,
- refresh trajectory overlays after edits.

## Launch

```bash
conda activate video_tracking
python scripts/review_multicam_predictions.py --monkey H1705011 --date 2026-07-01
```

Optional artifact tuning:

```bash
python scripts/review_multicam_predictions.py \
  --monkey H1705011 \
  --date 2026-07-01 \
  --artifact-velocity-thresh 120 \
  --artifact-dvdt-thresh 300 \
  --artifact-bridge-max-sec 8.0
```

## Data Inputs

- Predictions CSVs:
  - results/predictions/<monkey>/<date>/D0X_*_preds.csv
- Frame directories:
  - tmp/rtdetr_frames_2hz/<monkey>/<date>/D0X_*_downsampled/
- Trajectory CSV:
  - results/oft_analysis/<monkey>/<date>/trajectory_realworld.csv
- Optional precomputed artifact candidates:
  - results/oft_analysis/<monkey>/<date>/trajectory_artifact_candidates.csv
- Optional mask file written by reviewer:
  - results/oft_analysis/<monkey>/<date>/trajectory_ignore_mask.csv

## Core Controls

- Left / Right: move timeline by one synchronized step
- Shift + Left / Right: jump by larger step
- 1 / 2 / 3 / 4: select camera(s) for editing
- 0: clear selected cameras
- U: set selected prediction row(s) to no-detection (has_detection=false)
- I: toggle trajectory ignore mask at current trajectory time
- A: toggle artifact navigation mode
- R: refresh bundles, trajectory, and overlays from disk
- Q or Esc: quit

## Editing Behavior Notes

- U is non-destructive:
  - it does not delete rows,
  - it sets has_detection=false.
- At clip boundaries, U applies to all clip-local matches at the same absolute time to avoid frame-0 leftovers in the next clip.
- Non-detection rows with bbox coordinates are shown in yellow for QC visibility.
- Detection rows are shown in green.

## Mouse Box Editing

- Select exactly one camera panel.
- Click and drag inside that panel to draw/adjust bbox.
- Release to write updated bbox coordinates back to the underlying prediction CSV row.

## Time Jump

- Enter HH:MM:SS in the sidebar time field.
- Jump is relative to session trajectory time base.

## Artifact Mode

- Artifact mode focuses timeline navigation on artifact-candidate timepoints.
- Candidate source priority:
  1. trajectory_artifact_candidates.csv (preferred; precomputed in reconstruction)
  2. fallback on-the-fly computation in reviewer
- Fallback and reconstruction both support interval bridging:
  - seed events from velocity and |dV/dt| thresholds,
  - bridge intervals between nearby seed events within artifact-bridge-max-sec,
  - captures persistent artifact plateaus between abrupt transitions.

## Recommended Workflow

1. Run reconstruct_trajectory.py to generate trajectory and artifact candidates.
2. Open reviewer and inspect artifact mode first.
3. Use U/I and mouse edits to clean predictions and trajectory.
4. Press R after major edits.
5. Re-run reconstruct_trajectory.py.
6. Generate multi-session report.

## Troubleshooting

- If startup is slow:
  - ensure trajectory_artifact_candidates.csv exists for the session.
- If edits appear inconsistent across clip boundaries:
  - use R to force full reload,
  - verify that all D0X CSVs are writable.
- If report output seems unchanged:
  - confirm trajectory_ignore_mask.csv was written,
  - rerun reconstruct/report scripts after review.