# OFT Pipeline Status Snapshot

**Project:** PAL_ENHANCER OFT tracking  
**Environment:** `video_tracking`  
**Updated:** 2026-06-13

## Current state

The macOS OFT pipeline is operational end-to-end:

1. GUI calibration and manual labeling
2. Anchor-based multi-video time alignment
3. Real-world trajectory fusion
4. Window selection for quantification

## Stable workflow

Use this sequence for each session:

1. Label videos (`manual_labeler.py`)
2. Generate `label_time_anchors.csv` (`generate_time_anchor_template.py`)
3. Curate `anchor_timestamp` manually (VSPlayer)
4. Fuse trajectory (`fuse_trajectories.py`)
5. Select analysis window (`select_time_window.py`)

Primary reference:
- `docs/MULTI_VIDEO_WORKFLOW.md`

## Key implementation notes

- Paths are repository-relative (`PROJECT_ROOT` style), no Linux hardcoded paths.
- Fusion uses anchor-based absolute time per file.
- Projection cleanup is active (camera-region filtering and arena clipping).
- Selector displays window metrics including distance, max velocity, movement bouts, mean bout duration, and active time.

## Core outputs per session

- `trajectory_realworld.csv`
- `movement_bouts.csv`
- `trajectory_summary.json`
- `selected_time_window.json`

## Open items

- Keep anchor timestamps curated and auditable (`method`, `notes`).
- Re-run notebook visual QC when fusion parameters change.
- Optional: evaluate additional QC summaries for projection filtering.

## Sync reminder

```bash
git fetch origin
git checkout macos-port
git pull --ff-only origin/macos-port
```
