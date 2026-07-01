# Data Directory

**Purpose:** Store raw and processed video data for OFT (Open Field Test) analysis.

**Note:** This folder's contents are excluded from git due to large file sizes. Data files must be synced separately or accessed from the original source.

---

## Directory Structure

```
data/
├── oft_rhp/                   # OFT videos for rhesus macaques
│   ├── H1705011/              # Monkey ID
│   │   └── 2026-05-25/        # Session date
│   │       ├── D01_*.mp4      # Camera D1 video (top-left quadrant)
│   │       ├── D02_*.mp4      # Camera D2 video (top-right quadrant)
│   │       ├── D03_*.mp4      # Camera D3 video (bottom-left quadrant)
│   │       └── D04_*.mp4      # Camera D4 video (bottom-right quadrant)
│   └── H1803085/               # Another monkey (if applicable)
|
├── videos_downsampled         # Downsampled videos (scripts/downsample_ffmpeg.py)
|   ├── H1705011/              # Monkey ID
│   │   └── 2026-05-25/        # Session date
│   │       ├── D01_*_downsampled.mp4      # Downsampled camera D1 video (top-left quadrant)
│   │       ├── D02_*_downsampled.mp4      # Downsampled camera D2 video (top-right quadrant)
│   │       ├── D03_*_downsampled.mp4      # Downsampled camera D3 video (bottom-left quadrant)
│   │       └── D04_*_downsampled.mp4      # Downsampled camera D4 video (bottom-right quadrant)
│   └── H1803085/               # Another monkey (if applicable)
|
└-- [other datasets]
``` 

---

## File Naming Convention

### OFT Videos
- **Pattern:** `D0{camera}_{YYYYMMDDHHMMSS}.mp4`
- **Example:** `D02_20260525161934.mp4`
  - Camera: D02 (camera 2, top-right)
  - Date: 2026-05-25
  - Time: 16:19:34

### Video Specifications
- **Format:** HEVC (H.265) in MP4 container
- **Resolution:** 3840×2160 (4K UHD)
- **Frame Rate:** 20 FPS
- **Codec:** HEVC/H.265 Main profile
- **Source:** Hikvision cameras (hardware-synced)

**Important:** These videos require sequential reading due to HEVC's reference frame structure. See [docs/VIDEO_LOADING_SUMMARY.md](../docs/VIDEO_LOADING_SUMMARY.md) for details.

---

## Data Changelog

### 2026-06-09
- **Session:** H1705011 / 2026-05-25
- **Videos available:** D01, D02, D03, D04
- **Status:** All four cameras calibrated

---

## Data Sources

### Where to Find Original Data
- **Primary location:** [Fill in network path or server location]
- **Backup location:** [Fill in backup location]
- **External drive:** [If applicable]

### How to Sync Data to This Folder
```bash
# Example rsync from data server
rsync -avz --progress user@server:/path/to/oft_videos/ data/oft_rhp/

# Or copy from external drive
cp -r /Volumes/ExternalDrive/OFT_DATA/* data/oft_rhp/
```

---

## Storage Requirements

### Estimated Sizes per Session
- **Raw videos (4 cameras, 20-30 min each):** ~40-80 GB
- **Extracted frames (if saved):** ~10-20 GB
- **Processed results:** ~100-500 MB

### Current Usage
```bash
# Check data folder size
du -sh data/
```

---

## Data Management Best Practices

1. **Keep originals safe:** Never modify raw video files directly
2. **Document everything:** Update this README when adding new sessions
3. **Backup regularly:** Maintain at least 2 copies of raw data
4. **Clean up intermediates:** Remove temporary extracted frames after processing
5. **Use compression wisely:** Videos are already compressed (HEVC)

---

## Related Documentation

- [Video loading guide](../docs/VIDEO_LOADING_SUMMARY.md) - Technical details on HEVC video handling
- [Pipeline overview](../docs/OFT_PIPELINE_PROGRESS.md) - Full analysis workflow
- [Calibration guide](../docs/QUICKSTART_GUI.md) - How to calibrate cameras

---

**Last updated:** 2026-07-01
