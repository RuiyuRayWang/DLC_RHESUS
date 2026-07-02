# Quick Start Guide - OFT Video Analysis

## Setup Complete! ✅

We've successfully tested and configured the video analysis pipeline for monkey 134569R.

## What We Verified

### 1. Video Format
- ✅ **Codec**: HEVC (H.265) - confirmed working with OpenCV
- ✅ **Resolution**: 3840x2160 (4K)
- ✅ **Frame Rate**: 20 FPS
- ✅ **Sequential reading**: Works perfectly
- ⚠️  **Random access**: Has HEVC decoding issues (expected, not needed)

### 2. Environment
```bash
conda env: deeplabcut
python: 3.10
opencv: 4.13.0
ffmpeg: 8.1.1
```

## Next Steps

### Step 1: Test the Notebook

Open and run the analysis notebook:
```bash
cd /home/luolab/PROJECTS/DLC_RHESUS
conda activate deeplabcut
jupyter lab notebooks/oft_video_analysis.ipynb
```

Run cells 1-4 to verify:
- Video loading works
- Frame extraction is successful
- Sample frames look correct

### Step 2: Install Additional Dependencies

If not already installed:
```bash
conda activate deeplabcut
pip install tqdm pandas matplotlib seaborn scikit-image
```

### Step 3: Run Pilot Analysis

Test on a few videos first (cells 1-9 in the notebook):
1. Load video metadata
2. Extract and visualize frames
3. Test segmentation on sample frames
4. Calculate basic motion metrics
5. Review results

### Step 4: Optimize Segmentation

The notebook includes a classical CV segmentation method. You may need to adjust parameters:
- Threshold values
- Morphological kernel sizes
- Minimum area thresholds

Test on different videos/lighting conditions.

### Step 5: Batch Processing

Once segmentation works well:
1. Set `PROCESS_ALL = True` in cell 8
2. Or adjust `N_VIDEOS_TO_PROCESS` for batches
3. Run overnight for all 172 videos

## File Structure

```
PAL_ENHANCER/
├── data/
│   └── oft_rhp/
│       └── 134569R/          # 172 video files
├── notebooks/
│   └── oft_video_analysis.ipynb  # Main analysis notebook
├── scripts/
│   ├── test_video_loading.py     # Video compatibility test
│   └── oft_behavioral_metrics.py # (to be created)
├── results/
│   └── oft_analysis/         # Output directory (auto-created)
│       ├── video_metadata.csv
│       ├── behavioral_metrics_all_videos.csv
│       ├── tracking_data_all_videos.csv
│       └── *.png figures
└── docs/
    └── VIDEO_LOADING_SUMMARY.md  # Technical details
```

## Key Parameters to Adjust

In the notebook configuration cell:

```python
FPS = 20              # Video frame rate
SUBSAMPLE = 10        # Process every 10th frame (2 Hz effective rate)
                       # Increase for faster processing, decrease for more detail
```

For segmentation:

```python
# In segment_monkey_classical()
velocity_threshold = 5     # Pixels/frame for motion detection
min_area_threshold = 1000  # Minimum object size
```

## Expected Processing Time

**Per video** (~365 seconds, with subsample=10):
- Sequential reading: ~30-60 seconds
- Segmentation + tracking: ~2-3 minutes total
- **All 172 videos**: ~6-8 hours

## Troubleshooting

### Issue: "Could not open video"
```bash
# Verify ffmpeg is in PATH
which ffmpeg

# Reinstall if needed
conda install -c conda-forge ffmpeg opencv -y
```

### Issue: HEVC decoding errors
- **Expected** if using random access (cap.set())
- Our notebook uses sequential reading - should not see this
- If you see errors, verify you're using the updated notebook

### Issue: Segmentation not working well
- Check sample frames first - is monkey visible?
- Adjust lighting/contrast preprocessing
- Try different threshold values
- Consider color-based segmentation for better contrast

### Issue: Out of memory
- Increase SUBSAMPLE value (e.g., 15 or 20)
- Process fewer frames per video
- Process videos in smaller batches
- Close other applications

## Performance Tips

1. **Subsample appropriately**: 
   - subsample=10 → 2 Hz (recommended)
   - subsample=20 → 1 Hz (faster, less detail)
   
2. **Resize frames if needed**:
   ```python
   frame = cv2.resize(frame, (1920, 1080))  # Half resolution
   ```

3. **Process in batches**:
   - Group by day
   - Process overnight
   - Save results incrementally

## Output Files

After processing, you'll have:

1. **video_metadata.csv**: All video files with timestamps
2. **behavioral_metrics_all_videos.csv**: Aggregated metrics per video
3. **tracking_data_all_videos.csv**: Frame-by-frame tracking data
4. **Figures**: Trajectories, motion plots, temporal trends

## Questions or Issues?

- See `docs/VIDEO_LOADING_SUMMARY.md` for technical details
- Test script: `python scripts/test_video_loading.py`
- Check OpenCV version: `python -c "import cv2; print(cv2.__version__)"`

## Ready to Start!

```bash
# Activate environment
conda activate dlc

# Launch Jupyter
cd /home/luolab/PROJECTS/DLC_RHESUS
jupyter lab

# Open: notebooks/oft_video_analysis.ipynb
# Run cells sequentially
```

Good luck with your analysis! 🐒📹📊
