# DeepLabCut Training Workflow

**Automate monkey tracking using your manual labels with DeepLabCut.**

---

## Overview

You've manually labeled **~4,200 frames** across multiple OFT videos. This is enough data to train a DeepLabCut neural network to automatically track the monkey in new videos.

**Benefits:**
- ⚡ **Automate labeling** — DLC can label new videos at ~60 FPS
- 🎯 **High accuracy** — Typically >95% with your data quantity
- 📈 **Incremental improvement** — Add more manual labels if needed
- 🔄 **Transfer learning** — Fine-tune on new sessions

---

## Quick Start

### 1. Install DeepLabCut

```bash
# Use the dlc environment (already set up with GPU support)
conda activate dlc

# DeepLabCut 3.0.0 is already installed
```

### 2. Select Label Files

**Before setting up the project, select which CSV files contain your manual labels:**

```bash
python scripts/select_label_files.py
```

This opens a GUI where you can:
- ✅ **Preview** each CSV file to see columns and sample data
- ✅ **Select/deselect** files to include in training
- ✅ **Auto-select valid files** — automatically finds files with required columns
- ✅ **Save selection** — stores your choices for future use

**Required columns for valid label files:**
- `video_file`
- `centroid_x`
- `centroid_y`
- `monkey_visible`

**What to exclude:**
- Metadata files (e.g., `video_metadata.csv`)
- Trajectory fusion files
- Files without labels

**Your selection is saved to:** `deeplabcut_label_config.json`

### 3. Set Up DLC Project

This converts your selected manual labels to DeepLabCut format:

```bash
python scripts/setup_deeplabcut_project.py
```

**What this does:**
- Loads CSV files from your saved selection
- Extracts labeled frames as images
- Creates DeepLabCut project structure
- Converts labels to DLC's HDF5 format
- Generates config.yaml

**⚠️ Note:** Frame extraction may take 10-30 minutes due to HEVC video format requiring sequential reading. Be patient!

**Output:**
```
deeplabcut_projects/
└── monkey_oft_tracking-luolab-2026-06-15/
    ├── config.yaml
    ├── labeled-data/
    │   ├── D03_20260608144224/
    │   │   ├── frame_000000.png
    │   │   ├── frame_000020.png
    │   │   └── ...
    │   └── ...
    └── training-datasets/
        └── iteration-0/
            ├── CollectedData_luolab.h5
            └── CollectedData_luolab.csv
```

### 4. Train the Network

**Option A: Automated script**
```bash
python scripts/train_deeplabcut.py
```

**Option B: Manual (more control)**
```python
import deeplabcut

config = 'deeplabcut_projects/monkey_oft_tracking-luolab-2026-06-15/config.yaml'

# Create training dataset
deeplabcut.create_training_dataset(config, augmenter_type='imgaug')

# Train (takes 2-6 hours on GPU, longer on CPU)
deeplabcut.train_network(
    config,
    shuffle=1,
    displayiters=100,    # Show progress every 100 iterations
    saveiters=5000,      # Save checkpoint every 5000 iterations
    maxiters=100000,     # Max iterations
)
```

**Training time:**
- **GPU (NVIDIA):** 2-4 hours for 50k iterations
- **CPU:** 8-24 hours (not recommended)

### 4. Evaluate Performance

```python
import deeplabcut

config = 'deeplabcut_projects/monkey_oft_tracking-luolab-2026-06-15/config.yaml'

# Evaluate on test set (5% of your data)
deeplabcut.evaluate_network(config, plotting=True)
```

**Metrics to check:**
- **Train error:** Should be <3 pixels
- **Test error:** Should be <5 pixels  
- **PCK (% correct keypoints):** Should be >95%

### 5. Analyze New Videos

```python
import deeplabcut

config = 'deeplabcut_projects/monkey_oft_tracking-luolab-2026-06-15/config.yaml'

# Analyze single video
video_path = 'data/oft_rhp/H1705011/2026-06-07/D04_20260607132621.mp4'
deeplabcut.analyze_videos(config, [video_path], save_as_csv=True)

# Analyze all videos in a folder
deeplabcut.analyze_videos(config, ['data/oft_rhp/H1705011/2026-06-07/'], 
                         videotype='mp4', save_as_csv=True)
```

**Output:** CSV file with predictions at every frame:
```csv
bodyparts,centroid,centroid
coords,x,y
0,1920.5,1080.2
1,1925.3,1082.1
...
```

### 6. Visualize Results

Create labeled videos to check quality:

```python
deeplabcut.create_labeled_video(config, [video_path], 
                                draw_skeleton=False,
                                trailpoints=10)
```

This creates a video with the predicted centroid overlaid.

---

## Advanced: Active Learning Workflow

If DLC performance isn't good enough, iteratively improve:

### 1. Extract Outlier Frames

Find frames where DLC is uncertain:

```python
deeplabcut.extract_outlier_frames(config, [video_path], 
                                 outlieralgorithm='jump',
                                 epsilon=20)  # jump threshold in pixels
```

### 2. Label Outliers Manually

```python
deeplabcut.refine_labels(config)  # Opens GUI for labeling
```

Or use your custom tool:
```bash
# Extract frames where DLC prediction differs from expected
python scripts/extract_difficult_frames.py --video path/to/video.mp4 \
                                           --dlc-csv path/to/predictions.csv \
                                           --threshold 50  # pixels

# Label with your tool
python scripts/manual_labeler.py --camera D3 --video difficult_frames.mp4
```

### 3. Merge New Labels

```python
deeplabcut.merge_datasets(config)
```

### 4. Retrain

```python
# Create new training dataset with merged labels
deeplabcut.create_training_dataset(config, augmenter_type='imgaug')

# Train from previous checkpoint
deeplabcut.train_network(config, shuffle=1)
```

---

## Integration with OFT Pipeline

### Replace Manual Labeling

After DLC is trained, use it instead of `manual_labeler.py`:

```python
# dlc_labeler.py (new script)
import deeplabcut
import pandas as pd

def analyze_with_dlc(video_path, config_path, output_csv):
    """Analyze video with DLC and save in manual_labeler format."""
    
    # Run DLC
    deeplabcut.analyze_videos(config_path, [video_path], 
                             save_as_csv=True, destfolder=output_csv.parent)
    
    # Convert DLC CSV to manual_labeler format
    dlc_csv = output_csv.parent / (video_path.stem + 'DLC_resnet50_...' + '.csv')
    df_dlc = pd.read_csv(dlc_csv, header=[0,1,2])
    
    # Extract x, y coordinates
    df = pd.DataFrame({
        'camera': 'D3',  # from video filename
        'video_file': video_path.name,
        'video_frame': range(len(df_dlc)),
        'centroid_x': df_dlc[('luolab', 'centroid', 'x')],
        'centroid_y': df_dlc[('luolab', 'centroid', 'y')],
        'monkey_visible': True,
        'likelihood': df_dlc[('luolab', 'centroid', 'likelihood')],
    })
    
    # Filter low-confidence predictions
    df = df[df['likelihood'] > 0.9]
    
    df.to_csv(output_csv, index=False)
    return df
```

### Workflow:

1. **Label a few videos manually** (~500-1000 frames per camera)
2. **Train DLC** on these labels
3. **Use DLC for remaining videos** (much faster)
4. **Manual spot-check** difficult frames
5. **Retrain** if needed with additional labels

---

## Tips for Best Results

### Data Quality

✅ **Good practices:**
- Label diverse poses (moving, stationary, different locations)
- Include frames from different lighting conditions
- Label full range of arena coverage
- ~200-500 frames per camera is often enough

❌ **Avoid:**
- Labeling only similar frames (e.g., all monkey in center)
- Mislabeled frames (confuses the network)
- Too few labels (<100 frames)

### Training

- **Use GPU** if available (10-20× faster)
- **Monitor training loss** — should decrease steadily
- **Stop if overfitting** — test error increases while train error decreases
- **Data augmentation** helps with small datasets

### Inference

- **Batch processing** is faster than one video at a time
- **Set likelihood threshold** (0.9) to filter low-confidence predictions
- **Check videos with `create_labeled_video`** before trusting predictions
- **Use temporal filtering** to smooth jittery predictions

---

## Troubleshooting

### Training is slow
- Check if GPU is being used: `nvidia-smi`
- Reduce batch size if GPU memory error
- Use smaller network: `mobilenet_v2_1.0` instead of `resnet_50`

### Poor accuracy (<90%)
- Add more diverse training labels
- Increase augmentation strength
- Train longer (more iterations)
- Check for labeling errors in your data

### DLC predictions are jittery
- Apply temporal smoothing:
  ```python
  deeplabcut.filterpredictions(config, [video_path])
  ```
- Increase likelihood threshold to filter outliers

### Out of memory error
- Reduce batch size in `config.yaml` (batch_size: 4 or 2)
- Use smaller image size
- Close other GPU applications

---

## Expected Performance

With your **~4,200 labeled frames**:

| Metric | Expected Value |
|--------|---------------|
| Training error | 2-4 pixels |
| Test error | 3-6 pixels |
| PCK@10px | >95% |
| Inference speed | 30-60 FPS (GPU) |
| Inference speed | 2-5 FPS (CPU) |

For 4K video (3840×2160), 3-6 pixel error = **0.1-0.2% of image width** — excellent for centroid tracking!

---

## Next Steps

1. **Run setup script:** `python scripts/setup_deeplabcut_project.py`
2. **Train network:** `python scripts/train_deeplabcut.py`
3. **Evaluate results:** Check test error and PCK
4. **Apply to new videos:** Use `analyze_videos()` on unlabeled sessions
5. **Integrate with fusion:** Feed DLC predictions to `fuse_trajectories.py`

---

**Questions?** Check the [DeepLabCut docs](http://www.mackenziemathislab.org/deeplabcut) or ask!
