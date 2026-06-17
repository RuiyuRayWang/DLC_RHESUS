# GPU Setup for DeepLabCut Training

**Issue:** NVIDIA driver version 11.4 is too old for PyTorch 2.12 (requires CUDA 13.0)

---

## ✅ Option 1: Reinstall PyTorch for CUDA 11.8 (Recommended)

**No root access required. Quick fix.**

Your NVIDIA driver (version 11.4) supports CUDA 11.x. Reinstall PyTorch compiled for CUDA 11.8:

```bash
cd /home/luolab/PROJECTS/PAL_ENHANCER
bash scripts/fix_pytorch_cuda.sh
```

**Or manually:**
```bash
conda activate dlc

# Uninstall PyTorch with CUDA 13.0
pip uninstall -y torch torchvision triton

# Install PyTorch with CUDA 11.8 (compatible with your driver)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Expected result:**
```
CUDA: True
GPU: [Your GPU name]
```

Then verify with:
```bash
python scripts/verify_dlc_installation.py
```

---

## Option 2: Update NVIDIA Driver (Requires Root)

**Only if Option 1 fails or you want the latest features.**

### Check Current Driver

```bash
nvidia-smi
```

Current: **Driver Version: 470.x** (supports CUDA 11.4)  
Needed: **Driver Version: 525+** (supports CUDA 12.x+)

### Update Driver (Ubuntu/Debian)

```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver (requires sudo)
sudo ubuntu-drivers autoinstall
# Or specific version:
sudo apt install nvidia-driver-525

# Reboot
sudo reboot
```

### After Reboot

```bash
# Verify driver
nvidia-smi

# Should show Driver Version: 525+ or higher
```

Then verify GPU in Python:
```bash
conda activate dlc
python scripts/verify_dlc_installation.py
```

---

## Troubleshooting

### GPU Still Not Detected After Option 1

1. **Check driver version:**
   ```bash
   nvidia-smi
   ```
   If command not found → drivers not installed

2. **Check CUDA compatibility:**
   ```bash
   python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
   nvidia-smi | grep "CUDA Version"
   ```
   PyTorch CUDA version must be ≤ driver CUDA version

3. **Reinstall with correct CUDA version:**
   
   For driver 11.4 (CUDA 11.4):
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   For driver 12.x (CUDA 12.x):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Check What CUDA Version Your Driver Supports

```bash
nvidia-smi | grep "CUDA Version"
```

Output example:
```
CUDA Version: 11.4
```

Match PyTorch CUDA version to be ≤ this value.

### GPU Memory Errors During Training

If you get out-of-memory errors:

1. **Reduce batch size** in `config.yaml`:
   ```yaml
   batch_size: 4  # or 2 for very limited memory
   ```

2. **Check available memory:**
   ```bash
   nvidia-smi
   ```

3. **Close other GPU applications**

---

## Performance Comparison

| Setup | Training Speed | 100k iterations |
|-------|---------------|-----------------|
| **CPU only** | ~100 it/min | ~16 hours |
| **GPU (CUDA 11.8)** | ~1000 it/min | ~2 hours |
| **GPU (CUDA 13.0)** | ~1100 it/min | ~1.5 hours |

**Recommendation:** Option 1 is sufficient. The 10% performance difference between CUDA 11.8 and 13.0 is negligible.

---

## Verification Commands

After fixing GPU:

```bash
conda activate dlc

# 1. Quick check
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 2. Full verification
python scripts/verify_dlc_installation.py

# 3. GPU memory check
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('GPU not available')
"
```

Expected output:
```
CUDA available: True
GPU: [Your GPU Name]
Memory: [XX.XX] GB
```

---

## Training Without GPU (Fallback)

If GPU setup fails, you can still train on CPU (slower but works):

```bash
python scripts/train_deeplabcut.py
```

DLC will automatically use CPU if CUDA is unavailable. Training will take ~10-20 hours instead of 2-4 hours.

---

## Next Steps

Once GPU is working:

1. ✅ Verify installation: `python scripts/verify_dlc_installation.py`
2. Set up DLC project: `python scripts/setup_deeplabcut_project.py`
3. Train network: `python scripts/train_deeplabcut.py`

GPU will significantly speed up training!
