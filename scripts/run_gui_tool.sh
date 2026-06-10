#!/bin/bash
# Test script to demonstrate correct usage of GUI tools
# 2026-06-09

set -e  # Exit on error

echo "========================================"
echo "OFT Video Analysis Pipeline - GUI Tools"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ ERROR: conda not found. Please install Anaconda/Miniconda."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "video_tracking"; then
    echo "❌ ERROR: 'video_tracking' conda environment not found."
    echo "   Create it with:"
    echo "   bash scripts/setup_video_tracking_env.sh"
    exit 1
fi

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "video_tracking" ]]; then
    echo "⚠️  WARNING: 'video_tracking' environment is NOT activated."
    echo ""
    echo "   GUI tools require an ACTIVATED conda environment (not 'conda run')."
    echo "   This is due to Qt6 threading limitations."
    echo ""
    echo "   Please run:"
    echo "   conda activate video_tracking"
    echo "   $0"
    echo ""
    exit 1
fi

echo "✅ Conda environment 'video_tracking' is activated."
echo ""

# Show available commands
echo "Available commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Camera Calibration (one camera):"
echo "   python scripts/calibrate_cameras.py --camera D2 --frame-sec 30"
echo ""
echo "2. Camera Calibration (all cameras):"
echo "   python scripts/calibrate_cameras.py --all --frame-sec 30"
echo ""
echo "3. Manual Labeling (one camera):"
echo "   python scripts/manual_labeler.py --camera D3"
echo ""
echo "4. Trajectory Fusion (no GUI, processes labeled data):"
echo "   python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25"
echo ""
echo "5. Display Diagnostic (checks OpenCV GUI):"
echo "   python scripts/check_opencv_display.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Ask user what they want to do
read -p "Enter command number to run (or 'q' to quit): " choice

case $choice in
    1)
        read -p "Enter camera (D1-D4): " cam
        read -p "Enter frame time in seconds [30]: " fsec
        fsec=${fsec:-30}
        python scripts/calibrate_cameras.py --camera "$cam" --frame-sec "$fsec"
        ;;
    2)
        read -p "Enter frame time in seconds [30]: " fsec
        fsec=${fsec:-30}
        python scripts/calibrate_cameras.py --all --frame-sec "$fsec"
        ;;
    3)
        read -p "Enter camera (D1-D4): " cam
        read -p "Enter sample-every stride [20]: " stride
        stride=${stride:-20}
        python scripts/manual_labeler.py --camera "$cam" --sample-every "$stride"
        ;;
    4)
        read -p "Enter monkey ID [H1705011]: " monkey
        monkey=${monkey:-H1705011}
        read -p "Enter date [2026-05-25]: " date
        date=${date:-2026-05-25}
        python scripts/fuse_trajectories.py --monkey "$monkey" --date "$date"
        ;;
    5)
        python scripts/check_opencv_display.py
        ;;
    q|Q)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
