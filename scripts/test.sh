#!/bin/bash
#SBATCH -J test
#SBATCH -p q_cn
#SBATCH -o test.%j.out
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -e err.%j.out
#SBATCH --mem-per-cpu=4G

module purge
source /GPFS/luominmin_lab_permanent/wangruiyu/environments/miniconda3/bin/activate dlc
cd ~/DATA/GITHUB_REPO/DLC_RHESUS

python scripts/downsample_ffmpeg.py\
 --video-path data/oft_rhp/H1705011/2026-05-25/D01_20260525161829.mp4\
 --raw-video-root data/oft_rhp\
 --output-dir data/oft_rhp/videos_downsampled\
 --scale 0.5\
 --threads 32\
 --downsample-csv
