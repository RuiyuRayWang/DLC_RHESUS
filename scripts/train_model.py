#!/usr/bin/env python3
"""
Train DeepLabCut model with optimized settings for single-point tracking.

This script:
1. Creates training dataset (if not exists)
2. Optimizes training configuration for faster training
3. Starts training with progress updates
"""

import deeplabcut
from pathlib import Path
import yaml
import torch
import argparse

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'deeplabcut_projects' / 'monkey_oft_tracking_D3-luolab-2026-06-19' / 'config.yaml'


def parse_args():
    parser = argparse.ArgumentParser(description='Train DLC model with quick/full modes.')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                        help='quick: fast convergence check (~20 epochs), full: full run (~50 epochs)')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='Training crop size (default: 384, was 448)')
    return parser.parse_args()

def find_existing_shuffles(dlc_models):
    """Find all existing training datasets."""
    import re
    iter_dir = dlc_models / 'iteration-0'
    if not iter_dir.exists():
        return []
    
    shuffles = []
    for folder in iter_dir.iterdir():
        if folder.is_dir():
            shuffle_match = re.search(r'shuffle(\d+)', folder.name)
            if shuffle_match:
                shuffle_num = int(shuffle_match.group(1))
                pytorch_config = folder / 'train' / 'pytorch_config.yaml'
                if pytorch_config.exists():
                    # Check for checkpoints
                    train_dir = folder / 'train'
                    checkpoints = list(train_dir.glob('snapshot-*.pt'))
                    learning_stats = train_dir / 'learning_stats.csv'
                    
                    # Get last epoch from learning stats if available
                    last_epoch = 0
                    if learning_stats.exists():
                        with open(learning_stats, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                last_line = lines[-1].strip()
                                if last_line:
                                    last_epoch = int(last_line.split(',')[0])
                    
                    shuffles.append({
                        'num': shuffle_num,
                        'folder': folder.name,
                        'config': pytorch_config,
                        'checkpoints': len(checkpoints),
                        'last_epoch': last_epoch
                    })
    
    return sorted(shuffles, key=lambda x: x['num'])

def main():
    args = parse_args()

    print("=" * 70)
    print("Train DeepLabCut Model - Optimized for Single-Point Tracking")
    print("=" * 70)
    
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    
    print(f"\nConfig: {CONFIG_PATH}")
    
    # Step 1: Check for existing training datasets
    dlc_models = CONFIG_PATH.parent / 'dlc-models-pytorch'
    existing_shuffles = find_existing_shuffles(dlc_models) if dlc_models.exists() else []
    
    if existing_shuffles:
        print("\n📂 Found existing training datasets:")
        for s in existing_shuffles:
            checkpoint_info = f" ({s['checkpoints']} checkpoints" if s['checkpoints'] > 0 else " (no checkpoints"
            epoch_info = f", epoch {s['last_epoch']}" if s['last_epoch'] > 0 else ""
            checkpoint_info += epoch_info + ")"
            print(f"   • Shuffle {s['num']}: {s['folder']}{checkpoint_info}")
        
        # Use the most recent shuffle with checkpoints, or the last one
        shuffles_with_checkpoints = [s for s in existing_shuffles if s['checkpoints'] > 0]
        if shuffles_with_checkpoints:
            selected_shuffle = shuffles_with_checkpoints[-1]
            print(f"\n✅ Resuming from shuffle {selected_shuffle['num']} (has {selected_shuffle['checkpoints']} checkpoints)")
        else:
            selected_shuffle = existing_shuffles[-1]
            print(f"\n✅ Using existing shuffle {selected_shuffle['num']}")
        
        pytorch_config = selected_shuffle['config']
        shuffle_num = selected_shuffle['num']
        shuffle_folder = selected_shuffle['folder']
        
    else:
        print("\n📦 Creating new training dataset...")
        print("   This may take a few minutes...")
        
        try:
            deeplabcut.create_training_dataset(
                str(CONFIG_PATH),
                augmenter_type='imgaug'
            )
            print("   ✅ Training dataset created")
        except Exception as e:
            print(f"   ❌ Error creating training dataset: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Find the newly created shuffle
        existing_shuffles = find_existing_shuffles(dlc_models)
        if not existing_shuffles:
            print("   ❌ No training dataset found after creation!")
            return
        
        selected_shuffle = existing_shuffles[-1]
        pytorch_config = selected_shuffle['config']
        shuffle_num = selected_shuffle['num']
        shuffle_folder = selected_shuffle['folder']
        print(f"   Created shuffle {shuffle_num}: {shuffle_folder}")
    
    # Step 2: Optimize configuration
    print("\n⚙️  Optimizing training configuration...")
    print(f"   Using shuffle: {shuffle_num}")
    
    with open(pytorch_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Save original values
    original_epochs = cfg['train_settings']['epochs']
    original_batch = cfg['train_settings']['batch_size']

    if args.mode == 'quick':
        target_epochs = 20
        milestones = [12, 18]
    else:
        target_epochs = 50
        milestones = [30, 45]
    
    # Optimize for faster training
    cfg['train_settings']['epochs'] = target_epochs
    cfg['train_settings']['batch_size'] = 16  # Increased from 8
    cfg['runner']['eval_interval'] = 5  # More frequent evaluation
    cfg['runner']['snapshots']['save_epochs'] = 10  # Save every 10 epochs

    # Smaller crops speed up training and can improve focus on local head context.
    if 'data' in cfg and 'train' in cfg['data'] and 'crop_sampling' in cfg['data']['train']:
        cfg['data']['train']['crop_sampling']['width'] = args.crop_size
        cfg['data']['train']['crop_sampling']['height'] = args.crop_size
    
    # Adjust LR schedule to selected training mode
    if 'scheduler' in cfg['runner']:
        cfg['runner']['scheduler']['params']['milestones'] = milestones
    
    with open(pytorch_config, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"   Mode: {args.mode}")
    print(f"   Epochs: {original_epochs} → {target_epochs}")
    print(f"   Batch size: {original_batch} → 16 (2x faster per epoch)")
    print(f"   Crop size: {args.crop_size}x{args.crop_size}")
    print(f"   Eval interval: Every 5 epochs")
    print(f"   LR milestones: {milestones}")
    if args.mode == 'quick':
        print(f"   ✅ Estimated training time: short validation run")
    else:
        print(f"   ✅ Estimated training time: ~10-15 hours")
    
    # Step 3: Verify training data and check for checkpoints
    print("\n📊 Verifying training data...")
    train_dir = pytorch_config.parent
    train_data_file = train_dir / 'train_data.pickle'
    
    if train_data_file.exists():
        import pickle
        with open(train_data_file, 'rb') as f:
            data = pickle.load(f)
        n_train = len(data[0])
        n_test = len(data[1])
        print(f"   ✅ Training samples: {n_train}")
        print(f"   ✅ Test samples: {n_test}")
    else:
        print(f"   ⚠️  Warning: Could not find train_data.pickle")
    
    # Check for existing checkpoints
    checkpoints = list(train_dir.glob('snapshot-*.pt'))
    if checkpoints:
        # Get the latest checkpoint by number
        checkpoint_nums = []
        for cp in checkpoints:
            import re
            match = re.search(r'snapshot-(\d+|best-\d+)\.pt', cp.name)
            if match:
                if 'best' in match.group(1):
                    num = int(match.group(1).split('-')[1])
                    checkpoint_nums.append((num, cp, 'best'))
                else:
                    num = int(match.group(1))
                    checkpoint_nums.append((num, cp, 'regular'))
        
        if checkpoint_nums:
            # Use the highest epoch checkpoint (prefer regular over best)
            checkpoint_nums.sort(key=lambda x: (x[0], x[2] == 'regular'), reverse=True)
            latest_epoch, latest_checkpoint, cp_type = checkpoint_nums[0]
            print(f"   🔄 Found checkpoint: {latest_checkpoint.name}")
            print(f"   Will resume from epoch {latest_epoch}")
    else:
        print(f"   ℹ️  No checkpoints found - starting from scratch")
    
    # Step 4: Check GPU
    print("\n🖥️  Hardware check...")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ CUDA optimizations enabled")
    else:
        print(f"   ⚠️  No GPU detected - training will be very slow!")
    
    # Step 5: Start training
    print("\n🚀 Starting training...")
    if checkpoints:
        print("   ℹ️  DeepLabCut will automatically resume from the latest checkpoint")
    print("   Progress: Updates every 100 iterations")
    print("   Snapshots: Saved every 10 epochs")
    print("   Press Ctrl+C to stop (last checkpoint will be saved)")
    print("\n" + "=" * 70)
    
    try:
        deeplabcut.train_network(
            str(CONFIG_PATH),
            shuffle=shuffle_num,
            displayiters=100,
            allow_growth=True
        )
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print("\n📊 Next steps:")
        print("1. Evaluate model performance:")
        print(f"   deeplabcut.evaluate_network(r'{CONFIG_PATH}', plotting=True)")
        print("\n2. Analyze new videos:")
        print(f"   deeplabcut.analyze_videos(r'{CONFIG_PATH}', ['path/to/video.mp4'])")
        print("\n3. Create labeled videos:")
        print(f"   deeplabcut.create_labeled_video(r'{CONFIG_PATH}', ['path/to/video.mp4'])")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        print("   Last checkpoint saved - run this script again to resume")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
