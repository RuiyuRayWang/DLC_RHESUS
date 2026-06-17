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

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'deeplabcut_projects' / 'monkey_oft_tracking-luolab-2026-06-15' / 'config.yaml'

def main():
    print("=" * 70)
    print("Train DeepLabCut Model - Optimized for Single-Point Tracking")
    print("=" * 70)
    
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    
    print(f"\nConfig: {CONFIG_PATH}")
    
    # Step 1: Create training dataset
    print("\n📦 Creating training dataset...")
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
    
    # Step 2: Find the created shuffle and optimize configuration
    print("\n⚙️  Optimizing training configuration...")
    
    dlc_models = CONFIG_PATH.parent / 'dlc-models-pytorch'
    if not dlc_models.exists():
        print("   ❌ dlc-models-pytorch directory not created!")
        return
    
    pytorch_configs = list(dlc_models.glob('iteration-0/*/train/pytorch_config.yaml'))
    if not pytorch_configs:
        print("   ❌ No pytorch_config.yaml found!")
        print(f"   Checked: {dlc_models / 'iteration-0'}")
        # List what's actually there
        iter_dir = dlc_models / 'iteration-0'
        if iter_dir.exists():
            print(f"   Contents: {list(iter_dir.iterdir())}")
        return
    
    pytorch_config = pytorch_configs[0]
    
    # Extract shuffle number from path
    shuffle_folder = pytorch_config.parent.parent.name
    import re
    shuffle_match = re.search(r'shuffle(\d+)', shuffle_folder)
    shuffle_num = int(shuffle_match.group(1)) if shuffle_match else 1
    print(f"   Found training dataset: {shuffle_folder}")
    print(f"   Using shuffle: {shuffle_num}")
    
    with open(pytorch_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Save original values
    original_epochs = cfg['train_settings']['epochs']
    original_batch = cfg['train_settings']['batch_size']
    
    # Optimize for faster training
    cfg['train_settings']['epochs'] = 50  # Reduced from 200
    cfg['train_settings']['batch_size'] = 16  # Increased from 8
    cfg['runner']['eval_interval'] = 5  # More frequent evaluation
    cfg['runner']['snapshots']['save_epochs'] = 10  # Save every 10 epochs
    
    # Adjust LR schedule for 50 epochs
    if 'scheduler' in cfg['runner']:
        cfg['runner']['scheduler']['params']['milestones'] = [30, 45]
    
    with open(pytorch_config, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"   Epochs: {original_epochs} → 50 (4x faster)")
    print(f"   Batch size: {original_batch} → 16 (2x faster per epoch)")
    print(f"   Eval interval: Every 5 epochs")
    print(f"   LR milestones: [30, 45]")
    print(f"   ✅ Estimated training time: ~10-15 hours")
    
    # Step 3: Verify training data
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
        print("   Last checkpoint saved - you can resume training later")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
