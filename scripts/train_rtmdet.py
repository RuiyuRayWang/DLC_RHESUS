#!/usr/bin/env python3
"""
RTMDet training script for monkey bbox detection using Hugging Face Accelerate.
Supports distributed training on multi-GPU systems with automatic device detection.

Usage:
    # Single GPU (auto-detected)
    python scripts/train_rtmdet.py --config configs/rtmdet_m_monkey.py --work-dir results/rtmdet_models/

    # Multi-GPU with Accelerate (8 GPUs)
    accelerate launch --multi_gpu --num_processes=8 scripts/train_rtmdet.py \
        --config configs/rtmdet_m_monkey.py --work-dir results/rtmdet_models/

    # Or with config file
    accelerate launch scripts/train_rtmdet.py \
        --config configs/rtmdet_m_monkey.py --work-dir results/rtmdet_models/

    # With mixed precision
    accelerate launch --mixed_precision=fp16 scripts/train_rtmdet.py \
        --config configs/rtmdet_m_monkey.py --work-dir results/rtmdet_models/
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerate import Accelerator, DistributedDataParallelKwargs
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RTMDet for monkey bbox detection")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--work-dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cfg-options", nargs="+", help="Override config options (e.g., key1=value1 key2=value2)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Initialize Accelerator with DDP backend
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Log device info on main process
    if accelerator.is_main_process:
        logger.info(f"Running on {accelerator.num_processes} process(es)")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Distributed: {accelerator.distributed_type}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Apply config overrides
    if args.cfg_options:
        cfg_dict = ConfigDict()
        for option in args.cfg_options:
            if "=" not in option:
                raise ValueError(f"Invalid config option: {option}. Use key=value format")
            key, value = option.split("=", 1)
            # Try to convert value to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            cfg_dict[key] = value
        cfg.merge_from_dict(cfg_dict)
    
    # Update work directory
    cfg.work_dir = args.work_dir
    if accelerator.is_main_process:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Work directory: {args.work_dir}")
    
    # Synchronize processes before creating runner
    accelerator.wait_for_everyone()
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    
    # Prepare model for distributed training
    runner.model = accelerator.prepare(runner.model)
    
    # Prepare data loaders
    if hasattr(runner, 'train_dataloader') and runner.train_dataloader:
        runner.train_dataloader = accelerator.prepare(runner.train_dataloader)
    if hasattr(runner, 'val_dataloader') and runner.val_dataloader:
        runner.val_dataloader = accelerator.prepare(runner.val_dataloader)
    
    # Resume from checkpoint if specified
    if args.resume:
        accelerator.load_state(args.resume)
        if accelerator.is_main_process:
            logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Log config on main process
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("Training Configuration")
        logger.info("="*60)
        logger.info(f"Model: {cfg.model.type if hasattr(cfg, 'model') else 'N/A'}")
        logger.info(f"Max epochs: {cfg.train_cfg.max_epochs if hasattr(cfg, 'train_cfg') else 'N/A'}")
        logger.info(f"Batch size per GPU: {cfg.train_dataloader.batch_size if hasattr(cfg, 'train_dataloader') else 'N/A'}")
        logger.info("="*60 + "\n")
    
    # Start training
    runner.train()
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info(f"Best checkpoint saved to: {args.work_dir}")
        logger.info("="*60)


if __name__ == "__main__":
    main()
