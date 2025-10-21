#!/usr/bin/env python3
"""
Initial training script for SpeechLM with distributed training support.
This script handles argument parsing and distributed training setup.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

import torch
import deepspeed

from espnet2.speechlm.template import JOB_TEMPLATES
from espnet2.speechlm.dataloader.iterator import DataIteratorFactory


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="SpeechLM Distributed Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Distributed training arguments (handled by torchrun)
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training (set by torchrun)",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--train-config",
        type=Path,
        required=False,
        help="Path to training configuration file",
    )
    train_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/train"),
        help="Directory to save checkpoints and logs",
    )
    train_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--train-unregistered-specifier",
        type=str,
        default="",
        required=False,
        help="Unregistered train data specifier. "
        "Format: 'task:name:data_json[:factor]' "
        "(e.g., 'asr:librispeech:train.json:2.0')",
    )
    data_group.add_argument(
        "--train-registered-specifier",
        type=str,
        default="",
        required=False,
        help="Registered train data specifier. "
        "Format: 'task:name[:factor]' "
        "(e.g., 'tts:ljspeech:1.5')",
    )
    data_group.add_argument(
        "--valid-unregistered-specifier",
        type=str,
        default="",
        required=False,
        help="Unregistered validation data specifier. "
        "Format: 'task:name:data_json[:factor]' "
        "(e.g., 'asr:librispeech:valid.json')",
    )
    data_group.add_argument(
        "--valid-registered-specifier",
        type=str,
        default="",
        required=False,
        help="Registered validation data specifier. "
        "Format: 'task:name[:factor]' "
        "(e.g., 'tts:ljspeech:1.0')",
    )
    data_group.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="The folder of length statistics",
    )

    # Logging configuration
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) Setup distributed training first to get rank info
    deepspeed.init_distributed()

    assert torch.distributed.is_initialized()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # (2) Setup logging with rank-aware configuration
    log_format = (
        f"[Rank {rank}/{world_size}] "
        "%(asctime)s (%(module)s:%(lineno)d) "
        "%(levelname)s: %(message)s"
    )

    if rank == 0:
        log_level = args.log_level
    else:
        log_level = "CRITICAL"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    # Now we can log freely - only rank 0 will show INFO messages
    logger.info(f"Distributed training initialized")
    logger.info(f"World size: {world_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # (3) Initialize job template
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    logger.info(f"Loaded training config from: {args.train_config}")

    job_template_class = JOB_TEMPLATES[train_config['job_type']]
    job_template = job_template_class(train_config)

    # (4) build data iterator factory
    loading_config = train_config['data_loading']
    preprocessor = job_template.build_preprocessor()
    
    train_iterator_factory = DataIteratorFactory(
        args.train_unregistered_specifier,
        args.train_registered_specifier,
        stats_dir=args.stats_dir,
        loader_state=args.output_dir / "loader_state" / "train.json",
        collate_fn=preprocessor.collate_fn,
        batchfy_method=loading_config['batchfy_method'],
        batch_size=loading_config['batch_size'],
        num_workers=loading_config['num_workers'],
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=loading_config['seed'],
    )
    

    # valid_iterator_factory = DataIteratorFactory(
    #     args.valid_unregistered_specifier,
    #     args.valid_registered_specifier,
    #     stats_dir=args.stats_dir,
    #     loader_state=args.output_dir / "loader_state" / "valid.json",
    #     collate_fn=preprocessor.collate_fn,
    #     batchfy_method=loading_config['batchfy_method'],
    #     batch_size=loading_config['batch_size'],
    #     num_workers=loading_config['num_workers'],
    #     rank=rank,
    #     world_size=world_size,
    #     shuffle=False,
    #     seed=loading_config['seed'],
    # )

    # DEBUG: Test the data loading.
    train_iterator = train_iterator_factory.get_iterator()
    for batch in train_iterator:
        print(batch)
        assert 1 == 2
    
if __name__ == "__main__":
    main()
