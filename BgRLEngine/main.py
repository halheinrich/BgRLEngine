"""BgRLEngine — main entry point for training.

Usage:
    python -m bgrle.main [--config CONFIG_PATH] [--max-games N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch


def get_device(config_device: str = "auto") -> torch.device:
    """Determine the torch device to use.

    Args:
        config_device: "auto", "cpu", or "cuda".

    Returns:
        torch.device for training.
    """
    if config_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using CUDA: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device(config_device)
        print(f"Using device: {device}")

    return device


def load_config(path: str | Path) -> dict:
    """Load training configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="BgRLEngine Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of training games",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for checkpoints",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}

    # Setup
    device = get_device(config.get("device", "auto"))
    output_dir = Path(args.output_dir)

    # Set random seed if specified
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Random seed: {seed}")

    # Import here to avoid circular imports
    from training.td_trainer import Trainer

    trainer = Trainer(config, device, output_dir)
    stats = trainer.train(max_games=args.max_games)

    print(f"\nFinal skill score: Level {stats.levels_reached}")


if __name__ == "__main__":
    main()