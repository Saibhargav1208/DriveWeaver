#!/usr/bin/env python3
"""
DriveWeaver Phase 1: Extract Slots from nuScenes

Main script for extracting temporally consistent object-centric slots.

Usage:
    # Default config
    python scripts/extract_nuscenes_slots.py

    # Debug mode (5 scenes only)
    python scripts/extract_nuscenes_slots.py --config-name debug

    # Override parameters
    python scripts/extract_nuscenes_slots.py \
        data.max_scenes=10 \
        videosaur.n_slots=15 \
        output.save_path=/work/data/slots/custom.pkl
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from training.extract_slots import extract_slots


@hydra.main(version_base=None, config_path="../configs/extraction", config_name="default")
def main(cfg: DictConfig):
    """Main entry point."""

    # Print configuration
    print("=" * 80)
    print("DriveWeaver Phase 1: Slot Extraction")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80 + "\n")

    # Run extraction
    results = extract_slots(cfg)

    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
