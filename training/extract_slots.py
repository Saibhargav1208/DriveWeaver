"""
Slot Extraction Pipeline for DriveWeaver Phase 1

Extracts temporally consistent object-centric slots from nuScenes using VideoSAUR.
"""

import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import sys
sys.path.insert(0, '/work')

from datasets.nuscenes_dataset import NuScenesVideoDataset
from models.videosaur_wrapper import VideoSAURWrapper


class SlotExtractor:
    """Pipeline for extracting slots from nuScenes dataset."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Initialize VideoSAUR
        print("=" * 80)
        print("Initializing VideoSAUR Wrapper")
        print("=" * 80)
        self.videosaur = VideoSAURWrapper(
            checkpoint_path=cfg.videosaur.checkpoint_path,
            config_path=cfg.videosaur.config_path,
            n_slots=cfg.videosaur.n_slots,
            device=str(self.device),
            videosaur_path=cfg.videosaur.videosaur_path,
        )

        print(f"\nSlot Extractor initialized on {self.device}")
        print(f"  - N_slots: {cfg.videosaur.n_slots}")
        print(f"  - Slot dim: {self.videosaur.slot_dim}")
        print(f"  - Input size: {self.videosaur.input_size}\n")

    def extract_dataset(self, split: str = "train") -> Dict[str, Dict]:
        """Extract slots for entire split."""
        print("=" * 80)
        print(f"Extracting {split} split")
        print("=" * 80)

        # Create dataset
        dataset = NuScenesVideoDataset(
            dataroot=self.cfg.data.dataroot,
            version=self.cfg.data.version,
            split=split,
            camera=self.cfg.data.camera,
            input_size=self.videosaur.input_size,
            n_val_scenes=self.cfg.data.n_val_scenes,
            max_scenes=self.cfg.data.max_scenes,
            normalize=False,
        )

        results = {}

        # Process each scene
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split}"):
            sample = dataset[idx]

            scene_name = sample["scene_name"]
            video = sample["video"]  # [T, 3, H, W]

            # Add batch dimension: [1, T, 3, H, W]
            video_batch = video.unsqueeze(0)

            # Extract slots - CRITICAL: Full video in ONE forward pass
            slots = self.videosaur.extract_slots_numpy(video_batch, normalize=True)

            # Remove batch dimension: [1, T, N, D] -> [T, N, D]
            slots = slots[0]

            # Store with metadata
            results[scene_name] = {
                "slots": slots,
                "timestamps": sample["timestamps"],
                "sample_tokens": sample["sample_tokens"],
                "ego_poses": sample["ego_poses"],
            }

            if idx == 0:
                print(f"\n  Example shape: {slots.shape}")
                print(f"  Scene: {scene_name}")
                print(f"  Frames: {len(sample['sample_tokens'])}")

        print(f"\n  Extracted {len(results)} scenes for {split}\n")
        return results

    def run(self):
        """Run full extraction pipeline."""
        print("\n" + "=" * 80)
        print("DriveWeaver Phase 1: Slot Extraction")
        print("=" * 80 + "\n")

        # Extract train split
        train_results = self.extract_dataset(split="train")

        # Extract val split
        val_results = self.extract_dataset(split="val")

        # Combine results
        results = {
            "train": train_results,
            "val": val_results,
        }

        # Save to disk
        output_path = Path(self.cfg.output.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"Saving to {output_path}")
        print("=" * 80)

        with open(output_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Print summary
        print("\nExtraction Complete!")
        print(f"  Train scenes: {len(train_results)}")
        print(f"  Val scenes: {len(val_results)}")
        print(f"  Output: {output_path}")

        if train_results:
            example_scene = list(train_results.keys())[0]
            example_slots = train_results[example_scene]["slots"]
            print(f"\n  Example slot shape: {example_slots.shape}")

        return results


def extract_slots(cfg: DictConfig) -> Dict:
    """Main extraction function."""
    extractor = SlotExtractor(cfg)
    return extractor.run()
