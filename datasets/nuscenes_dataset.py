"""
nuScenes Dataset Loader for DriveWeaver

Loads video sequences from nuScenes dataset for slot extraction.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class NuScenesVideoDataset(Dataset):
    """
    NuScenes dataset for video-level slot extraction.

    Each sample is a complete scene with all frames from a single camera.

    Shape Convention:
    ----------------
    __getitem__ returns:
        video: [T, 3, H, W] - full scene frames
        metadata: dict with scene info
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        camera: str = "CAM_FRONT",
        input_size: int = 518,
        n_val_scenes: int = 150,
        max_scenes: Optional[int] = None,
        normalize: bool = False,
    ):
        """
        Args:
            dataroot: Path to nuScenes dataset
            version: Dataset version
            split: "train" or "val"
            camera: Camera to use (default: CAM_FRONT)
            input_size: Resize frames to this size
            n_val_scenes: Number of scenes for validation split
            max_scenes: Limit number of scenes (for debugging)
            normalize: Whether to normalize frames (usually False, done in wrapper)
        """
        super().__init__()

        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.camera = camera
        self.input_size = input_size
        self.normalize = normalize

        # Load nuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Split scenes into train/val
        all_scenes = self.nusc.scene
        train_scenes = all_scenes[:-n_val_scenes]
        val_scenes = all_scenes[-n_val_scenes:]

        if split == "train":
            self.scenes = train_scenes
        elif split == "val":
            self.scenes = val_scenes
        else:
            raise ValueError(f"Invalid split: {split}")

        # Limit scenes if requested
        if max_scenes is not None:
            self.scenes = self.scenes[:max_scenes]

        print(f"Loaded {len(self.scenes)} scenes for split=\"{split}\"")

        # Transforms
        self.resize = T.Resize((input_size, input_size), antialias=True)
        self.to_tensor = T.ToTensor()

        if normalize:
            self.norm = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.norm = None

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict:
        """Get a complete scene video."""
        scene = self.scenes[idx]
        scene_name = scene["name"]
        scene_token = scene["token"]

        frames = []
        timestamps = []
        sample_tokens = []
        ego_poses = []

        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            sd_token = sample["data"][self.camera]
            sd = self.nusc.get("sample_data", sd_token)

            img_path = os.path.join(self.dataroot, sd["filename"])
            img = Image.open(img_path).convert("RGB")
            img = self.resize(img)
            img_t = self.to_tensor(img)

            if self.norm is not None:
                img_t = self.norm(img_t)

            frames.append(img_t)
            timestamps.append(sample["timestamp"])
            sample_tokens.append(sample_token)

            ego_pose = self._get_ego_pose(sample)
            ego_poses.append(ego_pose)

            sample_token = sample["next"]

        video = torch.stack(frames, dim=0)
        timestamps = np.array(timestamps, dtype=np.float64)
        ego_poses = np.array(ego_poses, dtype=np.float32)

        return {
            "video": video,
            "scene_name": scene_name,
            "scene_token": scene_token,
            "n_frames": len(frames),
            "timestamps": timestamps,
            "sample_tokens": sample_tokens,
            "ego_poses": ego_poses,
        }

    def _get_ego_pose(self, sample: dict) -> np.ndarray:
        """Extract ego pose for a sample."""
        # Get ego pose token from sample data (not directly from sample)
        sd_token = sample["data"][self.camera]
        sd = self.nusc.get("sample_data", sd_token)
        ego_pose_token = sd["ego_pose_token"]
        ego_pose = self.nusc.get("ego_pose", ego_pose_token)

        translation = ego_pose["translation"]
        x, y, z = translation

        rotation = Quaternion(ego_pose["rotation"])
        yaw = rotation.yaw_pitch_roll[0]

        speed = 0.0  # Placeholder

        return np.array([x, y, yaw, speed], dtype=np.float32)
