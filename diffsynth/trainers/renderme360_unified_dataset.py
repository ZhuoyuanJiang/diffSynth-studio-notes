"""
RenderMe360 Dataset Wrapper for UnifiedDataset
Combines UnifiedDataset CSV loading with custom RenderMe360 operators
"""

import torch
import pandas as pd
from pathlib import Path
from .renderme360_operators import (
    LoadRenderMe360GridVideo,
    LoadRenderMe360InputImage,
    LoadRenderMe360Audio,
    LoadRenderMe360Prompt,
)


class RenderMe360UnifiedDataset(torch.utils.data.Dataset):
    """
    RenderMe360 dataset using custom operators.

    Compatible with upstream training scripts while maintaining custom data loading logic.

    Args:
        base_path: Path to /ssd4/zhuoyuan/renderme360_4cam/
        metadata_path: Path to metadata CSV
        repeat: Dataset repetitions per epoch
        cameras: List of 4 camera names for grid
    """

    def __init__(self, base_path, metadata_path, repeat=1, cameras=None):
        self.base_path = base_path
        self.repeat = repeat
        self.load_from_cache = False  # We always load from metadata CSV, not cache

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        self.data = [self.metadata.iloc[i].to_dict() for i in range(len(self.metadata))]

        # Initialize operators
        self.video_operator = LoadRenderMe360GridVideo(base_path, cameras)
        self.input_image_operator = LoadRenderMe360InputImage(base_path)
        self.audio_operator = LoadRenderMe360Audio(base_path)
        self.prompt_operator = LoadRenderMe360Prompt()

        print(f"[RenderMe360UnifiedDataset] Loaded {len(self.data)} samples × {repeat} repeats = {len(self)} total")

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, idx):
        # Get row data (with repeat support)
        row = self.data[idx % len(self.data)].copy()

        # Apply operators (each receives the full row dict)
        result = {
            "video": self.video_operator(row),
            "input_image": self.input_image_operator(row),
            "input_audio": self.audio_operator(row),
            "prompt": self.prompt_operator(row),
            "audio_sample_rate": 16000,  # Fixed for RenderMe360
        }

        # Include original metadata (useful for debugging)
        result["metadata"] = {
            "subject": row.get("subject", ""),
            "performance": row.get("performance", ""),
            "start_frame_30fps": row.get("start_frame_30fps", 0),
            "num_frames": row.get("num_frames", 81),
        }

        return result
