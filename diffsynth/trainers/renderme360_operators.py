"""
RenderMe360 Custom Data Operators for UnifiedDataset
Loads 4-camera views and creates 2×2 grid on-the-fly
"""

import torch
import librosa
import numpy as np
from PIL import Image
from pathlib import Path
from .unified_dataset import DataProcessingOperator


class LoadRenderMe360GridVideo(DataProcessingOperator):
    """
    Load 4-camera RenderMe360 frames and create 2×2 grid video.

    Input: Dictionary with keys:
        - subject: "0026"
        - performance: "s1_all"
        - start_frame_30fps: 0
        - num_frames: 81

    Output: List of 81 PIL Images (448×832 2×2 grid)

    Grid layout:
    ┌─────────┬─────────┐
    │ cam_28  │ cam_37  │  (224×416 each)
    ├─────────┼─────────┤
    │ cam_49  │ cam_54  │
    └─────────┴─────────┘
    Total: 448×832
    """

    def __init__(self, base_path, cameras=None):
        """
        Args:
            base_path: Path to /ssd4/zhuoyuan/renderme360_4cam/
            cameras: List of 4 camera names (default: ["cam_28", "cam_37", "cam_49", "cam_54"])
        """
        self.base = Path(base_path)
        self.cameras = cameras or ["cam_28", "cam_37", "cam_49", "cam_54"]

    def _get_frame_indices_nearest(self, start_30fps, num_frames_16fps):
        """
        Calculate 30fps frame indices for 16fps output using nearest-frame rounding.

        For each 16fps frame k (k=0..80):
        - Time: t_k = k / 16.0 seconds
        - Nearest 30fps frame: round(t_k * 30) = (k*30 + 8)//16
        """
        indices = []
        for k in range(num_frames_16fps):
            idx_30fps = start_30fps + ((k * 30 + 8) // 16)
            indices.append(idx_30fps)
        return indices

    def _resize_to_target(self, img, target_h, target_w):
        """Resize image to target size maintaining aspect ratio with center crop."""
        w, h = img.size
        scale = max(target_w / w, target_h / h)

        # Resize
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))

        return img

    def _create_grid_2x2(self, top_left, top_right, bottom_left, bottom_right):
        """Create 2×2 grid from 4 images (each 224×416)."""
        # Create empty canvas 448×832
        grid = Image.new("RGB", (832, 448))

        # Paste images
        grid.paste(top_left, (0, 0))           # top-left
        grid.paste(top_right, (416, 0))        # top-right
        grid.paste(bottom_left, (0, 224))      # bottom-left
        grid.paste(bottom_right, (416, 224))   # bottom-right

        return grid

    def __call__(self, data: dict):
        """
        Load video frames as 2×2 grid.

        Args:
            data: Dict with keys: subject, performance, start_frame_30fps, num_frames

        Returns:
            List of PIL Images (448×832 grid)
        """
        subject = str(data.get('subject', '')).zfill(4)
        performance = str(data.get('performance', ''))
        start_frame_30fps = int(data.get('start_frame_30fps', 0))
        num_frames = int(data.get('num_frames', 81))

        # Calculate 30fps frame indices
        frame_indices_30fps = self._get_frame_indices_nearest(start_frame_30fps, num_frames)

        frames = []
        for frame_idx in frame_indices_30fps:
            # Load 4 camera images
            cam_images = {}
            for cam in self.cameras:
                img_path = self.base / subject / performance / "images" / cam / f"frame_{frame_idx:06d}.jpg"

                if not img_path.exists():
                    raise FileNotFoundError(
                        f"Missing frame: {img_path}\n"
                        f"Subject: {subject}, Performance: {performance}, Frame: {frame_idx}, Camera: {cam}"
                    )

                img = Image.open(img_path).convert("RGB")
                img = self._resize_to_target(img, target_h=224, target_w=416)
                cam_images[cam] = img

            # Create 2×2 grid
            grid = self._create_grid_2x2(
                cam_images["cam_28"],   # top-left
                cam_images["cam_37"],   # top-right
                cam_images["cam_49"],   # bottom-left
                cam_images["cam_54"],   # bottom-right
            )
            frames.append(grid)

        return frames


class LoadRenderMe360InputImage(DataProcessingOperator):
    """
    Load single input image (cam_28 at start frame).

    Input: Dictionary with keys:
        - subject: "0026"
        - performance: "s1_all"
        - start_frame_30fps: 0

    Output: PIL Image (224×416)
    """

    def __init__(self, base_path, input_camera="cam_28"):
        """
        Args:
            base_path: Path to /ssd4/zhuoyuan/renderme360_4cam/
            input_camera: Camera to use for input (default: "cam_28")
        """
        self.base = Path(base_path)
        self.input_camera = input_camera

    def _resize_to_target(self, img, target_h, target_w):
        """Resize image to target size maintaining aspect ratio with center crop."""
        w, h = img.size
        scale = max(target_w / w, target_h / h)

        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))

        return img

    def __call__(self, data: dict):
        """
        Load input image.

        Args:
            data: Dict with keys: subject, performance, start_frame_30fps

        Returns:
            PIL Image (224×416)
        """
        subject = str(data.get('subject', '')).zfill(4)
        performance = str(data.get('performance', ''))
        start_frame_30fps = int(data.get('start_frame_30fps', 0))

        img_path = self.base / subject / performance / "images" / self.input_camera / f"frame_{start_frame_30fps:06d}.jpg"

        if not img_path.exists():
            raise FileNotFoundError(
                f"Missing input image: {img_path}\n"
                f"Subject: {subject}, Performance: {performance}, Frame: {start_frame_30fps}"
            )

        img = Image.open(img_path).convert("RGB")
        img = self._resize_to_target(img, target_h=224, target_w=416)

        return img


class LoadRenderMe360Audio(DataProcessingOperator):
    """
    Load audio segment from RenderMe360.

    Input: Dictionary with keys:
        - subject: "0026"
        - performance: "s1_all"
        - start_frame_30fps: 0
        - num_frames: 81

    Output: numpy array float32 [80000] (5.0s at 16kHz)
    """

    def __init__(self, base_path, sr=16000):
        """
        Args:
            base_path: Path to /ssd4/zhuoyuan/renderme360_4cam/
            sr: Sample rate (default: 16000)
        """
        self.base = Path(base_path)
        self.sr = sr

    def __call__(self, data: dict):
        """
        Load audio segment.

        Args:
            data: Dict with keys: subject, performance, start_frame_30fps, num_frames

        Returns:
            numpy array float32
        """
        subject = str(data.get('subject', '')).zfill(4)
        performance = str(data.get('performance', ''))
        start_frame_30fps = int(data.get('start_frame_30fps', 0))
        num_frames = int(data.get('num_frames', 81))

        audio_path = self.base / subject / performance / "audio" / "audio.mp3"

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        # Load full audio
        audio_full, _ = librosa.load(str(audio_path), sr=self.sr)

        # Calculate time range
        # Frame mapping: 16fps video → 30fps source
        # Duration: num_frames / 16 seconds
        start_time = start_frame_30fps / 30.0
        duration = num_frames / 16.0

        # Extract segment
        start_sample = int(start_time * self.sr)
        num_samples = int(duration * self.sr)

        audio_segment = audio_full[start_sample:start_sample + num_samples]

        # Pad if too short (edge case at end of audio)
        if len(audio_segment) < num_samples:
            audio_segment = np.pad(audio_segment, (0, num_samples - len(audio_segment)), mode='constant')

        return audio_segment


class LoadRenderMe360Prompt(DataProcessingOperator):
    """
    Load prompt from metadata or use default.

    Input: Dictionary with key 'prompt' (optional)
    Output: str
    """

    def __init__(self, default_prompt="a person speaking"):
        self.default_prompt = default_prompt

    def __call__(self, data: dict):
        return str(data.get('prompt', self.default_prompt))
