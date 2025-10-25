"""
RenderMe360 Dataset for Wan2.2-S2V Training
Single-view input (cam_54) → Multi-view output (2×2 grid) with audio
"""

import torch
import pandas as pd
import numpy as np
import torchaudio
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF


class RenderMe360S2VDataset(torch.utils.data.Dataset):
    """
    Custom dataset for RenderMe360 single→multi S2V training.

    Loads:
    - Single cam_54 input image (224×416)
    - 2×2 grid video with 4 cameras (448×832)
    - Audio segment (5.0s at 16kHz = 80,000 samples)

    Args:
        base_path: Path to /ssd2/zhuoyuan/renderme360_4cam/
        metadata_csv: Path to metadata CSV
        cameras: List of 4 camera names (default: ["cam_28", "cam_37", "cam_49", "cam_54"])
        repeat: Dataset repetitions per epoch
    """

    def __init__(self, base_path, metadata_csv, cameras=None, repeat=1):
        self.base = Path(base_path)
        self.df = pd.read_csv(metadata_csv)
        self.cameras = cameras or ["cam_28", "cam_37", "cam_49", "cam_54"]
        self.repeat = repeat

        # Set torchaudio backend for MP3 support
        try:
            torchaudio.set_audio_backend("sox_io")
        except Exception:
            pass  # Fall back to default backend

        print(f"[RenderMe360Dataset] Loaded {len(self.df)} samples × {repeat} repeats = {len(self)} total")

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        row = self.df.iloc[idx % len(self.df)]

        subject = str(row['subject']).zfill(4)  # Convert to string with leading zeros (e.g., "0026")
        performance = str(row['performance'])
        start_frame_30fps = int(row['start_frame_30fps'])
        num_frames = int(row['num_frames'])  # 81
        prompt = row.get('prompt', 'a person speaking')

        # 1. Calculate 30fps frame indices using NEAREST-FRAME rounding
        indices_30fps = self._get_frame_indices_nearest(start_frame_30fps, num_frames)

        # 2. Load 2×2 grid video frames
        video_frames = self._load_grid_video(subject, performance, indices_30fps)

        # 3. Load single input image (cam_54 at start frame)
        input_image = self._load_input_image(subject, performance, start_frame_30fps)

        # 4. Load audio segment
        audio_path = self.base / subject / performance / "audio" / "audio.mp3"
        input_audio, audio_sr = self._load_audio_segment(
            audio_path,
            start_frame_30fps,
            num_frames
        )

        return {
            "video": video_frames,           # List of 81 PIL Images (448×832 each)
            "input_image": input_image,      # PIL Image (224×416)
            "input_audio": input_audio,      # numpy array float32 [80000]
            "audio_sample_rate": audio_sr,   # 16000
            "prompt": prompt,
        }

    def _get_frame_indices_nearest(self, start_30fps, num_frames_16fps):
        """
        Calculate 30fps frame indices for 16fps output using nearest-frame rounding.
        Uses integer arithmetic to avoid platform-dependent floating-point rounding.

        For each 16fps frame k (k=0..80):
        - Time: t_k = k / 16.0 seconds
        - Nearest 30fps frame: round(t_k * 30) = (k*30 + 8)//16

        Args:
            start_30fps: Starting 30fps frame index
            num_frames_16fps: Number of 16fps frames (81)

        Returns:
            List of 30fps frame indices
        """
        indices = []
        for k in range(num_frames_16fps):
            # Integer-only version: (k*30 + 8)//16
            idx_30fps = start_30fps + ((k * 30 + 8) // 16)
            indices.append(idx_30fps)
        return indices

    def _load_grid_video(self, subject, performance, frame_indices_30fps):
        """
        Load 4 camera views and create 2×2 grid for each frame.

        Grid layout:
        ┌─────────┬─────────┐
        │ cam_28  │ cam_37  │  (224×416 each)
        ├─────────┼─────────┤
        │ cam_49  │ cam_54  │
        └─────────┴─────────┘
        Total: 448×832
        """
        frames = []

        for frame_idx in frame_indices_30fps:
            # Load 4 camera images
            cam_images = {}
            for cam in self.cameras:
                img_path = self.base / subject / performance / "images" / cam / f"frame_{frame_idx:06d}.jpg"

                # File exists guard (fail fast with clear error)
                if not img_path.exists():
                    raise FileNotFoundError(
                        f"Missing frame: {img_path}\n"
                        f"Subject: {subject}, Performance: {performance}, Frame: {frame_idx}, Camera: {cam}"
                    )

                img = Image.open(img_path).convert("RGB")
                # Resize to 224×416 (each camera view in grid)
                img = self._resize_to_target(img, target_h=224, target_w=416)
                cam_images[cam] = img

            # Create 2×2 grid (448×832)
            grid = self._create_grid_2x2(
                cam_images["cam_28"],  # top-left
                cam_images["cam_37"],  # top-right
                cam_images["cam_49"],  # bottom-left
                cam_images["cam_54"],  # bottom-right
            )
            frames.append(grid)

        return frames

    def _load_input_image(self, subject, performance, start_frame_30fps):
        """Load single cam_54 image at start frame (input to model)."""
        img_path = self.base / subject / performance / "images" / "cam_54" / f"frame_{start_frame_30fps:06d}.jpg"

        if not img_path.exists():
            raise FileNotFoundError(f"Missing input image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        # Resize to 224×416 (single camera size)
        img = self._resize_to_target(img, target_h=224, target_w=416)
        return img

    def _load_audio_segment(self, audio_path, start_frame_30fps, num_frames_16fps, target_sr=16000):
        """
        Load audio segment using torchaudio with frame-accurate offset.

        Args:
            audio_path: Path to audio.mp3
            start_frame_30fps: Starting 30fps frame index
            num_frames_16fps: Number of 16fps frames (81)
            target_sr: Target sample rate (16000)

        Returns:
            Tuple of (waveform numpy array float32, sample_rate)
        """
        # Calculate time range
        start_time_s = start_frame_30fps / 30.0
        duration_s = (num_frames_16fps - 1) / 16.0  # 80/16 = 5.0 seconds

        # Get source sample rate
        info = torchaudio.info(str(audio_path))
        src_sr = info.sample_rate

        # Calculate exact sample window in source rate
        start_sample = int(round(start_time_s * src_sr))
        num_samples = int(round(duration_s * src_sr))

        # Load segment (frame-accurate with torchaudio)
        waveform, sr = torchaudio.load(
            str(audio_path),
            frame_offset=start_sample,
            num_frames=num_samples
        )

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Enforce EXACTLY 80,000 samples (crop or pad)
        target_samples = 80_000
        waveform = waveform.squeeze(0).numpy().astype('float32')

        if waveform.shape[0] < target_samples:
            # Pad with zeros
            waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
        else:
            # Crop to exact length
            waveform = waveform[:target_samples]

        return waveform, target_sr

    def _resize_to_target(self, image, target_h, target_w):
        """
        Resize image to exact target size.
        Uses center crop approach to maintain aspect ratio without distortion.

        Args:
            image: PIL Image
            target_h: Target height
            target_w: Target width

        Returns:
            PIL Image resized to (target_w, target_h)
        """
        w, h = image.size

        # Calculate scale to cover target (larger of the two ratios)
        scale = max(target_w / w, target_h / h)

        # Resize to cover
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image = TF.resize(image, (new_h, new_w))

        # Center crop to exact target
        image = TF.center_crop(image, (target_h, target_w))

        return image

    def _create_grid_2x2(self, top_left, top_right, bottom_left, bottom_right):
        """
        Create 2×2 grid from 4 images (each 224×416) → 448×832.

        Args:
            top_left, top_right, bottom_left, bottom_right: PIL Images (224×416 each)

        Returns:
            PIL Image (448×832)
        """
        grid = Image.new("RGB", (832, 448))  # width, height
        grid.paste(top_left, (0, 0))          # cam_28
        grid.paste(top_right, (416, 0))       # cam_37
        grid.paste(bottom_left, (0, 224))     # cam_49
        grid.paste(bottom_right, (416, 224))  # cam_54
        return grid


def passthrough_collate(batch):
    """
    Collate function for batch_size=1.
    Dataset returns PIL Images and numpy arrays which default collate can't handle.

    Args:
        batch: List of samples (length 1 for batch_size=1)

    Returns:
        Single sample dict
    """
    assert len(batch) == 1, f"This pipeline requires batch_size=1, got {len(batch)}"
    return batch[0]
