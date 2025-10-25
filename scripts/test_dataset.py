"""
Test script for RenderMe360S2VDataset.
Validates data loading, shapes, and types before training.

Usage:
    python scripts/test_dataset.py
"""

import sys
sys.path.insert(0, '/home/zhuoyuan/projects/DiffSynth-Studio')

from diffsynth.trainers.renderme360_dataset import RenderMe360S2VDataset
from pathlib import Path


def test_dataset():
    """Test dataset loading with 1 sample."""

    print("="*80)
    print("Testing RenderMe360S2VDataset")
    print("="*80)

    # Check if metadata exists
    metadata_path = "/ssd2/zhuoyuan/renderme360_4cam/metadata_single2multi.csv"
    if not Path(metadata_path).exists():
        print(f"\n❌ ERROR: Metadata file not found: {metadata_path}")
        print("Please run: python scripts/generate_metadata_single2multi.py")
        return False

    # Create dataset
    print("\n[1/5] Creating dataset...")
    try:
        dataset = RenderMe360S2VDataset(
            base_path="/ssd2/zhuoyuan/renderme360_4cam/",
            metadata_csv=metadata_path,
            cameras=["cam_28", "cam_37", "cam_49", "cam_54"],
            repeat=1
        )
        print(f"✅ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ ERROR creating dataset: {e}")
        return False

    # Load one sample
    print("\n[2/5] Loading sample 0...")
    try:
        sample = dataset[0]
        print("✅ Sample loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify keys
    print("\n[3/5] Checking sample keys...")
    expected_keys = {"video", "input_image", "input_audio", "audio_sample_rate", "prompt"}
    actual_keys = set(sample.keys())

    if expected_keys == actual_keys:
        print(f"✅ All expected keys present: {expected_keys}")
    else:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            print(f"❌ Missing keys: {missing}")
        if extra:
            print(f"⚠️  Extra keys: {extra}")
        if missing:
            return False

    # Verify shapes and types
    print("\n[4/5] Checking data types and shapes...")
    checks = []

    # Video: List of 81 PIL Images (448×832)
    video = sample["video"]
    if isinstance(video, list) and len(video) == 81:
        first_frame = video[0]
        if hasattr(first_frame, 'size') and first_frame.size == (832, 448):
            print(f"✅ video: List[PIL.Image] with {len(video)} frames, each {first_frame.size[0]}×{first_frame.size[1]}")
            checks.append(True)
        else:
            print(f"❌ video: Frame size is {first_frame.size}, expected (832, 448)")
            checks.append(False)
    else:
        print(f"❌ video: Expected list of 81 frames, got {type(video)} with {len(video) if isinstance(video, list) else '?'} items")
        checks.append(False)

    # Input image: PIL Image (224×416)
    input_image = sample["input_image"]
    if hasattr(input_image, 'size') and input_image.size == (416, 224):
        print(f"✅ input_image: PIL.Image {input_image.size[0]}×{input_image.size[1]}")
        checks.append(True)
    else:
        print(f"❌ input_image: Expected size (416, 224), got {input_image.size if hasattr(input_image, 'size') else type(input_image)}")
        checks.append(False)

    # Audio: numpy array float32 [80000]
    input_audio = sample["input_audio"]
    import numpy as np
    if isinstance(input_audio, np.ndarray) and input_audio.dtype == np.float32 and input_audio.shape == (80000,):
        print(f"✅ input_audio: numpy.float32 array with shape {input_audio.shape}")
        audio_duration = 80000 / 16000
        print(f"   Audio duration: {audio_duration:.3f}s (80000 samples @ 16kHz)")
        checks.append(True)
    else:
        print(f"❌ input_audio: Expected np.float32[80000], got {type(input_audio)} {input_audio.dtype if isinstance(input_audio, np.ndarray) else ''} {input_audio.shape if isinstance(input_audio, np.ndarray) else ''}")
        checks.append(False)

    # Audio sample rate: int 16000
    audio_sr = sample["audio_sample_rate"]
    if audio_sr == 16000:
        print(f"✅ audio_sample_rate: {audio_sr}")
        checks.append(True)
    else:
        print(f"❌ audio_sample_rate: Expected 16000, got {audio_sr}")
        checks.append(False)

    # Prompt: string
    prompt = sample["prompt"]
    if isinstance(prompt, str):
        print(f"✅ prompt: '{prompt}'")
        checks.append(True)
    else:
        print(f"❌ prompt: Expected string, got {type(prompt)}")
        checks.append(False)

    # Summary
    print("\n[5/5] Summary:")
    if all(checks):
        print("✅ All checks passed!")
        print("\n" + "="*80)
        print("Dataset is ready for training!")
        print("="*80)
        print("\nNext steps:")
        print("1. Run small-scale validation (1 subject):")
        print("   Modify the training script to use a subset for testing")
        print("2. Launch full training:")
        print("   bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360.sh")
        return True
    else:
        print(f"❌ {sum(checks)}/{len(checks)} checks passed")
        print("\nPlease fix the issues above before training.")
        return False


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
