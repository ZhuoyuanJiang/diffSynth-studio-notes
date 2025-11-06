"""
Test Wan2.2-S2V Inference with RenderMe360 Dataset

This script:
1. Loads 1 sample from RenderMe360UnifiedDataset
2. Extracts input_image (cam_28) and audio (5-second segment)
3. Runs Wan2.2-S2V inference
4. Saves generated video and ground truth for comparison
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# DiffSynth imports
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.renderme360_unified_dataset import RenderMe360UnifiedDataset
from diffsynth import save_video_with_audio
import tempfile


def main():
    print("=" * 80)
    print("Testing Wan2.2-S2V Inference with RenderMe360 Dataset")
    print("=" * 80)

    # Configuration
    LOCAL_MODEL_PATH = "/ssd2/zhuoyuan/diffsynth_models/models"
    BASE_PATH = "/ssd2/zhuoyuan/renderme360_4cam"
    METADATA_PATH = f"{BASE_PATH}/metadata_single2multi.csv"
    DEVICE = "cuda:0"

    print(f"\nDataset: {BASE_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Models: {LOCAL_MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # ========================================================================
    # Step 1: Load RenderMe360 Dataset
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Loading RenderMe360 Dataset")
    print("=" * 80)

    dataset = RenderMe360UnifiedDataset(
        base_path=BASE_PATH,
        metadata_path=METADATA_PATH,
        repeat=1,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Load first sample
    sample_idx = 0
    sample = dataset[sample_idx]

    print(f"\n✓ Loaded sample {sample_idx}:")
    print(f"  - Input image: {sample['input_image'].size}")
    print(f"  - Audio: {sample['input_audio'].shape}")
    print(f"  - Video frames: {len(sample['video'])}")
    print(f"  - Prompt: {sample['prompt']}")

    # Extract data
    input_image = sample['input_image']  # PIL Image (416×224)
    audio = sample['input_audio']  # numpy array [81000]
    ground_truth_frames = sample['video']  # List of 81 PIL Images (832×448)
    prompt = sample['prompt']

    # Save ground truth grid (first frame) for comparison
    ground_truth_frames[0].save("test_renderme_ground_truth_frame0.png")
    print(f"\n✓ Saved ground truth frame 0: test_renderme_ground_truth_frame0.png")

    # Save input image
    input_image.save("test_renderme_input_image.png")
    print(f"✓ Saved input image: test_renderme_input_image.png")

    # ========================================================================
    # Step 2: Load Wan2.2-S2V Pipeline
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Loading Wan2.2-S2V Pipeline")
    print("=" * 80)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=DEVICE,
        model_configs=[
            # DiT model (30GB, 4 files)
            ModelConfig(
                model_id="Wan-AI/Wan2.2-S2V-14B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                local_model_path=LOCAL_MODEL_PATH,
                skip_download=True
            ),
            # Wav2vec2 audio encoder (1.2GB)
            ModelConfig(
                model_id="Wan-AI/Wan2.2-S2V-14B",
                origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors",
                local_model_path=LOCAL_MODEL_PATH,
                skip_download=True
            ),
            # T5 text encoder (11GB)
            ModelConfig(
                model_id="Wan-AI/Wan2.2-S2V-14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                local_model_path=LOCAL_MODEL_PATH,
                skip_download=True
            ),
            # VAE (485MB)
            ModelConfig(
                model_id="Wan-AI/Wan2.2-S2V-14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                local_model_path=LOCAL_MODEL_PATH,
                skip_download=True
            ),
        ],
        audio_processor_config=ModelConfig(
            model_id="Wan-AI/Wan2.2-S2V-14B",
            origin_file_pattern="wav2vec2-large-xlsr-53-english/",
            local_model_path=LOCAL_MODEL_PATH,
            skip_download=True
        ),
    )
    print("✓ Pipeline loaded!")

    # Enable VRAM management for single GPU
    print("\nEnabling VRAM management...")
    pipe.enable_vram_management()
    print("✓ VRAM management enabled!")

    # ========================================================================
    # Step 3: Run Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Running Wan2.2-S2V Inference")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Input image: {input_image.size}")
    print(f"Audio: {audio.shape[0]} samples @ 16kHz")
    print(f"Target: 81 frames @ 448×832")
    print(f"Inference steps: 40")
    print()

    generated_video = pipe(
        prompt=prompt,
        input_image=input_image,
        input_audio=audio,
        negative_prompt="",
        seed=0,
        num_frames=81,
        height=448,
        width=832,
        num_inference_steps=40,
        audio_sample_rate=16000,
    )

    print("\n✓ Inference complete!")

    # ========================================================================
    # Step 4: Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Saving Results")
    print("=" * 80)

    # Save audio to temp file
    import soundfile as sf
    temp_audio_path = "temp_audio.wav"
    sf.write(temp_audio_path, audio, 16000)
    print(f"✓ Saved temp audio: {temp_audio_path}")

    # Save generated video with audio (skip first frame - it's the reference)
    output_video = "test_renderme_generated.mp4"
    save_video_with_audio(generated_video[1:], output_video, temp_audio_path, fps=16, quality=5)
    print(f"✓ Saved generated video: {output_video}")

    # Save ground truth video with audio
    output_gt = "test_renderme_ground_truth.mp4"
    save_video_with_audio(ground_truth_frames, output_gt, temp_audio_path, fps=16, quality=5)
    print(f"✓ Saved ground truth video: {output_gt}")

    # Save first frame of generated video (frame 1, since frame 0 is reference)
    generated_video[1].save("test_renderme_generated_frame0.png")
    print(f"✓ Saved generated frame 0: test_renderme_generated_frame0.png")

    print("\n" + "=" * 80)
    print("✓ TESTING COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. test_renderme_input_image.png - Input image (cam_28)")
    print("  2. test_renderme_ground_truth_frame0.png - Ground truth frame 0")
    print("  3. test_renderme_generated_frame0.png - Generated frame 0")
    print("  4. test_renderme_ground_truth.mp4 - Ground truth 2×2 grid video")
    print("  5. test_renderme_generated.mp4 - Generated video (pretrained model)")
    print("\nNext steps:")
    print("  - Compare generated vs ground truth")
    print("  - Check if model understands the task (baseline)")
    print("  - If working, proceed to training integration")


if __name__ == "__main__":
    main()
