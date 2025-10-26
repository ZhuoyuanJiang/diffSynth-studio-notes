"""
Test S2V Inference with Local Models
Uses models from /ssd1/zhuoyuan/diffsynth_models/ to avoid re-downloading
"""
import torch
from PIL import Image
import librosa
from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

# Base path for models on vllab14
LOCAL_MODEL_PATH = "/ssd1/zhuoyuan/diffsynth_models/models"

print("=" * 80)
print("Testing Wan2.2-S2V-14B Inference with Local Models")
print("=" * 80)
print(f"Local model path: {LOCAL_MODEL_PATH}")
print()

# Initialize pipeline with local model paths (same structure as official script)
# Using skip_download=True to use existing models without re-downloading
print("Loading models...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",  # Use cuda:0 instead of cuda for VRAM management compatibility
    model_configs=[
        # DiT model (30GB, 4 files - glob pattern matches all)
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
    # Audio processor config (directory with tokenizer files)
    audio_processor_config=ModelConfig(
        model_id="Wan-AI/Wan2.2-S2V-14B",
        origin_file_pattern="wav2vec2-large-xlsr-53-english/",
        local_model_path=LOCAL_MODEL_PATH,
        skip_download=True
    ),
)
print("✓ Models loaded successfully!")
print()

# Enable VRAM management to avoid OOM on single GPU
# This enables layer-wise CPU offloading - essential for running on <80GB GPUs
print("Enabling VRAM management (layer-wise offloading)...")
pipe.enable_vram_management()  # Uses defaults: auto-detect VRAM, 0.5GB buffer
print("✓ VRAM management enabled!")
print()

# Download example data
print("Downloading example data...")
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_video_dataset",
    local_dir="./data/example_video_dataset",
    allow_file_pattern=f"wans2v/*"
)
print("✓ Example data ready!")
print()

# Configuration
num_frames = 81  # 4n+1
height = 448
width = 832

prompt = "a person is singing"
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Load input image
input_image = Image.open("data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))

# Load audio (s2v audio input, recommend 16kHz sampling rate)
audio_path = 'data/example_video_dataset/wans2v/sing.MP3'
input_audio, sample_rate = librosa.load(audio_path, sr=16000)

print("=" * 80)
print("Running Speech-to-Video Inference")
print("=" * 80)
print(f"Prompt: {prompt}")
print(f"Num frames: {num_frames}")
print(f"Resolution: {height}x{width}")
print(f"Inference steps: 40")
print()

# Speech-to-video
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    num_inference_steps=40,
)

output_path = "test_video_with_audio.mp4"
save_video_with_audio(video[1:], output_path, audio_path, fps=16, quality=5)
print(f"✓ Video saved to: {output_path}")
print()

print("=" * 80)
print("Running Speech-to-Video with Pose Inference")
print("=" * 80)

# s2v will use the first (num_frames) frames as reference
# height and width must be the same as input_image
# fps should be 16, the same as output video fps
pose_video_path = 'data/example_video_dataset/wans2v/pose.mp4'
pose_video = VideoData(pose_video_path, height=height, width=width)

# Speech-to-video with pose
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    s2v_pose_video=pose_video,
    num_inference_steps=40,
)

output_path_pose = "test_video_pose_with_audio.mp4"
save_video_with_audio(video[1:], output_path_pose, audio_path, fps=16, quality=5)
print(f"✓ Video with pose saved to: {output_path_pose}")
print()

print("=" * 80)
print("✓ ALL TESTS PASSED! Inference pipeline working correctly!")
print("=" * 80)
