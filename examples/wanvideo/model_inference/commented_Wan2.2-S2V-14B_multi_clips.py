# =============================================================================
# Wan2.2-S2V-14B Multi-Clips Inference Script (COMMENTED VERSION)
# =============================================================================
# Purpose: Generate speaking videos from audio + reference image
# Key Mechanism: Audio → wav2vec2 embeddings → DiT → Video frames
# =============================================================================

import torch
from PIL import Image
import librosa  # For loading audio files
from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from modelscope import dataset_snapshot_download


# =============================================================================
# MAIN FUNCTION: speech_to_video()
# =============================================================================
def speech_to_video(
    # ========== REQUIRED INPUTS ==========
    prompt,              # Text description of the video (e.g., "a person is singing")
    input_image,         # Reference image (PIL Image) - the face/person to animate
    audio_path,          # Path to audio file (.mp3, .wav, etc.)

    # ========== OPTIONAL INPUTS ==========
    negative_prompt="",  # What to avoid in generation (e.g., "blurry, low quality")
    num_clip=None,       # Limit generation to first N clips (None = generate all)
    audio_sample_rate=16000,  # ⭐ IMPORTANT: Audio must be 16kHz for S2V!
    pose_video_path=None,     # Optional: Pose guidance video (can be None)
    infer_frames=80,     # Number of frames per clip (must be 4n, so num_frames=81 satisfies %4==1)
    height=448,          # Video height
    width=832,           # Video width
    num_inference_steps=40,   # Denoising steps (more = better quality, slower)
    fps=16,              # ⭐ IMPORTANT: S2V works best at 16 fps (fixed)
    motion_frames=73,    # ⭐ HYPERPARAMETER: Number of frames used for motion context
    save_path=None,      # Where to save the output video
):
    # =========================================================================
    # STEP 1: Load and resample audio to 16kHz
    # =========================================================================
    # librosa.load() reads audio and resamples to target sample rate
    input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
    # input_audio: numpy array of audio waveform, shape (num_samples,)
    # sample_rate: confirmed sample rate (should be 16000)

    # =========================================================================
    # STEP 2: Load optional pose video
    # =========================================================================
    # Pose video provides motion guidance (facial expressions, head movement)
    # If not provided, S2V will generate motion from audio alone
    pose_video = VideoData(pose_video_path, height=height, width=width) if pose_video_path is not None else None

    # =========================================================================
    # STEP 3: ⭐⭐⭐ PRE-CALCULATE AUDIO AND POSE EMBEDDINGS ⭐⭐⭐
    # =========================================================================
    # This is a KEY step! Audio is processed BEFORE the main generation loop
    # Why? Audio is split into chunks (one per clip) for long videos
    audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
        pipe=pipe,              # The loaded pipeline (contains wav2vec2 encoder)
        input_audio=input_audio,  # Raw audio waveform
        audio_sample_rate=sample_rate,  # Confirmed 16kHz
        s2v_pose_video=pose_video,  # Optional pose video
        num_frames=infer_frames + 1,  # 81 frames (satisfies %4==1 constraint)
        height=height,
        width=width,
        fps=fps,
    )
    # Returns:
    # - audio_embeds: List of audio embeddings, one per clip [clip1_embeds, clip2_embeds, ...]
    # - pose_latents: List of pose latents (if pose_video provided), else None
    # - num_repeat: Number of clips needed to cover the full audio duration

    # =========================================================================
    # STEP 4: Limit number of clips if user specified num_clip
    # =========================================================================
    num_repeat = min(num_repeat, num_clip) if num_clip is not None else num_repeat
    print(f"Generating {num_repeat} video clips...")

    # =========================================================================
    # STEP 5: Initialize motion context and output video list
    # =========================================================================
    motion_videos = []  # Stores recent frames for temporal consistency
    video = []          # Accumulates all generated frames

    # =========================================================================
    # STEP 6: ⭐⭐⭐ MAIN GENERATION LOOP ⭐⭐⭐
    # =========================================================================
    # Generate video clip by clip (each clip is ~80 frames)
    for r in range(num_repeat):
        # ---------------------------------------------------------------------
        # Extract pose latents for this clip (if available)
        # ---------------------------------------------------------------------
        s2v_pose_latents = pose_latents[r] if pose_latents is not None else None

        # ---------------------------------------------------------------------
        # ⭐⭐⭐ CALL THE PIPELINE TO GENERATE ONE CLIP ⭐⭐⭐
        # ---------------------------------------------------------------------
        current_clip = pipe(
            prompt=prompt,                   # Text prompt
            input_image=input_image,         # Reference image (first frame)
            negative_prompt=negative_prompt,
            seed=0,                          # Random seed for reproducibility
            num_frames=infer_frames + 1,     # 81 frames
            height=height,
            width=width,
            audio_embeds=audio_embeds[r],    # ⭐ Audio embeddings for THIS clip
            s2v_pose_latents=s2v_pose_latents,  # Pose guidance (optional)
            motion_video=motion_videos,      # ⭐ Motion context from previous frames
            num_inference_steps=num_inference_steps,
        )
        # current_clip: List of PIL Images, length = 81 frames

        # ---------------------------------------------------------------------
        # Remove the first frame (overlap with previous clip's last frame)
        # ---------------------------------------------------------------------
        current_clip = current_clip[-infer_frames:]  # Keep last 80 frames (drop first 1)

        # ---------------------------------------------------------------------
        # Special handling for first clip: remove first 3 frames
        # ---------------------------------------------------------------------
        if r == 0:
            current_clip = current_clip[3:]  # Drop first 3 frames (transition smoothing)

        # ---------------------------------------------------------------------
        # ⭐ UPDATE MOTION CONTEXT for next clip
        # ---------------------------------------------------------------------
        # Keep last 73 frames as motion reference for temporal consistency
        overlap_frames_num = min(motion_frames, len(current_clip))  # Usually 73
        motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]
        # This sliding window ensures smooth transitions between clips

        # ---------------------------------------------------------------------
        # Add current clip to output video
        # ---------------------------------------------------------------------
        video.extend(current_clip)  # Append frames to full video

        # ---------------------------------------------------------------------
        # Save video incrementally (so you can see progress)
        # ---------------------------------------------------------------------
        save_video_with_audio(video, save_path, audio_path, fps=16, quality=5)
        print(f"processed the {r+1}th clip of total {num_repeat} clips.")

    return video  # Return list of all frames


# =============================================================================
# PIPELINE INITIALIZATION
# =============================================================================
# ⭐⭐⭐ CRITICAL: Load all 4 model components ⭐⭐⭐
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device="cuda",               # Run on GPU
    model_configs=[
        # Component 1: DiT (Diffusion Transformer) - 14B parameters
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),

        # Component 2: T5 Text Encoder - Encodes prompts into embeddings
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),

        # Component 3: ⭐ wav2vec2 Audio Encoder - Converts audio to embeddings
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),

        # Component 4: VAE (Video Autoencoder) - Encodes/decodes video latents
        ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    # Audio processor config (wav2vec2 tokenizer/config files)
    audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
)

# =============================================================================
# DOWNLOAD EXAMPLE DATA
# =============================================================================
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_video_dataset",
    local_dir="./data/example_video_dataset",
    allow_file_pattern=f"wans2v/*",  # Download only S2V examples
)

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
infer_frames = 80  # 4n, so num_frames = 81 (satisfies %4==1)
height = 448       # Video resolution
width = 832

# Text prompt for video generation
prompt = "a person is singing"

# Negative prompt (Chinese text describing bad quality attributes to avoid)
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Load and resize reference image
input_image = Image.open("data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))

# =============================================================================
# EXAMPLE 1: Generate full video with audio and pose guidance
# =============================================================================
video_with_audio = speech_to_video(
    prompt=prompt,
    input_image=input_image,
    audio_path='data/example_video_dataset/wans2v/sing.MP3',  # Audio file
    negative_prompt=negative_prompt,
    pose_video_path='data/example_video_dataset/wans2v/pose.mp4',  # Optional pose guidance
    save_path="video_with_audio_full.mp4",  # Output path
    infer_frames=infer_frames,
    height=height,
    width=width,
)

# =============================================================================
# EXAMPLE 2: Generate only first 2 clips (2 * 80 = 160 frames)
# =============================================================================
# num_clip means generating only the first n clips with n * infer_frames frames.
video_with_audio_pose = speech_to_video(
    prompt=prompt,
    input_image=input_image,
    audio_path='data/example_video_dataset/wans2v/sing.MP3',
    negative_prompt=negative_prompt,
    pose_video_path='data/example_video_dataset/wans2v/pose.mp4',
    save_path="video_with_audio_pose_clip_2.mp4",
    num_clip=2  # ⭐ Only generate first 2 clips (faster for testing)
)


# =============================================================================
# KEY TAKEAWAYS FOR TRAINING
# =============================================================================
# 1. Required inputs: audio (16kHz) + reference image + prompt
# 2. Audio is preprocessed into embeddings BEFORE generation loop
# 3. Frame constraint: num_frames % 4 == 1 (e.g., 81, 85, 89)
# 4. 4 model components: DiT, T5, wav2vec2, VAE
# 5. Pose video is OPTIONAL (can be None)
# 6. Motion context (73 frames) ensures temporal consistency
#
# For training, you need:
# - CSV with columns: video, audio, prompt, input_image, [s2v_pose_video]
# - Audio files resampled to 16kHz
# - Video clips with num_frames % 4 == 1
# - Training script will use WanVideoUnit_S2V to process audio automatically
# =============================================================================
