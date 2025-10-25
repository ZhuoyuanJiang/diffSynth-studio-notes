# ============================================================================
# EDUCATIONAL COMMENTS FOR PHASE 2 LEARNING - Training Script
# ============================================================================
"""
📚 COMMENTED VERSION OF examples/wanvideo/model_training/train.py

This is the MAIN TRAINING SCRIPT for Wan Video models (S2V, Animate, etc.).
This script shows HOW the foundation classes (BasePipeline, ModelConfig, PipelineUnit)
are used in practice for training.

📍 WHAT'S IN THIS FILE:
1. WanTrainingModule - Custom training module for Wan Video (lines 26-90)
2. Main training script - Dataset setup + training launch (lines 92-132)

🎯 WHY THIS MATTERS FOR S2V TRAINING:
- This is THE script you run to train S2V models
- Shows how inputs flow through the pipeline during training
- Demonstrates how to configure LoRA training
- Shows how UnifiedDataset loads video+audio data

📖 KEY CONCEPTS TO UNDERSTAND:
1. forward_preprocess(): Prepares inputs (video, audio, prompt) for training
2. Pipeline units: Automatically process inputs (audio encoding, VAE encoding, etc.)
3. training_loss(): Computes diffusion loss for training
4. Three input dicts: inputs_shared, inputs_posi, inputs_nega

📊 THE TRAINING FLOW:
1. UnifiedDataset loads video+audio from disk
2. forward_preprocess() creates three input dicts
3. Pipeline units process inputs (extract features, encode, etc.)
4. training_loss() computes loss using DiT forward pass
5. Optimizer updates weights (only trainable parts, e.g., LoRA)
"""

# ===== IMPORTS =====
import torch, os, json  # Core libraries: PyTorch, file operations, JSON parsing
from diffsynth import load_state_dict  # Utility for loading model checkpoints
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig  # The pipeline we'll train
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser  # Training framework components
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath  # Dataset loading utilities
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid warnings during multiprocessing



# ============================================================================
# CLASS: WanTrainingModule - The Core Training Logic
# ============================================================================
# PURPOSE: Wraps WanVideoPipeline for training
#
# INHERITANCE: DiffusionTrainingModule (provides common training utilities)
#
# KEY METHODS:
# - __init__(): Initialize pipeline, configure LoRA, enable gradient checkpointing
# - forward_preprocess(): Convert dataset batch into pipeline inputs
# - forward(): Compute training loss
#
# HOW IT'S USED:
#   model = WanTrainingModule(model_paths=..., trainable_models="dit", ...)
#   loss = model.forward(data)  # data from UnifiedDataset
#   loss.backward()  # Gradients only flow to trainable parts
# ============================================================================

class WanTrainingModule(DiffusionTrainingModule):
    # ========================================================================
    # METHOD: __init__ - Initialize Training Module
    # ========================================================================
    # PURPOSE: Set up pipeline, configure training mode (full or LoRA)
    #
    # PARAMETERS:
    # - model_paths: LEGACY - comma-separated paths to model files
    # - model_id_with_origin_paths: NEW - format "model_id:file_pattern,..."
    #   Example: "Wan-AI/Wan2.2-S2V-14B:*.safetensors,openai/clip-vit-large-patch14:*"
    # - trainable_models: Which models to train (e.g., "dit" or "dit,vae")
    # - lora_base_model: Which model to add LoRA to (e.g., "dit")
    # - lora_target_modules: Comma-separated module names for LoRA injection
    #   Default: "q,k,v,o,ffn.0,ffn.2" (attention + feedforward layers)
    # - lora_rank: Rank of LoRA matrices (lower = fewer parameters, 32 is typical)
    # - lora_checkpoint: Path to existing LoRA checkpoint to resume from
    # - use_gradient_checkpointing: Enable to reduce memory (recompute activations)
    # - use_gradient_checkpointing_offload: Offload checkpointed activations to CPU
    # - extra_inputs: Comma-separated extra inputs (e.g., "input_audio,input_image")
    # - max_timestep_boundary: Maximum noise level for training (1.0 = full noise)
    # - min_timestep_boundary: Minimum noise level for training (0.0 = no noise)
    # ========================================================================
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,  # Model loading config (NEW style preferred)
        trainable_models=None,  # Which models to train (e.g., "dit")
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,  # LoRA configuration
        use_gradient_checkpointing=True,  # Memory optimization (recommended for 14B models)
        use_gradient_checkpointing_offload=False,  # Extra memory savings (slower)
        extra_inputs=None,  # Additional inputs beyond video+prompt (e.g., audio, images)
        max_timestep_boundary=1.0,  # Training noise range: max (1.0 = fully noised)
        min_timestep_boundary=0.0,  # Training noise range: min (0.0 = clean)
    ):
        super().__init__()  # Initialize parent DiffusionTrainingModule

        # ===== STEP 1: Load Models =====
        # Convert command-line args into list of ModelConfig objects
        # parse_model_configs() parses "model_id:file_pattern,..." format
        # Returns: [ModelConfig(model_id="...", origin_file_pattern="..."), ...]
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)  # Parse model paths/IDs into ModelConfig objects

        # Initialize WanVideoPipeline with the parsed configs
        # - torch_dtype=bfloat16: Use BF16 for training (good balance of precision/memory)
        # - device="cpu": Load models to CPU initially (will move to GPU later)
        # - model_configs: List of ModelConfig objects specifying what to load
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)  # Initialize pipeline and load all models

        # ===== STEP 2: Configure Training Mode =====
        # This method:
        # 1. Freezes all models EXCEPT those in trainable_models
        # 2. If lora_base_model specified, adds LoRA layers to that model
        # 3. Sets up gradient computation for trainable parts only
        #
        # Example: trainable_models="dit", lora_base_model="dit"
        #   → Freezes vae, text_encoder, audio_encoder (no gradients)
        #   → Adds LoRA to dit's attention and FFN layers
        #   → Only LoRA parameters (few MB) get updated during training
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,  # Which models to train (e.g., "dit")
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,  # LoRA configuration
            enable_fp8_training=False,  # FP8 training not supported for Wan Video yet
        )

        # ===== STEP 3: Store Training Configuration =====
        # These configs will be passed to the pipeline during forward pass
        self.use_gradient_checkpointing = use_gradient_checkpointing  # If True, recompute activations during backward pass (saves memory)
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload  # If True, offload checkpointed activations to CPU (saves more memory, slower)
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []  # Parse comma-separated extra inputs (e.g., ["input_audio", "input_image"])
        self.max_timestep_boundary = max_timestep_boundary  # Maximum noise level for training (1.0 = fully noised latent)
        self.min_timestep_boundary = min_timestep_boundary  # Minimum noise level for training (0.0 = clean latent)



    # ========================================================================
    # ⭐ CRITICAL METHOD: forward_preprocess() - Prepare Training Inputs
    # ========================================================================
    # PURPOSE: Convert UnifiedDataset batch into the three input dictionaries
    #          that WanVideoPipeline expects (inputs_shared, inputs_posi, inputs_nega)
    #
    # WHAT IT DOES:
    # 1. Create inputs_posi with positive conditioning (prompt, audio, etc.)
    # 2. Create inputs_nega with negative conditioning (for CFG during inference)
    # 3. Create inputs_shared with CFG-insensitive data (video, dimensions, configs)
    # 4. Run all pipeline units to process inputs (extract features, encode, etc.)
    # 5. Merge all three dicts and return for training_loss()
    #
    # 📥 INPUT: data (dict from UnifiedDataset) - RAW DATA FROM DISK
    #   Example for S2V:
    #   {
    #     "video": [PIL.Image, PIL.Image, ...],  # 81 frames as PIL Images
    #     "audio": "/path/to/dataset/audio/sample001.wav",  # Path to audio file (NOT loaded yet!)
    #     "prompt": "A person speaking with neutral expression"  # Text prompt
    #   }
    #
    # 📤 OUTPUT: dict with PROCESSED inputs ready for DiT - AFTER PIPELINE UNITS
    #   After merging {**inputs_shared, **inputs_posi}:
    #   {
    #     # From inputs_shared (processed by units):
    #     "latents": torch.Tensor,  # [1, 16, 81, H/8, W/8] - VAE-encoded video
    #     "height": 720,
    #     "width": 1280,
    #     "num_frames": 81,
    #     ... (other configs)
    #
    #     # From inputs_posi (processed by units):
    #     "prompt_embeds": torch.Tensor,  # [1, 77, 768] - CLIP text embeddings
    #     "audio_embeds": torch.Tensor,  # [1, T_audio, 1280] - wav2vec2 audio embeddings
    #   }
    #
    # ⚠️ IMPORTANT: We return merged dict because training_loss() needs all inputs together.
    # But internally, pipeline units work with separate dicts (shared/posi/nega) for CFG support.
    #
    # 🔍 KEY INSIGHT:
    # Pipeline units (WanVideoUnit_S2V, WanVideoUnit_PromptEmbedder, etc.)
    # automatically extract and process features during this method.
    # For S2V, WanVideoUnit_S2V extracts audio_embeds from inputs_shared["input_audio"]
    # ========================================================================
    def forward_preprocess(self, data):
        # ===== STEP 1: Prepare Positive Conditioning (CFG-Sensitive) =====
        # These inputs affect the CONTENT of generation (prompt, audio, reference images)
        # During CFG, positive and negative conditions are processed separately
        inputs_posi = {"prompt": data["prompt"]}  # Text prompt describing the desired output (e.g., "A person speaking")
        # Note: For S2V, audio will be added to inputs_posi by WanVideoUnit_S2V.process()
        # The unit extracts it from inputs_shared["input_audio"] and adds it here

        # ===== STEP 2: Prepare Negative Conditioning (CFG-Sensitive) =====
        # Used for classifier-free guidance during inference
        # For training, we typically use cfg_scale=1 (no CFG), so this is empty
        inputs_nega = {}  # Empty for training (no negative prompt)

        # ===== STEP 3: Prepare Shared Inputs (CFG-Insensitive) =====
        # These inputs don't affect conditioning but are needed for processing
        # (video data, dimensions, training configs)
        inputs_shared = {
            # ===== Video Data =====
            "input_video": data["video"],  # List of PIL.Image frames from UnifiedDataset (e.g., 81 frames)
            "height": data["video"][0].size[1],  # Video height in pixels (e.g., 720)
            "width": data["video"][0].size[0],  # Video width in pixels (e.g., 1280)
            "num_frames": len(data["video"]),  # Number of frames (e.g., 81, must satisfy: 81 % 4 == 1)

            # ===== Training Configuration =====
            # These parameters control how the pipeline operates during training
            "cfg_scale": 1,  # Classifier-free guidance scale. 1 = no CFG (standard for training)
            "tiled": False,  # Tiled generation for high-res videos (not used during training)
            "rand_device": self.pipe.device,  # Device for random number generation (CPU for determinism)
            "use_gradient_checkpointing": self.use_gradient_checkpointing,  # Enable gradient checkpointing (saves memory)
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,  # Offload checkpointed activations to CPU
            "cfg_merge": False,  # Merge positive/negative in one forward pass (not used in training)
            "vace_scale": 1,  # VACE (Visual Audio Condition Enhancement) scale (1 = normal)
            "max_timestep_boundary": self.max_timestep_boundary,  # Max noise level for training (1.0 = fully noised)
            "min_timestep_boundary": self.min_timestep_boundary,  # Min noise level for training (0.0 = clean)
        }

        # ===== STEP 4: Add Extra Inputs (Conditional) =====
        # Extra inputs specified via --extra_inputs command-line argument
        # Common examples:
        # - "input_audio": Audio file path for S2V (CRITICAL for speech-to-video!)
        # - "input_image": First frame for image conditioning
        # - "end_image": Last frame for end frame conditioning
        # - "reference_image": Reference image for face/style consistency
        for extra_input in self.extra_inputs:  # Iterate through extra inputs (e.g., ["input_audio", "input_image"])
            if extra_input == "input_image":  # First frame conditioning
                inputs_shared["input_image"] = data["video"][0]  # Use first frame from video as input image
            elif extra_input == "end_image":  # Last frame conditioning
                inputs_shared["end_image"] = data["video"][-1]  # Use last frame from video as end image
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":  # Reference image (first frame of reference video)
                inputs_shared[extra_input] = data[extra_input][0]  # Extract first frame if reference is a video
            else:  # Generic extra input (e.g., "input_audio")
                inputs_shared[extra_input] = data[extra_input]  # Copy directly from dataset batch
                # For S2V: inputs_shared["input_audio"] = data["audio"] (path to audio file)

        # ===== STEP 5: Run Pipeline Units to Process Inputs =====
        # THIS IS WHERE THE MAGIC HAPPENS!
        # Each unit in self.pipe.units processes the inputs sequentially:
        #
        # For S2V, the units are typically:
        # 1. WanVideoUnit_S2V (take_over=True):
        #    - Extracts audio from inputs_shared["input_audio"]
        #    - Encodes audio → audio_embeds (shape: [1, T_audio, 1280])
        #    - Adds audio_embeds to inputs_posi
        # 2. WanVideoUnit_PromptEmbedder (seperate_cfg=True):
        #    - Encodes prompt → prompt_embeds using CLIP
        #    - Adds prompt_embeds to inputs_posi
        # 3. WanVideoUnit_VAEEncoder (normal mode):
        #    - Encodes video → latents using VAE
        #    - Adds latents to inputs_shared
        # 4. WanVideoUnit_ShapeChecker (normal mode):
        #    - Validates all shapes are correct
        #
        # After all units run, inputs contain everything needed for training:
        # - latents: Encoded video (shape: [1, 16, 81, H/8, W/8])
        # - prompt_embeds: Encoded text (shape: [1, T_text, 768])
        # - audio_embeds: Encoded audio (shape: [1, T_audio, 1280])
        for unit in self.pipe.units:  # Iterate through all pipeline units (S2V, PromptEmbedder, VAEEncoder, etc.)
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)  # Run this unit, update all three dicts

        # ===== STEP 6: Merge and Return =====
        # Merge inputs_shared and inputs_posi into a single dict
        # This contains ALL inputs needed for training_loss():
        # - latents, prompt_embeds, audio_embeds, height, width, num_frames, etc.
        #
        # ❓ WHY MERGE? Why not return (inputs_shared, inputs_posi, inputs_nega)?
        #
        # Answer: During TRAINING with cfg_scale=1, we don't use CFG, so we only need
        # positive conditioning. The merged dict is more convenient for training_loss().
        #
        # During INFERENCE with cfg_scale>1, the pipeline keeps them separate to:
        # 1. Run DiT twice: once with inputs_posi, once with inputs_nega
        # 2. Combine predictions: pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
        #
        # But for training, we just need one forward pass, so merge them!
        return {**inputs_shared, **inputs_posi}  # Merge shared and positive inputs (negative not needed for training with cfg_scale=1)



    # ========================================================================
    # METHOD: forward() - Compute Training Loss
    # ========================================================================
    # PURPOSE: Main forward pass for training - compute diffusion loss
    #
    # PROCESS:
    # 1. Preprocess inputs (if not already done)
    # 2. Gather trainable models (e.g., dit)
    # 3. Call pipe.training_loss() to compute diffusion loss
    #
    # INPUT: data (dict from UnifiedDataset batch)
    # OUTPUT: loss (scalar tensor for backpropagation)
    #
    # WHAT training_loss() DOES:
    # 1. Add random noise to latents (based on random timestep t)
    # 2. Forward pass through DiT to predict noise
    # 3. Compute MSE loss between predicted noise and actual noise
    # 4. Return loss for backpropagation
    # ========================================================================
    def forward(self, data, inputs=None):
        # ===== STEP 1: Preprocess Inputs (if needed) =====
        if inputs is None: inputs = self.forward_preprocess(data)  # If inputs not provided, run preprocessing (usually called once per batch)
        # At this point, inputs contains:
        # - latents: [1, 16, 81, H/8, W/8] (VAE-encoded video)
        # - prompt_embeds: [1, T_text, 768] (CLIP-encoded text)
        # - audio_embeds: [1, T_audio, 1280] (wav2vec2-encoded audio, for S2V)
        # - height, width, num_frames, etc.

        # ===== STEP 2: Gather In-Iteration Models =====
        # "in_iteration_models" are models that run INSIDE the training loop
        # For Wan Video, this is typically just ["dit"]
        # Other models (vae, text_encoder, audio_encoder) run outside the loop (in preprocessing)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}  # Create dict: {"dit": self.pipe.dit}

        # ===== STEP 3: Compute Training Loss =====
        # pipe.training_loss() implements the standard diffusion training objective:
        # 1. Sample random timestep t ~ Uniform[min_timestep, max_timestep]
        # 2. Add noise to latents: noisy_latents = sqrt(alpha_t) * latents + sqrt(1-alpha_t) * noise
        # 3. Forward pass: noise_pred = dit(noisy_latents, t, prompt_embeds, audio_embeds)
        # 4. Compute loss: MSE(noise_pred, noise)
        # 5. Return loss for backpropagation
        #
        # Gradients will ONLY flow through trainable parts (e.g., LoRA layers in DiT)
        # All other models are frozen (no gradients)
        loss = self.pipe.training_loss(**models, **inputs)  # Pass models and inputs to training_loss()

        return loss  # Return scalar loss for optimizer.backward()


# ============================================================================
# MAIN: Training Script Entry Point
# ============================================================================
# PURPOSE: Set up dataset, model, and launch training
#
# WHAT HAPPENS HERE:
# 1. Parse command-line arguments (model paths, dataset paths, training configs)
# 2. Create UnifiedDataset (handles data loading and preprocessing)
# 3. Create WanTrainingModule (wraps pipeline for training)
# 4. Create ModelLogger (saves checkpoints during training)
# 5. Launch training with Accelerate (distributed training framework)
#
# TYPICAL COMMAND:
#   accelerate launch train.py \
#     --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:*.safetensors,..." \
#     --dataset_base_path /path/to/dataset \
#     --dataset_metadata_path /path/to/metadata.csv \
#     --extra_inputs input_audio \
#     --trainable_models dit \
#     --lora_base_model dit \
#     --lora_rank 32 \
#     --output_path ./outputs \
#     --num_frames 81 \
#     --max_pixels 1048576
# ============================================================================

if __name__ == "__main__":
    # ===== STEP 1: Parse Command-Line Arguments =====
    parser = wan_parser()  # Create argument parser with Wan Video-specific defaults
    args = parser.parse_args()  # Parse arguments from command line
    # Common args:
    # - args.model_id_with_origin_paths: Models to load (e.g., "Wan-AI/Wan2.2-S2V-14B:*.safetensors")
    # - args.dataset_base_path: Root directory of dataset (e.g., "/data/renderme360")
    # - args.dataset_metadata_path: CSV file with video/audio paths and prompts
    # - args.extra_inputs: Extra inputs to include (e.g., "input_audio" for S2V)
    # - args.trainable_models: Models to train (e.g., "dit")
    # - args.lora_base_model: Model to add LoRA to (e.g., "dit")
    # - args.num_frames: Number of frames per video (e.g., 81)
    # - args.output_path: Where to save checkpoints

    # ===== STEP 2: Create Dataset =====
    # UnifiedDataset is a flexible dataset loader that:
    # 1. Reads metadata CSV with columns: video_path, audio_path, prompt, etc.
    # 2. Loads video frames and converts to PIL.Image
    # 3. Applies preprocessing (crop, resize, format conversion)
    # 4. Returns dict: {"video": [PIL.Image, ...], "audio": "path.wav", "prompt": "..."}
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,  # Root path for dataset (e.g., "/data/renderme360")
        metadata_path=args.dataset_metadata_path,  # CSV file with data paths (e.g., "train.csv")
        repeat=args.dataset_repeat,  # Repeat dataset N times per epoch (for small datasets)
        data_file_keys=args.data_file_keys.split(","),  # Keys to load from CSV (e.g., ["video", "audio", "prompt"])

        # Main data operator: processes the primary data (video in this case)
        # This applies to all videos UNLESS they match a special_operator_map key
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,  # Base path for resolving relative paths
            max_pixels=args.max_pixels,  # Max resolution (e.g., 1048576 = 1024x1024). Videos resized if larger
            height=args.height,  # Target height (None = dynamic based on max_pixels)
            width=args.width,  # Target width (None = dynamic based on max_pixels)
            height_division_factor=16,  # Height must be divisible by 16 (VAE constraint)
            width_division_factor=16,  # Width must be divisible by 16 (VAE constraint)
            num_frames=args.num_frames,  # Number of frames to load (e.g., 81)
            time_division_factor=4,  # (num_frames - 1) must be divisible by 4 (temporal compression)
            time_division_remainder=1,  # num_frames % 4 must equal 1 (e.g., 81, 85, 89)
        ),

        # Special operators: custom processing for specific data types
        # Example: animate_face_video uses fixed 512x512 resolution
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
            # ToAbsolutePath: Convert relative path to absolute
            # LoadVideo: Load video with num_frames, time_division_factor=4, time_division_remainder=1
            # ImageCropAndResize: Crop+resize each frame to 512x512 (center crop, divisible by 16)
        }
    )

    # ===== STEP 3: Create Training Module =====
    # WanTrainingModule wraps WanVideoPipeline for training
    # It handles model loading, LoRA setup, and forward pass
    model = WanTrainingModule(
        model_paths=args.model_paths,  # LEGACY: comma-separated model paths
        model_id_with_origin_paths=args.model_id_with_origin_paths,  # NEW: "model_id:pattern,..." format
        trainable_models=args.trainable_models,  # Which models to train (e.g., "dit")
        lora_base_model=args.lora_base_model,  # Which model to add LoRA to (e.g., "dit")
        lora_target_modules=args.lora_target_modules,  # LoRA target modules (e.g., "q,k,v,o,ffn.0,ffn.2")
        lora_rank=args.lora_rank,  # LoRA rank (e.g., 32)
        lora_checkpoint=args.lora_checkpoint,  # Resume from existing LoRA checkpoint (optional)
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,  # Offload checkpointed activations to CPU
        extra_inputs=args.extra_inputs,  # Extra inputs (e.g., "input_audio" for S2V)
        max_timestep_boundary=args.max_timestep_boundary,  # Max noise level (1.0 = fully noised)
        min_timestep_boundary=args.min_timestep_boundary,  # Min noise level (0.0 = clean)
    )

    # ===== STEP 4: Create Model Logger =====
    # ModelLogger handles checkpoint saving during training
    # It saves checkpoints periodically (e.g., every N steps or every epoch)
    model_logger = ModelLogger(
        args.output_path,  # Output directory for checkpoints (e.g., "./outputs")
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt  # Remove "module." prefix (for DataParallel/DistributedDataParallel)
    )

    # ===== STEP 5: Launch Training =====
    # launch_training_task() is a wrapper around Accelerate's training loop
    # It handles:
    # - Distributed training setup (multi-GPU, multi-node)
    # - Optimizer and scheduler creation
    # - Training loop (forward, backward, optimizer step)
    # - Checkpoint saving via model_logger
    # - Logging (loss, learning rate, etc.)
    #
    # This will run until:
    # - args.num_training_steps steps completed, OR
    # - args.num_epochs epochs completed
    launch_training_task(dataset, model, model_logger, args=args)  # Start training!


# ============================================================================
# KEY TAKEAWAYS: Understanding the Training Script
# ============================================================================
"""
✅ WHAT YOU LEARNED FROM THIS FILE:

1. **Training Flow**:
   UnifiedDataset (load data) → forward_preprocess() (prepare inputs) →
   Pipeline units (process inputs) → training_loss() (compute loss) →
   Optimizer (update weights)

2. **Three Input Dictionaries**:
   - inputs_shared: CFG-insensitive (video, dimensions, configs)
   - inputs_posi: Positive conditioning (prompt, audio)
   - inputs_nega: Negative conditioning (for CFG during inference)

3. **Pipeline Units Process Inputs**:
   - WanVideoUnit_S2V: Extracts audio from inputs_shared["input_audio"],
     encodes to audio_embeds, adds to inputs_posi
   - WanVideoUnit_PromptEmbedder: Encodes prompt → prompt_embeds
   - WanVideoUnit_VAEEncoder: Encodes video → latents
   - Units run sequentially, updating the three dicts

4. **LoRA Training**:
   - trainable_models="dit" → only DiT gets gradients
   - lora_base_model="dit" → add LoRA layers to DiT
   - lora_target_modules="q,k,v,o,ffn.0,ffn.2" → which layers get LoRA
   - Result: only LoRA parameters trained (few MB vs 14B full model)

5. **For S2V Training**:
   - MUST specify --extra_inputs input_audio (critical!)
   - Dataset CSV must have "audio" column with audio file paths
   - UnifiedDataset loads: {"video": [...], "audio": "path.wav", "prompt": "..."}
   - forward_preprocess() puts audio in inputs_shared["input_audio"]
   - WanVideoUnit_S2V extracts and encodes audio

📊 EXAMPLE S2V TRAINING COMMAND:
```bash
accelerate launch train.py \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:*.safetensors,openai/clip-vit-large-patch14:*,facebook/wav2vec2-large-xlsr-53:*" \
  --dataset_base_path /data/renderme360 \
  --dataset_metadata_path /data/renderme360/train.csv \
  --extra_inputs input_audio \
  --trainable_models dit \
  --lora_base_model dit \
  --lora_rank 32 \
  --num_frames 81 \
  --max_pixels 1048576 \
  --output_path ./outputs
```

🎯 NEXT STEPS:
Now you understand the training script! Next, study:
- WanVideoPipeline.training_loss() - how loss is computed
- WanVideoUnit_S2V.process() - how audio is extracted and encoded
- UnifiedDataset internals - how data is loaded
"""
