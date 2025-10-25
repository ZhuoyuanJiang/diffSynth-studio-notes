# ============================================================================
# EDUCATIONAL COMMENTS FOR PHASE 2 LEARNING - Foundation Classes
# ============================================================================
"""
📚 COMMENTED VERSION OF diffsynth/utils/__init__.py

This file contains the FOUNDATIONAL CLASSES that all pipelines inherit from.
These are the "building blocks" you need to understand before studying wan_video_new.py.

📍 WHAT'S IN THIS FILE:
1. BasePipeline - Parent class for all pipelines (lines 38-177)
2. ModelConfig - Configuration for model loading (lines 181-241)
3. PipelineUnit - Base class for modular processors (lines 245-264)
4. PipelineUnitRunner - Orchestrates unit execution (lines 268-298)

🎯 WHY THIS MATTERS FOR S2V TRAINING:
- WanVideoPipeline inherits from BasePipeline
- ModelConfig is used to load wav2vec2 audio encoder
- WanVideoUnit_S2V inherits from PipelineUnit
- PipelineUnitRunner calls WanVideoUnit_S2V.process()

📖 READING ORDER:
1. BasePipeline (understand the foundation)
2. ModelConfig (understand model loading)
3. PipelineUnit (understand modular architecture)
4. PipelineUnitRunner (understand how units are executed)
"""

import torch, warnings, glob, os
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
import numpy as np
from PIL import Image
from typing import Optional


# ============================================================================
# CLASS 1: BasePipeline - The Foundation for All Pipelines
# ============================================================================
# PURPOSE: Parent class for all diffusion pipelines in DiffSynth-Studio
#
# WHAT IT PROVIDES:
# - Device management (GPU/CPU)
# - Shape validation (ensure dimensions meet model requirements)
# - Image/video preprocessing (PIL → torch.Tensor)
# - VRAM management (offload models when not in use)
# - Training utilities (freeze_except for LoRA)
#
# WHO USES IT:
# - WanVideoPipeline (for S2V, Animate, etc.)
# - FluxImagePipeline
# - SDImagePipeline
# - All other pipelines in DiffSynth-Studio
# ============================================================================

class BasePipeline(torch.nn.Module):
    """
    Base class for all diffusion pipelines.

    Provides common functionality for:
    - Managing device (CPU/GPU) and precision
    - Validating and adjusting dimensions
    - Preprocessing images/videos
    - Managing VRAM (offloading models)
    """

    def __init__(
        self,
        device="cuda",                      # Device to run on (cuda/cpu)
        torch_dtype=torch.float16,          # Precision for intermediate variables
        height_division_factor=64,          # Height must be divisible by this
        width_division_factor=64,           # Width must be divisible by this
        time_division_factor=None,          # For video: (num_frames - remainder) divisible by this
        time_division_remainder=None,       # For video: num_frames % time_division_factor == this
    ):
        """
        Initialize the base pipeline.

        Args:
            device: Device to store intermediate tensors (NOT where models are stored)
            torch_dtype: Data type for intermediate tensors
            height_division_factor: Height constraint (e.g., 16 for Wan Video)
            width_division_factor: Width constraint (e.g., 16 for Wan Video)
            time_division_factor: Frame count constraint (e.g., 4 for Wan Video)
            time_division_remainder: Required remainder (e.g., 1 for Wan Video)

        Example for Wan Video:
            height_division_factor = 16  → height must be divisible by 16
            width_division_factor = 16   → width must be divisible by 16
            time_division_factor = 4     → (num_frames - 1) divisible by 4
            time_division_remainder = 1  → num_frames % 4 == 1 (81, 85, 89, etc.)
        """
        super().__init__()

        # ===== DEVICE AND PRECISION =====
        # IMPORTANT: These are for INTERMEDIATE variables (tensors created during generation)
        # Models can be loaded to different devices dynamically (via load_models_to_device)
        self.device = device            # e.g., "cuda" or "cpu"
        self.torch_dtype = torch_dtype  # e.g., torch.bfloat16 or torch.float16

        # ===== SHAPE CONSTRAINTS =====
        # These enforce dimension requirements for the model architecture
        self.height_division_factor = height_division_factor      # e.g., 16
        self.width_division_factor = width_division_factor        # e.g., 16
        self.time_division_factor = time_division_factor          # e.g., 4
        self.time_division_remainder = time_division_remainder    # e.g., 1

        # ===== VRAM MANAGEMENT =====
        # When True, models are automatically offloaded to CPU when not needed
        # This allows running large models (14B params) on limited VRAM
        self.vram_management_enabled = False
        
        
    def to(self, *args, **kwargs):
        """
        Override torch.nn.Module.to() to update device and dtype attributes.

        When you call pipe.to("cuda") or pipe.to(torch.bfloat16),
        this updates self.device and self.torch_dtype accordingly.
        """
        # Parse arguments (same as torch.nn.Module.to)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        # Update internal device tracking
        if device is not None:
            self.device = device

        # Update internal dtype tracking
        if dtype is not None:
            self.torch_dtype = dtype

        # Call parent's to() method (moves model parameters)
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        """
        Validate and adjust dimensions to meet model constraints.

        Args:
            height: Desired height
            width: Desired width
            num_frames: Desired number of frames (optional, for video)

        Returns:
            Adjusted (height, width) or (height, width, num_frames)

        Example for Wan Video (division_factor=16, time_factor=4, time_remainder=1):
            Input: 450 x 830 → Output: 448 x 832 (rounded to nearest 16)
            Input: 80 frames → Output: 81 frames (80 % 4 != 1, so round to 81)
            Input: 85 frames → Output: 85 frames (85 % 4 == 1, already valid)
        """
        # ===== CHECK HEIGHT =====
        if height % self.height_division_factor != 0:  # If height NOT divisible (e.g., 450 % 16 = 2, not 0)
            # Round up to nearest multiple of height_division_factor
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor  # Formula: ceil(height / factor) * factor. Example: 450 → (450+15)//16 = 29, 29*16 = 464
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")  # Print warning

        # ===== CHECK WIDTH =====
        if width % self.width_division_factor != 0:  # If width NOT divisible (e.g., 830 % 16 = 14, not 0)
            # Round up to nearest multiple of width_division_factor
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor  # Formula: ceil(width / factor) * factor. Example: 830 → (830+15)//16 = 52, 52*16 = 832
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")  # Print warning

        # ===== CHECK NUM_FRAMES (if video) =====
        if num_frames is None:  # If image (not video)
            return height, width  # Return adjusted dimensions (2 values)
        else:  # If video
            # Check if num_frames satisfies: num_frames % time_division_factor == time_division_remainder
            # For Wan Video: num_frames % 4 == 1 (valid: 81, 85, 89, 93, etc.)
            if num_frames % self.time_division_factor != self.time_division_remainder:  # Example: 80 % 4 = 0, not 1
                # Adjust to meet constraint
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder  # Formula: ceil(num_frames / factor) * factor + remainder. Example: 80 → (80+3)//4 = 20, 20*4 = 80, 80+1 = 81
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")  # Print warning
            return height, width, num_frames  # Return adjusted dimensions (3 values)


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        """
        Convert PIL.Image to torch.Tensor.

        Process: PIL (0-255, H×W×C) → Tensor (normalized, B×C×H×W)
        """
        # PIL → numpy → tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))  # Convert PIL.Image to numpy array, then to torch.Tensor. Shape: (H, W, 3), values: [0, 255]
        # Move to device and convert dtype
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)  # Move tensor to specified device (e.g., cuda) and convert to specified dtype (e.g., bfloat16)
        # Normalize from [0, 255] to [min_value, max_value] (usually [-1, 1])
        image = image * ((max_value - min_value) / 255) + min_value  # Linear normalization. Example: [0, 255] → [-1, 1]: pixel * (2/255) + (-1)
        # Rearrange: H W C → B C H W
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))  # Rearrange dimensions using einops. (H, W, 3) → (1, 3, H, W). Adds batch dimension, moves channels first
        return image  # Return tensor of shape (1, 3, H, W) with values in [min_value, max_value]


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        """
        Convert list of PIL.Image (video frames) to torch.Tensor.

        Process: [PIL, PIL, ...] → preprocess each → stack → (B×C×T×H×W)
        """
        # Preprocess each frame individually
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]  # Apply preprocess_image() to each PIL.Image frame. Result: list of tensors, each (1, 3, H, W)
        # Stack along time dimension
        video = torch.stack(video, dim=pattern.index("T") // 2)  # Stack list of frames into single tensor. pattern.index("T")//2 finds position of T dimension. Example: "B C T H W" → T at index 2, dim=2//2=1, so stack at dim=1 → (1, num_frames, 3, H, W) → then reorder to (1, 3, num_frames, H, W)
        return video  # Return tensor of shape (1, 3, T, H, W) with values in [min_value, max_value]


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        """
        Convert VAE output tensor to PIL.Image.

        Reverse of preprocess_image(): Tensor (normalized) → PIL (0-255, H×W×C)
        """
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":  # If tensor is not already in H×W×C format (e.g., it's B×C×H×W)
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")  # Rearrange dimensions and reduce batch/other dims by averaging. Example: (1, 3, H, W) → (H, W, 3)
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)  # Denormalize from [min_value, max_value] to [0, 255]. Example: [-1, 1] → [0, 255]: (pixel - (-1)) * (255/2) = (pixel + 1) * 127.5
        image = image.to(device="cpu", dtype=torch.uint8)  # Move to CPU and convert to uint8 (0-255 integer values)
        image = Image.fromarray(image.numpy())  # Convert numpy array to PIL.Image
        return image  # Return PIL.Image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        """
        Convert VAE output tensor to list of PIL.Image (video frames).

        Reverse of preprocess_video(): Tensor (B×C×T×H×W) → [PIL, PIL, ...]
        """
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":  # If tensor is not already in T×H×W×C format (e.g., it's B×C×T×H×W)
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")  # Rearrange dimensions and reduce batch/channel dims by averaging. Example: (1, 3, T, H, W) → (T, H, W, 3)
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]  # Convert each frame (H, W, 3) to PIL.Image using vae_output_to_image(). Result: list of PIL.Image objects
        return video  # Return list of PIL.Image frames


    # ========================================================================
    # ⭐ CRITICAL FOR S2V: load_models_to_device() - VRAM Management
    # ========================================================================
    # PURPOSE: Load specified models to GPU, offload others to CPU
    # WHY: Allows running 14B models on limited VRAM by loading models on-demand
    # EXAMPLE: pipe.load_models_to_device(["audio_encoder"]) → only audio_encoder on GPU
    # ========================================================================
    def load_models_to_device(self, model_names=[]):  # model_names: list of models to load to GPU (e.g., ["audio_encoder", "vae"])
        if self.vram_management_enabled:  # Only manage VRAM if enabled
            # STEP 1: Offload models NOT in model_names to CPU
            for name, model in self.named_children():  # Iterate through all child models (dit, vae, audio_encoder, etc.)
                if name not in model_names:  # If this model is NOT in the requested list
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:  # Check if model supports layer-by-layer offloading
                        for module in model.modules():  # Iterate through all layers
                            if hasattr(module, "offload"):  # If layer has offload method
                                module.offload()  # Offload this layer to CPU
                    else:  # Model doesn't support layer-by-layer, use simple offload
                        model.cpu()  # Move entire model to CPU
            torch.cuda.empty_cache()  # Free unused VRAM

            # STEP 2: Load models IN model_names to GPU
            for name, model in self.named_children():  # Iterate through all child models again
                if name in model_names:  # If this model IS in the requested list
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:  # Check if model supports layer-by-layer loading
                        for module in model.modules():  # Iterate through all layers
                            if hasattr(module, "onload"):  # If layer has onload method
                                module.onload()  # Load this layer to GPU
                    else:  # Model doesn't support layer-by-layer, use simple load
                        model.to(self.device)  # Move entire model to GPU


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        """
        Generate random Gaussian noise for diffusion process.

        Args:
            shape: Shape of noise tensor (e.g., (1, 16, 81, 90, 160) for latents)
            seed: Random seed for reproducibility (None = non-deterministic)
            rand_device: Device to generate random numbers on (CPU for determinism)
            rand_torch_dtype: Dtype for random generation (float32 for precision)
            device: Target device (GPU) to move noise to
            torch_dtype: Target dtype (bfloat16/float16) for memory efficiency
        """
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)  # Create random generator with seed if provided. If seed=None, use non-deterministic generation. If seed provided, create generator on rand_device and set seed for reproducibility
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)  # Generate random Gaussian noise (mean=0, std=1) with given shape. Uses generator for seeded randomness, created on rand_device with rand_torch_dtype for precision
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)  # Move noise to target device and convert to target dtype. Use provided device/dtype or fall back to self.device/self.torch_dtype
        return noise  # Return noise tensor ready for diffusion


    def enable_cpu_offload(self):
        """
        Enable VRAM management (deprecated method name).

        Legacy method - use enable_vram_management() instead.
        """
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")  # Warn user about deprecated name
        self.vram_management_enabled = True  # Enable VRAM management (offload/onload models dynamically)


    def get_vram(self):
        """
        Get total VRAM available on the current device.

        Returns: Total VRAM in GB
        """
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)  # Get total VRAM info for self.device. [1] is total VRAM (not free), convert bytes to GB by dividing by 1024³
    
    
    # ========================================================================
    # ⭐ CRITICAL FOR LORA TRAINING: freeze_except() - Selective Freezing
    # ========================================================================
    # PURPOSE: Freeze all models EXCEPT those in model_names (for training)
    # WHY: In LoRA training, only DiT is trainable, all others are frozen
    # EXAMPLE: pipe.freeze_except(["dit"]) → only DiT trainable, rest frozen
    # ========================================================================
    def freeze_except(self, model_names):  # model_names: list of models to keep trainable (e.g., ["dit"])
        for name, model in self.named_children():  # Iterate through all child models (dit, vae, text_encoder, etc.)
            if name in model_names:  # If this model should be trainable
                model.train()  # Set to training mode (enables dropout, batchnorm updates, etc.)
                model.requires_grad_(True)  # Enable gradient computation for this model
            else:  # If this model should be frozen
                model.eval()  # Set to evaluation mode (disables dropout, batchnorm updates, etc.)
                model.requires_grad_(False)  # Disable gradient computation (saves memory and computation)
                
    
    def blend_with_mask(self, base, addition, mask):
        """
        Blend two tensors using a mask (for inpainting).

        Formula: base * (1 - mask) + addition * mask
        - Where mask=0: keep base
        - Where mask=1: use addition
        """
        return base * (1 - mask) + addition * mask  # Linear interpolation between base and addition controlled by mask. mask=0 → base, mask=1 → addition, mask=0.5 → 50% blend


    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        """
        Perform one denoising step in the diffusion process.

        Args:
            scheduler: Noise scheduler (controls denoising trajectory)
            latents: Current latent state
            progress_id: Current step index in the denoising process
            noise_pred: Predicted noise from the model
            input_latents: Original latents (for inpainting)
            inpaint_mask: Mask for inpainting (None = no inpainting)

        Returns: Next latent state after denoising step
        """
        timestep = scheduler.timesteps[progress_id]  # Get the timestep value for this progress step from scheduler's timestep schedule
        if inpaint_mask is not None:  # If inpainting mode (mask provided)
            noise_pred_expected = scheduler.return_to_timestep(scheduler.timesteps[progress_id], latents, input_latents)  # Calculate expected noise at this timestep for the original input_latents (what noise should be at current timestep)
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)  # Blend expected noise (from original) with predicted noise (from model) using mask. Masked regions use predicted noise, unmasked regions use expected noise
        latents_next = scheduler.step(noise_pred, timestep, latents)  # Apply denoising step: remove predicted noise from current latents to get next latents (one step closer to final image)
        return latents_next  # Return denoised latents for next iteration



# ============================================================================
# CLASS 2: ModelConfig - Configuration for Model Loading
# ============================================================================
# PURPOSE: Specifies HOW to load a model (local path or auto-download)
#
# TWO USAGE MODES:
# 1. Local file:
#    ModelConfig(path="/path/to/model.safetensors")
#
# 2. Auto-download from ModelScope/HuggingFace:
#    ModelConfig(
#        model_id="Wan-AI/Wan2.2-S2V-14B",
#        origin_file_pattern="diffusion_pytorch_model*.safetensors"
#    )
#
# USED IN TRAINING SCRIPTS:
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:*.safetensors,..."
#   This gets parsed into multiple ModelConfig objects
# ============================================================================

@dataclass
class ModelConfig:
    # ===== OPTION 1: Direct path (if model already downloaded) =====
    path: Union[str, list[str]] = None  # Direct path to model file(s), e.g., "/path/to/model.safetensors"

    # ===== OPTION 2: Auto-download (if model needs to be downloaded) =====
    model_id: str = None  # ModelScope/HuggingFace repo ID, e.g., "Wan-AI/Wan2.2-S2V-14B"
    origin_file_pattern: Union[str, list[str]] = None  # File pattern to download, e.g., "*.safetensors" or "wav2vec2-large-xlsr-53-english/model.safetensors"
    download_resource: str = "ModelScope"  # Where to download from (ModelScope or HuggingFace)

    # ===== Advanced options =====
    offload_device: Optional[Union[str, torch.device]] = None  # Device to offload to (for VRAM management)
    offload_dtype: Optional[torch.dtype] = None  # Dtype for offloaded weights
    local_model_path: str = None  # Where to save downloads (default: "./models")
    skip_download: bool = False  # Skip download (for distributed training, only rank 0 downloads)

    def download_if_necessary(self, use_usp=False):  # use_usp: unified sequence parallel (distributed training)
        if self.path is None:  # If path not already set (need to download)
            # STEP 1: Validate inputs
            if self.model_id is None:  # Must have model_id to download
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")

            # STEP 2: Handle distributed training (only rank 0 downloads)
            if use_usp:  # If using distributed training
                import torch.distributed as dist
                skip_download = self.skip_download or dist.get_rank() != 0  # Only rank 0 downloads
            else:
                skip_download = self.skip_download

            # STEP 3: Determine if downloading folder or specific files
            if self.origin_file_pattern is None or self.origin_file_pattern == "":  # Download entire repo
                self.origin_file_pattern = ""
                allow_file_pattern = None  # No filter, download all
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):  # Download specific folder
                allow_file_pattern = self.origin_file_pattern + "*"  # Add wildcard
                is_folder = True
            else:  # Download specific file(s) matching pattern
                allow_file_pattern = self.origin_file_pattern  # e.g., "*.safetensors"
                is_folder = False

            # STEP 4: Download from ModelScope/HuggingFace
            if self.local_model_path is None:  # Set default download location
                self.local_model_path = "./models"
            if not skip_download:  # If this is rank 0 or single-GPU
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))  # Check what's already downloaded
                snapshot_download(  # Download from ModelScope
                    self.model_id,  # e.g., "Wan-AI/Wan2.2-S2V-14B"
                    local_dir=os.path.join(self.local_model_path, self.model_id),  # e.g., "./models/Wan-AI/Wan2.2-S2V-14B"
                    allow_file_pattern=allow_file_pattern,  # Which files to download
                    ignore_file_pattern=downloaded_files,  # Skip already downloaded files
                    local_files_only=False  # Download from remote if not local
                )

            # STEP 5: Wait for rank 0 to finish downloading (distributed training)
            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])  # All ranks wait here until rank 0 finishes

            # STEP 6: Set self.path to downloaded location
            if is_folder:  # If downloaded a folder
                self.path = os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern)  # Path to folder
            else:  # If downloaded specific files
                self.path = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))  # List of matching files
            if isinstance(self.path, list) and len(self.path) == 1:  # If only one file, convert list to string
                self.path = self.path[0]



# ============================================================================
# CLASS 3: PipelineUnit - Base Class for Modular Processors
# ============================================================================
# PURPOSE: Modular component that processes inputs in the pipeline
#
# THREE PROCESSING MODES:
# 1. take_over=True (S2V uses this):
#    - Unit receives ALL THREE dicts: inputs_shared, inputs_posi, inputs_nega
#    - Unit can modify all of them and return all three
#    - Example: S2V extracts audio from inputs_shared, adds audio_embeds to inputs_posi
#
# 2. seperate_cfg=True:
#    - Unit processes positive and negative sides separately (for CFG)
#    - Useful when positive/negative need different processing
#
# 3. Normal mode:
#    - Unit only processes inputs_shared
#    - Simple transformation of shared inputs
#
# WHO USES IT:
# - WanVideoUnit_S2V (take_over=True)
# - WanVideoUnit_PromptEmbedder (seperate_cfg=True)
# - WanVideoUnit_ShapeChecker (normal mode)
# ============================================================================

class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,  # If True, process positive/negative separately
        take_over: bool = False,  # If True, unit controls all three input dicts (inputs_shared, inputs_posi, inputs_nega)
        input_params: tuple[str] = None,  # Params to extract from inputs_shared (normal mode)
        input_params_posi: dict[str, str] = None,  # Params from inputs_posi (seperate_cfg mode)
        input_params_nega: dict[str, str] = None,  # Params from inputs_nega (seperate_cfg mode)
        onload_model_names: tuple[str] = None  # Models to load to GPU before processing (e.g., ("audio_encoder", "vae"))
    ):
        self.seperate_cfg = seperate_cfg  # Store processing mode flag
        self.take_over = take_over  # Store take-over flag
        self.input_params = input_params  # Store input params (normal mode)
        self.input_params_posi = input_params_posi  # Store positive params (CFG mode)
        self.input_params_nega = input_params_nega  # Store negative params (CFG mode)
        self.onload_model_names = onload_model_names  # Store models to load


    def process(self, pipe: BasePipeline, inputs: dict, positive=True, **kwargs) -> dict:  # Main processing method (must be implemented by subclasses)
        raise NotImplementedError("`process` is not implemented.")  # Subclasses must implement this



# ============================================================================
# CLASS 4: PipelineUnitRunner - Orchestrates Unit Execution
# ============================================================================
# PURPOSE: Executes units and manages the three input dictionaries
#
# THE THREE INPUT DICTS:
# - inputs_shared: CFG-insensitive inputs (height, width, num_frames, etc.)
# - inputs_posi: Positive conditioning (prompt_embeds, audio_embeds, etc.)
# - inputs_nega: Negative conditioning (for classifier-free guidance)
#
# HOW IT WORKS:
# - For each unit in pipeline.units:
#     - Runner calls unit.process() based on unit's mode
#     - Updates input dicts with unit's outputs
#     - Passes updated dicts to next unit
#
# USED BY: WanVideoPipeline (and all other pipelines)
# ============================================================================

class PipelineUnitRunner:
    def __init__(self):
        pass  # No initialization needed

    def __call__(self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:  # Execute one unit
        if unit.take_over:  # MODE 1: TAKE_OVER (S2V uses this)
            # Let unit take full control of all three dicts
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)  # Unit receives all three, returns all three
        elif unit.seperate_cfg:  # MODE 2: SEPARATE CFG (for prompt embedding, etc.)
            # Process positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}  # Extract params from inputs_posi
            if unit.input_params is not None:  # Also extract params from inputs_shared
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)  # Process positive side
            inputs_posi.update(processor_outputs)  # Add outputs to inputs_posi
            # Process negative side
            if inputs_shared["cfg_scale"] != 1:  # Only process negative if CFG is enabled
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}  # Extract params from inputs_nega
                if unit.input_params is not None:  # Also extract params from inputs_shared
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)  # Process negative side
                inputs_nega.update(processor_outputs)  # Add outputs to inputs_nega
            else:  # CFG disabled, copy positive to negative
                inputs_nega.update(processor_outputs)
        else:  # MODE 3: NORMAL (simple transformation)
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}  # Extract params from inputs_shared
            processor_outputs = unit.process(pipe, **processor_inputs)  # Process inputs
            inputs_shared.update(processor_outputs)  # Add outputs back to inputs_shared
        return inputs_shared, inputs_posi, inputs_nega  # Return updated dicts


# ============================================================================
# KEY TAKEAWAYS: Understanding the Foundation
# ============================================================================
"""
✅ WHAT YOU LEARNED FROM THIS FILE:

1. **BasePipeline** - The foundation for all pipelines:
   - Manages device (cuda/cpu) and precision (bfloat16/float16)
   - Enforces shape constraints (height, width, num_frames)
   - load_models_to_device(): VRAM management (offload/onload models on-demand)
   - freeze_except(): Selective freezing for LoRA training

2. **ModelConfig** - How models are loaded:
   - Two modes: local path OR auto-download from ModelScope/HF
   - download_if_necessary(): Downloads models and sets self.path
   - Used in training scripts: --model_id_with_origin_paths

3. **PipelineUnit** - Modular processor base class:
   - Three modes:
     * take_over=True: Unit controls all three dicts (S2V uses this)
     * seperate_cfg=True: Process positive/negative separately (CFG)
     * Normal: Only process inputs_shared
   - Each unit implements process() method

4. **PipelineUnitRunner** - Orchestrates unit execution:
   - Manages three dicts: inputs_shared, inputs_posi, inputs_nega
   - Calls units sequentially
   - Updates dicts with each unit's outputs

📊 THE FLOW IN S2V TRAINING:
1. Training script creates inputs_shared["input_audio"] = data["audio"]
2. PipelineUnitRunner calls each unit in sequence
3. WanVideoUnit_S2V (take_over=True) extracts audio from inputs_shared
4. S2V unit processes audio → audio_embeds
5. S2V unit adds audio_embeds to inputs_posi
6. DiT receives inputs_posi["audio_embeds"] as conditioning

🎯 NEXT STEPS:
Now you understand the foundation! Next, study:
- WanVideoPipeline in commented_wan_video_new.py
- WanVideoUnit_S2V.process() method (the key!)
- How training script uses these classes
"""
