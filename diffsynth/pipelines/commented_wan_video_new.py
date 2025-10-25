import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader


# ============================================================================
# EDUCATIONAL COMMENTS FOR PHASE 2 LEARNING - Wan S2V Training
# ============================================================================
"""
📚 COMMENTED VERSION OF wan_video_new.py FOR LEARNING

This file contains detailed line-by-line comments for Phase 2 of your learning plan.
Focus on understanding how audio flows through the S2V pipeline unit.

📍 WHAT'S COMMENTED:
1. WanVideoPipeline class (lines 70-157) - Main pipeline structure
2. WanVideoUnit_S2V class (lines 1051-1286) - ⭐ THE KEY UNIT FOR S2V
3. WanVideoPostUnit_S2V class (lines 1325-1378) - Post-processing

📖 HOW TO USE THIS FILE:
1. Read the header comments explaining BasePipeline, ModelConfig, PipelineUnit
2. Study WanVideoPipeline.__init__() to see how units are registered
3. Focus on WanVideoUnit_S2V.process() - this is THE KEY method!
4. Trace the flow: input_audio → process() → audio_embeds → inputs_posi

🎯 LEARNING OBJECTIVES (PHASE 2):
- Q: Where does the unit expect to find input_audio? → inputs_shared
- Q: What does the unit add to inputs_posi? → audio_embeds
- Q: What does the unit add to inputs_nega? → 0.0 * audio_embeds (for CFG)
- Q: Is pose video required? → NO (optional)

KEY CLASSES YOU NEED TO UNDERSTAND (imported from diffsynth/utils/__init__.py):

1. BasePipeline:
   - Parent class for all pipelines
   - Manages device (cuda/cpu), torch_dtype (precision)
   - Has shape constraints (height_division_factor, width_division_factor, time_division_factor)
   - Methods: check_resize_height_width(), preprocess_image/video(), load_models_to_device()
   - For Wan Video: height/width divisible by 16, num_frames % 4 == 1 (e.g., 81, 85, 89)

2. ModelConfig:
   - Configuration for loading models from ModelScope/HuggingFace
   - Two modes:
     a) Local: ModelConfig(path="/path/to/model.safetensors")
     b) Auto-download: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="*.safetensors")
   - Method: download_if_necessary() downloads and sets self.path

3. PipelineUnit:
   - Modular processor that transforms inputs
   - Three modes:
     a) take_over=True: Unit controls inputs_shared, inputs_posi, inputs_nega (S2V uses this)
     b) seperate_cfg=True: Process positive/negative separately for CFG
     c) Normal: Only process inputs_shared
   - Method: process() - must be implemented by subclasses

4. PipelineUnitRunner:
   - Orchestrates unit execution
   - Manages three dictionaries:
     - inputs_shared: CFG-insensitive (height, width, num_frames, etc.)
     - inputs_posi: Positive conditioning (prompt_embeds, audio_embeds, etc.)
     - inputs_nega: Negative conditioning (for classifier-free guidance)
"""


class WanVideoPipeline(BasePipeline):
    """
    Main pipeline for Wan Video models (Text-to-Video, Image-to-Video, Speech-to-Video, Animate)

    Inherits from BasePipeline and adds:
    - Model components (DiT, VAE, text encoder, audio encoder, etc.)
    - Pipeline units (modular processing steps)
    - Scheduler (controls diffusion process)
    """

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        # ===== INITIALIZE BASE PIPELINE =====
        # Set dimension constraints for Wan Video:
        # - height/width must be divisible by 16
        # - num_frames must satisfy: (num_frames - 1) % 4 == 0, i.e., num_frames % 4 == 1
        #   Valid: 81, 85, 89, 93, etc.
        super().__init__(
            device=device,                    # GPU device
            torch_dtype=torch_dtype,          # bfloat16 for memory efficiency
            height_division_factor=16,        # Height divisible by 16
            width_division_factor=16,         # Width divisible by 16
            time_division_factor=4,           # (num_frames - 1) divisible by 4
            time_division_remainder=1         # So num_frames % 4 == 1
        )

        # ===== DIFFUSION SCHEDULER =====
        # Controls noise scheduling and denoising process
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        # ===== TEXT PROMPTER =====
        # Handles text tokenization (uses T5 tokenizer)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)

        # ===== MODEL COMPONENTS (initialized as None, loaded later) =====
        self.text_encoder: WanTextEncoder = None              # T5 text encoder
        self.image_encoder: WanImageEncoder = None            # CLIP image encoder
        self.dit: WanModel = None                             # Main DiT (Diffusion Transformer)
        self.dit2: WanModel = None                            # Secondary DiT (for certain modes)
        self.vae: WanVideoVAE = None                          # VAE for encoding/decoding videos
        self.audio_encoder = None                             # wav2vec2 (for S2V only) ⭐
        self.audio_processor = None                           # Audio processor (for S2V only) ⭐
        self.motion_controller: WanMotionControllerModel = None  # Motion control
        self.vace: VaceWanModel = None                        # VACE model
        self.vace2: VaceWanModel = None                       # VACE model 2
        self.animate_adapter: WanAnimateAdapter = None        # Animate adapter

        # ===== MODELS USED DURING DENOISING ITERATIONS =====
        # These models are loaded to GPU during the denoising loop
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter")

        # ===== PIPELINE UNIT RUNNER =====
        # Orchestrates the execution of pipeline units
        self.unit_runner = PipelineUnitRunner()

        # ===== PIPELINE UNITS (ORDER MATTERS!) =====
        # Units process inputs sequentially. Each unit transforms the input dictionaries.
        self.units = [
            WanVideoUnit_ShapeChecker(),         # 1. Validate/fix dimensions (height, width, num_frames)
            WanVideoUnit_NoiseInitializer(),     # 2. Initialize latent noise
            WanVideoUnit_PromptEmbedder(),       # 3. Encode text prompt → prompt_embeds
            WanVideoUnit_S2V(),                  # 4. ⭐ Process audio → audio_embeds (S2V ONLY)
            WanVideoUnit_InputVideoEmbedder(),   # 5. Encode input video (for V2V)
            WanVideoUnit_ImageEmbedderVAE(),     # 6. Encode reference image with VAE
            WanVideoUnit_ImageEmbedderCLIP(),    # 7. Encode reference image with CLIP
            WanVideoUnit_ImageEmbedderFused(),   # 8. Fuse VAE+CLIP embeddings
            WanVideoUnit_FunControl(),           # 9. ControlNet conditioning
            WanVideoUnit_FunReference(),         # 10. Reference image conditioning
            WanVideoUnit_FunCameraControl(),     # 11. Camera movement control
            WanVideoUnit_SpeedControl(),         # 12. Motion speed control
            WanVideoUnit_VACE(),                 # 13. VACE conditioning
            WanVideoPostUnit_AnimateVideoSplit(),     # 14. Split animate video
            WanVideoPostUnit_AnimatePoseLatents(),    # 15. Process animate pose
            WanVideoPostUnit_AnimateFacePixelValues(), # 16. Process animate face
            WanVideoPostUnit_AnimateInpaint(),        # 17. Inpainting
            WanVideoUnit_UnifiedSequenceParallel(),   # 18. Distributed training setup
            WanVideoUnit_TeaCache(),             # 19. TeaCache acceleration
            WanVideoUnit_CfgMerger(),            # 20. Merge positive/negative for CFG
        ]

        # ===== POST UNITS (run after denoising, before VAE decoding) =====
        self.post_units = [
            WanVideoPostUnit_S2V(),  # Add motion context for S2V
        ]

        # ===== MODEL FORWARD FUNCTION =====
        # Function that runs the DiT model during denoising
        self.model_fn = model_fn_wan_video
    
    # ========================================================================
    # load_lora() - Load LoRA Weights into Model
    # ========================================================================
    # PURPOSE: Load trained LoRA weights into a module (usually DiT)
    # WHEN USED: During inference to apply fine-tuned LoRA, or during training for validation
    # TWO MODES:
    #   - hotload=True: Append LoRA weights to existing list (for multi-LoRA)
    #   - hotload=False: Load LoRA using GeneralLoRALoader (standard loading)
    # ========================================================================
    def load_lora(
        self,
        module: torch.nn.Module,  # Target module to load LoRA into (e.g., pipe.dit)
        lora_config: Union[ModelConfig, str] = None,  # Path to LoRA checkpoint OR ModelConfig
        alpha=1,  # LoRA strength multiplier (1.0 = full strength, 0.5 = half strength)
        hotload=False,  # If True, append to existing LoRA weights instead of replacing
        state_dict=None,  # Pre-loaded state dict (skips loading from file)
    ):
        # ===== STEP 1: Load LoRA state dict from file or use provided one =====
        # NOTE: We LOAD FROM file (reading), NOT save TO file
        if state_dict is None:  # If state_dict not already provided
            if isinstance(lora_config, str):  # If lora_config is a string (file path like "/path/to/my_lora.pth")
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)  # Load LoRA weights FROM that file path
            else:  # If lora_config is a ModelConfig object
                lora_config.download_if_necessary()  # Download file first from ModelScope/HF (if not already local)
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)  # Load LoRA weights FROM downloaded file
        else:  # If state_dict already provided (skip loading from file)
            lora = state_dict  # Use the provided state_dict directly

        # ===== STEP 2: Load LoRA weights into module =====
        if hotload:  # MODE 1: HOTLOAD - Append LoRA to existing list (for multi-LoRA support)
            for name, module in module.named_modules():  # Iterate through all submodules
                if isinstance(module, AutoWrappedLinear):  # If module is wrapped linear layer (supports LoRA lists)
                    lora_a_name = f'{name}.lora_A.default.weight'  # Construct LoRA A weight key
                    lora_b_name = f'{name}.lora_B.default.weight'  # Construct LoRA B weight key
                    if lora_a_name in lora and lora_b_name in lora:  # If LoRA weights exist for this layer
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)  # Append LoRA A (scaled by alpha)
                        module.lora_B_weights.append(lora[lora_b_name])  # Append LoRA B
        else:  # MODE 2: STANDARD LOAD - Replace existing LoRA weights
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)  # Create LoRA loader
            loader.load(module, lora, alpha=alpha)  # Load LoRA into module with specified alpha
        
    # ========================================================================
    # ⭐ CRITICAL FOR TRAINING: training_loss() - Compute Diffusion Loss
    # ========================================================================
    # PURPOSE: Compute the training loss for one batch (standard diffusion training)
    # CALLED BY: Training script's training step
    # HOW IT WORKS:
    #   1. Sample random timestep from scheduler
    #   2. Add noise to input_latents at that timestep
    #   3. Model predicts the noise
    #   4. Compute MSE loss between predicted and actual noise
    # ========================================================================
    def training_loss(self, **inputs):  # inputs contains: input_latents, noise, audio_embeds, prompt_embeds, etc.
        # ===== STEP 1: Sample random timestep for this training step =====
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)  # Max timestep (default: 1.0 * num_train_timesteps = full range)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)  # Min timestep (default: 0 * num_train_timesteps = start of range)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))  # Randomly sample timestep index in range [min, max)
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)  # Get actual timestep value from scheduler's timestep list and move to device

        # ===== STEP 2: Add noise to clean latents (forward diffusion) =====
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)  # Add noise to clean input_latents at sampled timestep. Result: noisy latents at timestep t
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)  # Compute what model should predict (usually the noise itself, but depends on scheduler)

        # ===== STEP 3: Model predicts noise =====
        noise_pred = self.model_fn(**inputs, timestep=timestep)  # Run model forward pass: DiT takes noisy latents + conditioning (audio_embeds, prompt_embeds) → predicts noise

        # ===== STEP 4: Compute loss =====
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())  # MSE loss between predicted noise and actual noise (convert to float32 for numerical stability)
        loss = loss * self.scheduler.training_weight(timestep)  # Apply timestep-dependent weighting (some schedulers weight timesteps differently)
        return loss  # Return scalar loss for backpropagation


    # ========================================================================
    # enable_vram_management() - Enable Layer-by-Layer Offloading
    # ========================================================================
    # PURPOSE: Enable aggressive VRAM optimization via layer-by-layer offloading
    # WHY: Allows running huge models (14B params) on limited VRAM (e.g., 24GB)
    # HOW: Wraps each layer (Linear, Conv, etc.) to auto-offload/onload during forward pass
    # TWO MODES:
    #   1. num_persistent_param_in_dit: Keep N params in GPU, offload rest
    #   2. vram_limit: Keep loading layers until VRAM limit reached
    # ========================================================================
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):  # vram_buffer: reserve 0.5GB free VRAM
        self.vram_management_enabled = True  # Set flag (BasePipeline checks this)

        # ===== STEP 1: Determine VRAM strategy =====
        if num_persistent_param_in_dit is not None:  # MODE 1: Keep N parameters in GPU
            vram_limit = None  # Disable VRAM limit (use param count instead)
        else:  # MODE 2: Use VRAM limit
            if vram_limit is None:  # If not specified, auto-detect
                vram_limit = self.get_vram()  # Get total VRAM (e.g., 24 GB for RTX 3090)
            vram_limit = vram_limit - vram_buffer  # Reserve buffer (e.g., 24 - 0.5 = 23.5 GB usable)
        # ===== STEP 2: Wrap text_encoder layers for VRAM management =====
        if self.text_encoder is not None:  # If text encoder loaded
            dtype = next(iter(self.text_encoder.parameters())).dtype  # Get model's stored dtype (e.g., bfloat16)
            enable_vram_management(  # Wrap text encoder layers
                self.text_encoder,  # Target model to wrap
                module_map = {  # Map: original layer type → wrapped layer type
                    torch.nn.Linear: AutoWrappedLinear,  # Replace Linear with auto-offloading version
                    torch.nn.Embedding: AutoWrappedModule,  # Replace Embedding
                    T5RelativeEmbedding: AutoWrappedModule,  # T5-specific layers
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(  # Configuration for wrapped layers
                    offload_dtype=dtype,  # Store weights on CPU in original dtype
                    offload_device="cpu",  # Offload to CPU
                    onload_dtype=dtype,  # Load to CPU first (then copy to GPU during forward)
                    onload_device="cpu",  # Stage on CPU before GPU
                    computation_dtype=self.torch_dtype,  # Compute in bfloat16
                    computation_device=self.device,  # Compute on GPU
                ),
                vram_limit=vram_limit,  # VRAM budget (if specified)
            )
        # ===== STEP 3: Wrap DiT layers for VRAM management (MOST IMPORTANT) =====
        if self.dit is not None:  # If DiT loaded
            dtype = next(iter(self.dit.parameters())).dtype  # Get DiT's dtype
            device = "cpu" if vram_limit is not None else self.device  # If using VRAM limit, stage on CPU first; otherwise load directly to GPU
            enable_vram_management(  # Wrap DiT layers (largest model, ~14B params)
                self.dit,  # Target: main diffusion transformer
                module_map = {  # Map: original → wrapped types
                    torch.nn.Linear: AutoWrappedLinear,  # Most parameters are in Linear layers
                    torch.nn.Conv3d: AutoWrappedModule,  # 3D convolutions for video
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,  # LayerNorm with auto-casting
                    RMSNorm: AutoWrappedModule,  # RMS normalization
                    torch.nn.Conv2d: AutoWrappedModule,  # 2D convolutions
                    torch.nn.Conv1d: AutoWrappedModule,  # 1D convolutions
                    torch.nn.Embedding: AutoWrappedModule,  # Embeddings
                },
                module_config = dict(  # Config for layers that FIT in VRAM
                    offload_dtype=dtype,  # Store on CPU in original dtype
                    offload_device="cpu",  # Offload to CPU
                    onload_dtype=dtype,  # Load in original dtype
                    onload_device=device,  # Load to CPU or GPU (depending on strategy)
                    computation_dtype=self.torch_dtype,  # Compute in bfloat16
                    computation_device=self.device,  # Compute on GPU
                ),
                max_num_param=num_persistent_param_in_dit,  # Max params to keep in GPU (if using param-count mode)
                overflow_module_config = dict(  # Config for layers that DON'T fit in VRAM (overflow)
                    offload_dtype=dtype,  # Store on CPU
                    offload_device="cpu",
                    onload_dtype=dtype,  # Load to CPU first
                    onload_device="cpu",  # Stage on CPU (slower, but necessary for overflow)
                    computation_dtype=self.torch_dtype,  # Compute in bfloat16
                    computation_device=self.device,  # Compute on GPU
                ),
                vram_limit=vram_limit,  # VRAM budget (if specified)
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    # ========================================================================
    # ⭐ CRITICAL: from_pretrained() - Load Pipeline from Pretrained Models
    # ========================================================================
    # PURPOSE: Create pipeline and load all required models from ModelScope/HF
    # USAGE: pipe = WanVideoPipeline.from_pretrained(model_configs=[...])
    # HOW IT WORKS:
    #   1. Redirect common files (avoid redundant downloads)
    #   2. Initialize empty pipeline
    #   3. Download all models via ModelConfig
    #   4. Load models via ModelManager
    #   5. Fetch models from manager and assign to pipeline
    # ========================================================================
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,  # Precision for model weights
        device: Union[str, torch.device] = "cuda",  # Target device
        model_configs: list[ModelConfig] = [],  # List of models to load (DiT, VAE, audio encoder, etc.)
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),  # T5 tokenizer config
        audio_processor_config: ModelConfig = None,  # wav2vec2 processor config (for S2V)
        redirect_common_files: bool = True,  # Avoid downloading same file from different repos
        use_usp=False,  # Use Unified Sequence Parallel (distributed training)
    ):
        # ===== STEP 1: Redirect common files to avoid redundant downloads =====
        if redirect_common_files:  # If redirection enabled
            redirect_dict = {  # Map: filename → canonical repo
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",  # T5 encoder (shared across models)
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",  # VAE (shared across models)
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",  # CLIP encoder
            }
            for model_config in model_configs:  # For each model to load
                if model_config.origin_file_pattern is None or model_config.model_id is None:  # Skip if not using auto-download
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:  # If this file should be redirected
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]  # Update to canonical repo

        # ===== STEP 2: Initialize empty pipeline =====
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)  # Create pipeline (models are None at this point)
        if use_usp: pipe.initialize_usp()  # Initialize distributed training if enabled

        # ===== STEP 3: Download and load all models via ModelManager =====
        model_manager = ModelManager()  # Create manager (handles model loading and identification)
        for model_config in model_configs:  # For each model config
            model_config.download_if_necessary(use_usp=use_usp)  # Download from ModelScope/HF if needed (sets model_config.path)
            model_manager.load_model(  # Load model from path
                model_config.path,  # Path to model checkpoint
                device=model_config.offload_device or device,  # Load to specified device (or default)
                torch_dtype=model_config.offload_dtype or torch_dtype  # Load in specified dtype (or default)
            )

        # ===== STEP 4: Fetch models from manager and assign to pipeline =====
        # ModelManager automatically identifies model type (text_encoder, dit, vae, etc.)
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")  # Get T5 text encoder
        dit = model_manager.fetch_model("wan_video_dit", index=2)  # Get DiT (index=2 means can have up to 2 DiTs)
        if isinstance(dit, list):  # If multiple DiTs loaded (e.g., Animate mode uses 2 DiTs)
            pipe.dit, pipe.dit2 = dit  # Assign both
        else:  # Only one DiT
            pipe.dit = dit  # Assign to dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")  # Get VAE
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")  # Get CLIP image encoder
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")  # Get motion controller
        vace = model_manager.fetch_model("wan_video_vace", index=2)  # Get VACE (can have up to 2)
        if isinstance(vace, list):  # If multiple VACEs
            pipe.vace, pipe.vace2 = vace  # Assign both
        else:  # Only one VACE
            pipe.vace = vace  # Assign to vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")  # ⭐ Get wav2vec2 audio encoder (for S2V)
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")  # Get animate adapter

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    # ========================================================================
    # ⭐⭐⭐ MOST IMPORTANT: __call__() - Main Inference Method
    # ========================================================================
    # PURPOSE: Generate video from inputs (text, image, audio, etc.)
    # USAGE: video = pipe(prompt="...", input_audio=audio, ...)
    # HOW IT WORKS:
    #   1. Setup scheduler and organize inputs into 3 dicts
    #   2. Run pipeline units (encode text, audio, images)
    #   3. Denoising loop (iteratively denoise latents)
    #   4. VAE decode latents to video frames
    # SUPPORTS MULTIPLE MODES:
    #   - Text-to-Video (T2V): prompt only
    #   - Image-to-Video (I2V): prompt + input_image
    #   - Speech-to-Video (S2V): prompt + input_audio ⭐
    #   - Video-to-Video (V2V): prompt + input_video
    # ========================================================================
    @torch.no_grad()  # Disable gradient computation (inference only)
    def __call__(
        self,
        # ===== TEXT PROMPT =====
        prompt: str,  # Positive text prompt (e.g., "a person talking")
        negative_prompt: Optional[str] = "",  # Negative prompt for CFG (e.g., "blurry, distorted")

        # ===== IMAGE-TO-VIDEO MODE =====
        input_image: Optional[Image.Image] = None,  # Reference image (for I2V mode)

        # ===== FIRST-LAST-FRAME MODE =====
        end_image: Optional[Image.Image] = None,  # Last frame (for interpolation)

        # ===== VIDEO-TO-VIDEO MODE =====
        input_video: Optional[list[Image.Image]] = None,  # Input video frames (for V2V mode)
        denoising_strength: Optional[float] = 1.0,  # How much to denoise (1.0 = full, 0.5 = half)

        # ===== SPEECH-TO-VIDEO MODE ⭐ =====
        input_audio: Optional[np.array] = None,  # Audio waveform (numpy array, 16kHz) ⭐
        audio_embeds: Optional[torch.Tensor] = None,  # Pre-computed audio embeddings (skip wav2vec2 if provided) ⭐
        audio_sample_rate: Optional[int] = 16000,  # Audio sample rate (must be 16kHz for wav2vec2) ⭐
        s2v_pose_video: Optional[list[Image.Image]] = None,  # Pose guidance video (for S2V) ⭐
        s2v_pose_latents: Optional[torch.Tensor] = None,  # Pre-encoded pose latents ⭐
        motion_video: Optional[list[Image.Image]] = None,  # Motion guidance video ⭐
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # ===== STEP 1: Setup Scheduler =====
        # Initialize timestep schedule for denoising process
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)  # Create timestep schedule (e.g., 50 steps from noise to clean)

        # ===== STEP 2: Organize Inputs into Three Dictionaries =====
        # THE THREE DICTS EXPLAINED:
        # - inputs_posi: Positive conditioning (prompt, audio, etc.) → guides generation TOWARD desired content
        # - inputs_nega: Negative conditioning (negative_prompt) → guides generation AWAY from undesired content
        # - inputs_shared: CFG-insensitive inputs (image, video, dimensions) → same for both positive/negative
        #
        # WHY THREE DICTS? Classifier-Free Guidance (CFG) requires:
        #   - One forward pass with positive conditioning
        #   - One forward pass with negative conditioning (or unconditional)
        #   - Combine predictions: output = nega_pred + cfg_scale * (posi_pred - nega_pred)

        inputs_posi = {  # Positive conditioning inputs
            "prompt": prompt,  # Text prompt to guide generation
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,  # TeaCache acceleration params
        }
        inputs_nega = {  # Negative conditioning inputs
            "negative_prompt": negative_prompt,  # Text to avoid (e.g., "blurry")
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,  # TeaCache params
        }
        inputs_shared = {  # CFG-insensitive inputs (shared by both positive and negative)
            "input_image": input_image,  # Reference image (for I2V mode)
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,  # ⭐ S2V inputs (audio waveform goes here)
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,  # Animate mode inputs
        }

        # ===== STEP 3: Run Pipeline Units (Process All Inputs) =====
        # Units execute sequentially, each transforming the three input dicts
        # KEY UNITS FOR S2V:
        #   - WanVideoUnit_ShapeChecker: Validates/adjusts height, width, num_frames
        #   - WanVideoUnit_NoiseInitializer: Creates initial latent noise
        #   - WanVideoUnit_PromptEmbedder: Encodes text prompt → prompt_embeds
        #   - WanVideoUnit_S2V: ⭐ Extracts audio from inputs_shared, encodes to audio_embeds, adds to inputs_posi
        #   - WanVideoUnit_ImageEmbedderVAE/CLIP: Encodes input_image (for I2V)
        #   - ... (other units for controlnet, camera, etc.)
        for unit in self.units:  # Iterate through all pipeline units
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)  # Run unit, update dicts

        # ===== STEP 4: Denoising Loop (Iteratively Remove Noise) =====
        # Load models needed during denoising to GPU
        self.load_models_to_device(self.in_iteration_models)  # Load dit, motion_controller, vace, animate_adapter to GPU (if VRAM management enabled)
        models = {name: getattr(self, name) for name in self.in_iteration_models}  # Create dict: {"dit": self.dit, "motion_controller": self.motion_controller, ...}

        # Denoising loop: iterate through timesteps (e.g., 50 steps from noisy → clean)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):  # For each timestep (progress_id = 0 to num_steps-1)
            # ===== Switch DiT if Necessary (for Animate mode) =====
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:  # If timestep < 0.875 * total and dit2 loaded
                self.load_models_to_device(self.in_iteration_models_2)  # Switch to dit2, vace2
                models["dit"] = self.dit2  # Use dit2 for remaining steps
                models["vace"] = self.vace2  # Use vace2

            # ===== Prepare Timestep Tensor =====
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # Convert scalar to tensor (1,), move to GPU

            # ===== Model Forward Pass (Predict Noise) =====
            # POSITIVE PASS: Use positive conditioning (prompt_embeds, audio_embeds, image_embeds)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)  # DiT predicts noise conditioned on positive inputs. inputs_posi contains audio_embeds (from S2V unit) and prompt_embeds

            # ===== Classifier-Free Guidance (CFG) =====
            if cfg_scale != 1.0:  # If CFG enabled
                if cfg_merge:  # If batch was merged (positive and negative in same batch for efficiency)
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)  # Split batch into positive and negative predictions
                else:  # If separate passes needed
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)  # NEGATIVE PASS: Use negative conditioning (negative_prompt)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)  # CFG formula: output = unconditional + scale * (conditional - unconditional)
            else:  # CFG disabled (cfg_scale=1.0)
                noise_pred = noise_pred_posi  # Use positive prediction directly

            # ===== Scheduler Step (Remove Predicted Noise) =====
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])  # Remove predicted noise from current latents → get next latents (one step closer to clean image)
            if "first_frame_latents" in inputs_shared:  # If I2V mode (first frame should match input_image)
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]  # Replace first frame latents with fixed reference (ensures first frame matches input_image)
        
        # ===== VACE Post-Processing (TODO: remove it) =====
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):  # If VACE or Animate mode
            if vace_reference_image is not None and isinstance(vace_reference_image, list):  # If multiple VACE reference images
                f = len(vace_reference_image)  # Number of reference frames
            else:  # Single reference image
                f = 1  # 1 reference frame
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]  # Remove first f frames (VACE reference frames)

        # ===== STEP 5: Run Post Units (After Denoising, Before Decoding) =====
        for unit in self.post_units:  # Run post-processing units (e.g., WanVideoPostUnit_S2V adds motion context)
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)  # Update inputs_shared with post-processed latents

        # ===== STEP 6: VAE Decode (Latents → Video Frames) =====
        self.load_models_to_device(['vae'])  # Load VAE to GPU (offload DiT if VRAM management enabled)
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)  # Decode latents (1, 16, T, H/16, W/16) → video tensor (1, 3, T, H, W). Tiling reduces VRAM for high-res
        video = self.vae_output_to_video(video)  # Convert tensor (1, 3, T, H, W) → list of PIL.Image frames
        self.load_models_to_device([])  # Offload all models to CPU (free VRAM)

        return video  # Return list of PIL.Image frames (generated video)



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


# ============================================================================
# ⭐⭐⭐ CRITICAL FOR PHASE 2: WanVideoUnit_S2V ⭐⭐⭐
# ============================================================================
# This is THE KEY UNIT for Speech-to-Video (S2V) training!
# It handles audio processing and adds audio conditioning to the DiT model.
#
# WHAT IT DOES:
# 1. Checks if input_audio exists in inputs_shared
# 2. Converts audio waveform → audio_embeds using wav2vec2
# 3. Adds audio_embeds to inputs_posi (positive conditioning)
# 4. Adds zeroed audio_embeds to inputs_nega (for CFG)
# 5. Also processes optional pose video and motion context
#
# HOW IT WORKS IN TRAINING:
# - Training script adds: inputs_shared["input_audio"] = data["audio"]
# - This unit extracts it, processes it, and passes audio_embeds to DiT
# ============================================================================

class WanVideoUnit_S2V(PipelineUnit):
    """
    Speech-to-Video (S2V) Pipeline Unit

    Processes audio for S2V generation by:
    - Converting audio → embeddings via wav2vec2
    - Adding audio conditioning to positive/negative inputs
    - Handling optional pose video and motion context
    """

    def __init__(self):
        super().__init__(
            take_over=True,  # ⭐ IMPORTANT: This unit takes full control of all input dicts
            onload_model_names=("audio_encoder", "vae",)  # Models to load to GPU before processing
        )

    # ------------------------------------------------------------------------
    # METHOD: process_audio - Convert audio waveform → embeddings
    # ------------------------------------------------------------------------
    # PURPOSE: Converts raw audio to audio embeddings using wav2vec2 encoder
    #
    # INPUTS:
    #   - input_audio: Raw audio waveform (numpy array or tensor)
    #   - audio_sample_rate: Sample rate (MUST be 16000 for wav2vec2)
    #   - num_frames: Number of video frames to generate (e.g., 81)
    #   - fps: Frames per second (default: 16)
    #   - audio_embeds: Pre-computed embeddings (optional, for caching)
    #   - return_all: If True, return all clips (for multi-clip generation)
    #
    # OUTPUT:
    #   - {"audio_embeds": tensor} - Audio embeddings for DiT conditioning
    #
    # KEY DETAIL:
    #   - batch_frames = num_frames - 1 (e.g., 80 for 81-frame video)
    #   - This is because the first frame is the reference image
    # ------------------------------------------------------------------------
    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        # If embeddings already computed (cached), return directly
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}

        # Load wav2vec2 audio encoder to GPU
        pipe.load_models_to_device(["audio_encoder"])

        # Convert audio waveform → embeddings using wav2vec2
        # batch_frames = num_frames - 1 because we generate (num_frames - 1) frames per clip
        # Example: num_frames=81 → batch_frames=80
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
            input_audio,           # Raw audio waveform
            audio_sample_rate,     # Sample rate (16kHz)
            pipe.audio_processor,  # wav2vec2 processor
            fps=fps,               # Video FPS (16)
            batch_frames=num_frames-1,  # 80 frames for 81-frame video
            dtype=pipe.torch_dtype,
            device=pipe.device
        )

        # Return based on mode:
        if return_all:
            # Multi-clip mode (inference): return all clips' embeddings as list
            return audio_embeds
        else:
            # Single-clip mode (training): return first clip's embeddings
            return {"audio_embeds": audio_embeds[0]}

    # ------------------------------------------------------------------------
    # METHOD: process_motion_latents - Encode motion context video
    # ------------------------------------------------------------------------
    # PURPOSE: Encodes 73-frame motion video from previous clip (for multi-clip generation)
    #
    # INPUTS (passed by caller in process() method):
    #   - pipe: WanVideoPipeline instance (to access vae, device, etc.)
    #   - height, width: Video dimensions (e.g., 480, 832)
    #   - tiled, tile_size, tile_stride: VAE tiling params (to reduce VRAM)
    #   - motion_video: List of 73 PIL.Image frames from previous clip (optional)
    #
    # OUTPUT:
    #   - {"motion_latents": tensor, "drop_motion_frames": bool}
    #
    # WHEN USED:
    #   - First clip: motion_video=None → Create zero latents, drop_motion_frames=True
    #   - Subsequent clips: motion_video=last 73 frames → Encode, drop_motion_frames=False
    # ------------------------------------------------------------------------
    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        # Load VAE to GPU (needed for encoding)
        pipe.load_models_to_device(["vae"])

        # Motion context always uses 73 frames (hardcoded design choice)
        motion_frames = 73

        # Prepare output dictionary
        kwargs = {}

        # CASE 1: Motion video provided (subsequent clips in multi-clip generation)
        if motion_video is not None and len(motion_video) > 0:
            # Validate: Must be exactly 73 frames
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"

            # Preprocess: List[PIL.Image] → tensor [1, 3, 73, height, width]
            motion_latents = pipe.preprocess_video(motion_video)

            # Flag: Use motion latents (don't drop)
            kwargs["drop_motion_frames"] = False

        # CASE 2: No motion video (first clip in generation)
        else:
            # Create zero tensor as placeholder [1, 3, 73, height, width]
            motion_latents = torch.zeros(
                [1, 3, motion_frames, height, width],
                dtype=pipe.torch_dtype,  # e.g., bfloat16
                device=pipe.device        # e.g., "cuda"
            )

            # Flag: Drop motion latents (don't use in DiT)
            kwargs["drop_motion_frames"] = True

        # VAE encode: [1, 3, 73, H, W] → [1, 16, 73, H/16, W/16]
        motion_latents = pipe.vae.encode(
            motion_latents,
            device=pipe.device,
            tiled=tiled,              # Use tiled encoding to save VRAM
            tile_size=tile_size,      # e.g., (30, 52)
            tile_stride=tile_stride   # e.g., (15, 26)
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        # Add encoded latents to output dict
        kwargs.update({"motion_latents": motion_latents})

        # Return: {"motion_latents": tensor, "drop_motion_frames": bool}
        return kwargs

    # ------------------------------------------------------------------------
    # METHOD: process_pose_cond - Encode pose guidance video
    # ------------------------------------------------------------------------
    # PURPOSE: Encodes pose video (e.g., skeleton/dwpose) to guide character motion
    #
    # INPUTS (passed by caller in process() method):
    #   - pipe: WanVideoPipeline instance
    #   - s2v_pose_video: List of PIL.Image frames showing pose (e.g., stick figures)
    #   - num_frames: Target video frames (e.g., 81)
    #   - height, width: Video dimensions
    #   - tiled, tile_size, tile_stride: VAE tiling params
    #   - s2v_pose_latents: Pre-computed pose latents (optional, for caching)
    #   - num_repeats: Number of clips (for multi-clip generation)
    #   - return_all: If True, return all clips' pose latents (multi-clip mode)
    #
    # OUTPUT:
    #   - {"s2v_pose_latents": tensor} or list of tensors (if return_all=True)
    #
    # WHAT IT DOES:
    #   - Encodes pose video to latents that DiT adds to generated latents
    #   - DiT formula: generated = patch_embedding(latents) + cond_encoder(pose_latents)
    #   - This makes generated video follow the pose guidance
    # ------------------------------------------------------------------------
    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        # CASE 1: Pre-computed pose latents provided (cached)
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}

        # CASE 2: No pose video provided (no pose guidance)
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}

        # Load VAE to GPU (needed for encoding)
        pipe.load_models_to_device(["vae"])

        # Calculate number of frames to generate (excludes reference frame)
        # Example: num_frames=81 → infer_frames=80
        infer_frames = num_frames - 1

        # Preprocess pose video: List[PIL.Image] → tensor [1, 3, T, H, W]
        # Then slice to needed frames: [:, :, :infer_frames * num_repeats]
        # Example: For 2 clips (num_repeats=2), need 80*2=160 pose frames
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]

        # Pad with -1.0 if not enough frames provided
        # Example: If only 100 frames but need 160, pad 60 frames with -1.0
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([
            input_video,
            -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)
        ], dim=2)

        # Split into num_repeats clips along time dimension
        # Example: [1, 3, 160, H, W] → 2 chunks of [1, 3, 80, H, W]
        input_videos = input_video.chunk(num_repeats, dim=2)

        # Process each clip separately
        pose_conds = []
        for r in range(num_repeats):
            # Get current clip: [1, 3, 80, H, W]
            cond = input_videos[r]

            # Duplicate first frame at the beginning: [first_frame, clip]
            # Why? To match the structure: reference frame + generated frames
            # Result: [1, 3, 81, H, W] (first frame repeated, then 80 pose frames)
            cond = torch.cat([
                cond[:, :, 0:1].repeat(1, 1, 1, 1, 1),  # First frame repeated once
                cond                                     # Original 80 frames
            ], dim=2)

            # VAE encode: [1, 3, 81, H, W] → [1, 16, 81, H/16, W/16]
            cond_latents = pipe.vae.encode(
                cond,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            # Remove first frame latents (only need 80 pose frames for conditioning)
            # [:,:,1:] → [1, 16, 80, H/16, W/16]
            pose_conds.append(cond_latents[:, :, 1:])

        # Return based on mode
        if return_all:
            # Multi-clip inference: return list of all clips' pose latents
            return pose_conds
        else:
            # Single-clip (training): return first clip's pose latents
            return {"s2v_pose_latents": pose_conds[0]}

    # ------------------------------------------------------------------------
    # ⭐⭐⭐ METHOD: process() - THE MAIN PROCESSING LOGIC ⭐⭐⭐
    # ------------------------------------------------------------------------
    # PURPOSE: Main entry point called during unit execution
    #          This is WHERE audio gets processed and added to DiT conditioning!
    #
    # FLOW (THIS IS THE KEY TO UNDERSTANDING S2V TRAINING):
    #   1. Check if input_audio exists in inputs_shared
    #   2. If no audio OR no audio_encoder → skip (return unchanged)
    #   3. Extract audio, pose, motion from inputs_shared
    #   4. Process audio → audio_embeds
    #   5. Add audio_embeds to inputs_posi (positive conditioning)
    #   6. Add 0.0 * audio_embeds to inputs_nega (negative for CFG)
    #   7. Process motion latents and pose conditioning
    #   8. Return updated dictionaries
    #
    # TRAINING FLOW:
    #   - Training script: inputs_shared["input_audio"] = data["audio"]
    #   - This method: extracts input_audio, processes it
    #   - Result: inputs_posi["audio_embeds"] = <embeddings>
    #   - DiT receives audio_embeds as conditioning during forward pass
    # ------------------------------------------------------------------------
    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        # ===== STEP 1: CHECK IF AUDIO PROCESSING SHOULD HAPPEN =====
        # Skip if:
        #   - No input_audio AND no pre-computed audio_embeds in inputs_shared
        #   - OR audio_encoder not loaded
        #   - OR audio_processor not loaded
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega  # Skip S2V processing

        # ===== STEP 2: EXTRACT VIDEO GENERATION PARAMETERS FROM inputs_shared =====
        # These are needed for VAE encoding
        num_frames = inputs_shared.get("num_frames")      # e.g., 81
        height = inputs_shared.get("height")              # e.g., 448
        width = inputs_shared.get("width")                # e.g., 832
        tiled = inputs_shared.get("tiled")                # Use tiled VAE?
        tile_size = inputs_shared.get("tile_size")        # Tile size
        tile_stride = inputs_shared.get("tile_stride")    # Tile stride

        # ===== STEP 3: EXTRACT AND REMOVE AUDIO/POSE INPUTS FROM inputs_shared =====
        # NOTE: .pop() removes the key from dict (so later units don't see it)
        input_audio = inputs_shared.pop("input_audio")           # Raw audio waveform
        audio_embeds = inputs_shared.pop("audio_embeds")         # Or pre-computed embeds
        audio_sample_rate = inputs_shared.get("audio_sample_rate")  # Keep this (don't pop)

        s2v_pose_video = inputs_shared.pop("s2v_pose_video")     # Optional pose video
        s2v_pose_latents = inputs_shared.pop("s2v_pose_latents") # Or pre-computed pose
        motion_video = inputs_shared.pop("motion_video")         # 73-frame motion context

        # ===== STEP 4: CONVERT AUDIO → EMBEDDINGS =====
        # This is the KEY transformation: audio waveform → embeddings
        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        # audio_input_positive = {"audio_embeds": <tensor shape [batch, seq_len, hidden_dim]>}

        # ===== STEP 5: ADD AUDIO EMBEDDINGS TO POSITIVE CONDITIONING =====
        # ⭐ THIS IS WHERE AUDIO CONDITIONING IS ADDED TO THE DiT MODEL! ⭐
        inputs_posi.update(audio_input_positive)  # inputs_posi["audio_embeds"] = <tensor>

        # ===== STEP 6: ADD ZEROED AUDIO TO NEGATIVE CONDITIONING (CFG) =====
        # For classifier-free guidance: negative condition = no audio
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        # ===== STEP 7: PROCESS MOTION CONTEXT (73 frames from previous clip) =====
        # Adds: inputs_shared["motion_latents"], inputs_shared["drop_motion_frames"]
        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))

        # ===== STEP 8: PROCESS OPTIONAL POSE CONDITIONING =====
        # Adds: inputs_shared["s2v_pose_latents"] (or None if no pose video)
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))

        # ===== STEP 9: RETURN UPDATED DICTIONARIES =====
        # Now:
        #   - inputs_posi has audio_embeds
        #   - inputs_nega has zeroed audio_embeds
        #   - inputs_shared has motion_latents, s2v_pose_latents
        # DiT will receive these as conditioning in the forward pass!
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        """
        UTILITY METHOD: Pre-calculate audio/pose embeddings for multi-clip generation

        Used in INFERENCE (not training) to:
        - Process long audio into multiple clips
        - Process long pose video into multiple clips
        - Return embeddings for all clips

        This allows generating long videos by processing clips sequentially.
        """
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."

        # Validate and fix dimensions
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]

        # Create S2V unit instance
        unit = WanVideoUnit_S2V()

        # Process audio (return all clips)
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)

        # Process pose (return all clips, matching number of audio clips)
        pose_latents = unit.process_pose_cond(
            pipe, s2v_pose_video, num_frames, height, width,
            num_repeats=len(audio_embeds),  # Match number of audio clips
            return_all=True,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride
        )
        pose_latents = None if s2v_pose_video is None else pose_latents

        return audio_embeds, pose_latents, len(audio_embeds)


# ============================================================================
# KEY TAKEAWAYS FOR S2V TRAINING (SUMMARY OF WanVideoUnit_S2V)
# ============================================================================
"""
✅ WHAT YOU LEARNED ABOUT WanVideoUnit_S2V:

1. **The Unit's Role:**
   - Takes over all three input dicts (take_over=True)
   - Processes audio → embeddings using wav2vec2
   - Adds audio conditioning to DiT model

2. **The Data Flow:**
   CSV → data["audio"] → inputs_shared["input_audio"] →
   WanVideoUnit_S2V.process() → audio_embeds →
   inputs_posi["audio_embeds"] → DiT receives as conditioning

3. **Key Methods:**
   - process_audio(): audio waveform → embeddings
   - process_motion_latents(): handles 73-frame motion context
   - process_pose_cond(): handles optional pose video
   - process(): main logic (extracts audio from inputs_shared, adds embeds to inputs_posi)

4. **Training Requirements:**
   - CSV must have "audio" column with path to audio file
   - Audio must be 16kHz sample rate
   - Training script must use: --extra_inputs "input_image,input_audio,audio_sample_rate,..."
   - This makes forward_preprocess() add audio to inputs_shared

5. **Pose Video:**
   - OPTIONAL (can be None)
   - If provided, adds additional pose guidance

Next: Phase 3 - Understand how CSV data is loaded and transformed!
"""


# ============================================================================
# WanVideoPostUnit_S2V - Post-processing for S2V
# ============================================================================
# This unit runs AFTER denoising, BEFORE VAE decoding
# It adds motion context from previous clip for temporal consistency
# ============================================================================

class WanVideoPostUnit_S2V(PipelineUnit):
    """
    S2V Post-processing Unit

    Runs after DiT denoising, before VAE decoding.
    Adds motion context (73 frames) from previous clip to generated latents.

    For multi-clip generation:
    - First clip: No motion latents (drop_motion_frames=True)
    - Subsequent clips: Prepend 73 motion frames from previous clip
    """

    def __init__(self):
        super().__init__(
            input_params=("latents", "motion_latents", "drop_motion_frames")
        )

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        """
        Add motion context to generated latents

        Args:
            latents: Newly generated latents from DiT (shape: [B, C, T, H, W])
            motion_latents: Latents from previous clip's last 73 frames
            drop_motion_frames: If True, don't use motion latents

        Returns:
            dict with updated "latents" (or empty dict if skipping)

        Process:
        - If not S2V mode OR no motion_latents OR drop_motion_frames=True → skip
        - Otherwise: latents = [motion_latents, new_latents[1:]]
          (Skip first frame of new latents to avoid duplication)
        """
        # Skip if:
        #   - No audio_encoder (not S2V mode)
        #   - No motion_latents
        #   - drop_motion_frames=True (first clip, no motion context)
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}  # Don't modify latents

        # Concatenate motion frames with new latents (skip first new frame)
        # [motion_latents (73 frames), new_latents[1:] (80 frames)] = 153 frames total
        # (But we only use the last 80 frames after this)
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)

        return {"latents": latents}


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"))

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return {}
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
