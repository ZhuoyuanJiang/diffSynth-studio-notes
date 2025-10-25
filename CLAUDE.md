# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffSynth-Studio is a Diffusion model engine for image and video generation, developed by the ModelScope team. It supports multiple SOTA diffusion models including FLUX, Qwen-Image, Wan Video series, and more. The project focuses on aggressive technical exploration and cutting-edge model capabilities.

## Current Project Goal

Fine-tuning Wan2.2-S2V-14B (Speech-to-Video) on the RenderMe360 dataset.

**Dataset:** RenderMe360 - multi-view human performance capture dataset
**Task:** Train the model to generate talking head videos conditioned on audio input
**Approach:** (To be determined)

**Reference files:**
- `TRAINING_PLAN_WAN_S2V_RENDERME360.md` - Training plan and dataset details
- `LEARNING_PLAN_WAN_S2V.md` - Learning plan for understanding the codebase


## Installation & Setup

**Install from source (recommended):**
```bash
pip install -e .
```

**Important:** HuggingFace cache is configured at `/ssd1/zhuoyuan/hf_cache` to avoid filling the home directory quota (100GB limit on NAS). Large model checkpoints should be stored on local drives (/ssd1/, /ssd2/, etc.) for faster access.

## Core Architecture

### Pipeline System
The codebase uses a modular pipeline architecture where models are composed into pipelines:

- **Pipelines** (`diffsynth/pipelines/`): High-level interfaces combining models for specific tasks
  - Image pipelines: `FluxImagePipeline`, `QwenImagePipeline`, `SDImagePipeline`, `SDXLImagePipeline`, etc.
  - Video pipelines: `WanVideoPipeline`, `HunyuanVideoPipeline`, `CogVideoPipeline`, `StepVideoPipeline`, etc.
  - All pipelines follow pattern: `from_pretrained()` → configure units → inference/training

- **Models** (`diffsynth/models/`): Individual model components (transformers, VAEs, text encoders)
  - Each model has a `state_dict_converter()` for loading from different formats (civitai, diffusers)
  - Models are loaded via `model_manager.py` which handles downloading and conversion

- **Pipeline Units**: Pipelines are composed of units that process inputs sequentially
  - Units handle text encoding, image encoding, denoising, decoding
  - Training mode switches specific units to trainable state

### Model Loading System
Models are loaded using `ModelConfig` which supports:
- ModelScope/HuggingFace model IDs with origin file patterns
- Local file paths
- Automatic downloading via `download_models()` and `download_customized_models()`

Example:
```python
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        # ... more configs
    ],
)
```

### Training Framework
Training is handled by `diffsynth/trainers/`:

- **Training Modules** inherit from `DiffusionTrainingModule`:
  - `FluxTrainingModule`, `QwenImageTrainingModule`, etc.
  - Each implements `forward_preprocess()` and `forward()` for training loop

- **Dataset System** (`UnifiedDataset`): Flexible dataset loader supporting:
  - CSV metadata with base paths
  - Dynamic resolution or fixed resolution
  - Custom data operators for different modalities
  - Multi-key data loading (image, controlnet inputs, masks, etc.)

- **LoRA Training**: Supports LoRA training via PEFT library
  - Specify `lora_base_model` (e.g., "dit") and `lora_target_modules`
  - LoRA checkpoints saved with `align_to_opensource_format` for compatibility

- **Training Launch**: Uses Accelerate for distributed training
  ```bash
  accelerate launch examples/{model}/model_training/train.py --dataset_base_path ...
  ```

### Key Training Parameters
- `--model_id_with_origin_paths`: Colon-separated model_id:file_pattern pairs
- `--lora_base_model`: Which component to add LoRA to (e.g., "dit")
- `--lora_target_modules`: Comma-separated module names for LoRA injection
- `--use_gradient_checkpointing`: Enable to reduce memory usage
- `--dataset_repeat`: Repeat dataset N times per epoch
- `--max_pixels`: Maximum image resolution (e.g., 1048576 for 1024x1024)

## Common Development Tasks

### Running Inference
Examples are organized by model in `examples/`:
- `examples/flux/model_inference/` - FLUX model inference scripts
- `examples/qwen_image/model_inference/` - Qwen-Image inference scripts
- `examples/wanvideo/model_inference/` - Wan Video inference scripts

Low VRAM variants available in `model_inference_low_vram/` subdirectories.

### Training Models
Training scripts in `examples/{model}/model_training/`:
- `lora/` - LoRA training scripts (.sh files)
- `full/` - Full model training scripts
- `validate_lora/` - Validation after LoRA training
- `validate_full/` - Validation after full training
- `train.py` - Main training script

### Adding New Models
1. Create model class in `diffsynth/models/` with `state_dict_converter()`
2. Create pipeline in `diffsynth/pipelines/`
3. Create training module if needed
4. Add examples in `examples/{model_name}/`

### VRAM Management
The codebase includes advanced VRAM management:
- Layer-by-layer offloading: `.enable_vram_management()` on pipelines
- FP8 quantization support for some models
- Tiled processing for high-resolution generation
- Gradient checkpointing for training

## Project Structure

```
diffsynth/
├── configs/          # Model configuration system
├── controlnets/      # ControlNet implementations
├── data/             # Dataset loaders
├── distributed/      # Distributed training utilities (xDiT)
├── extensions/       # Extensions (ESRGAN, RIFE, FastBlend, ImageQualityMetric)
├── lora/             # LoRA utilities
├── models/           # Model implementations
├── pipelines/        # High-level pipelines
├── processors/       # Image/video processors
├── prompters/        # Prompt processing
├── schedulers/       # Noise schedulers
├── trainers/         # Training framework
├── utils/            # Utility functions
└── vram_management/  # VRAM optimization

examples/
├── flux/             # FLUX model examples
├── qwen_image/       # Qwen-Image examples
├── wanvideo/         # Wan Video examples
├── HunyuanVideo/     # Hunyuan Video examples
├── train/            # Training examples (legacy structure)
└── ...               # Other model examples

apps/
├── gradio/           # Gradio web interfaces
└── streamlit/        # Streamlit web interfaces
```

## Important Notes

- **HuggingFace Cache**: Always verify HF_HOME is set to `/ssd1/zhuoyuan/hf_cache` before operations that download models
- **Storage**: Use local SSDs (/ssd1/, /ssd2/, etc.) for datasets and large files, not /home/ (NAS with 100GB quota)
- **Tokenizers**: `TOKENIZERS_PARALLELISM=false` is set in training scripts to avoid warnings
- **Gradient Checkpointing**: Most training uses gradient checkpointing to fit in GPU memory
- **Model Formats**: The codebase can load from CivitAI, Diffusers, and custom formats via state_dict converters
- **Pipeline Units**: When modifying pipelines, units process inputs sequentially - maintain this pattern
