#!/bin/bash
#
# Wan2.2-S2V-14B LoRA Training on RenderMe360 Dataset (4-GPU VALIDATION TEST)
# Single-view input (cam_54) → Multi-view output (2×2 grid) with audio
#
# Purpose: Validate dataset/model/training code with 50 samples for 1 epoch
# Strategy: Use 4 GPUs (single NUMA node) to avoid cross-NUMA communication issues
#

# Activate conda environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Set cache directories to /ssd2 to avoid home directory quota (100GB limit)
export TRITON_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
export DEEPSPEED_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache

# Point to CUDA 11.8 to match PyTorch
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Use only GPUs 0-3 (single NUMA node 0)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Verify environment
echo "========================================="
echo "Environment Check"
echo "========================================="
echo "HF_HOME: ${HF_HOME:-Not set}"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "DEEPSPEED_CACHE_DIR: $DEEPSPEED_CACHE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA PATH: $(which nvcc)"
echo ""

echo "========================================="
echo "4-GPU VALIDATION TEST"
echo "50 samples, 1 epoch"
echo "Using GPUs 0-3 (NUMA node 0 only)"
echo "Gradient accumulation: 2 steps"
echo "========================================="

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_4GPU.yaml \
  examples/wanvideo/model_training/train_s2v.py \
  --dataset_base_path /ssd2/zhuoyuan/renderme360_4cam \
  --dataset_metadata_path /ssd2/zhuoyuan/renderme360_4cam/metadata_test.csv \
  --height 448 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/ssd2/zhuoyuan/diffsynth_training/Wan2.2-S2V-14B_RenderMe360_lora_TEST_4GPU" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio,audio_sample_rate" \
  --use_gradient_checkpointing_offload \
  --save_steps 25

echo ""
echo "========================================="
echo "4-GPU Test complete! Check for:"
echo "1. LoRA modules matched (should see 'Found 966 potential target modules')"
echo "2. Training started without timeout"
echo "3. Loss values appear and decrease"
echo "4. No OOM errors"
echo "5. VRAM < 47GB per GPU"
echo ""
echo "If successful → Dataset/model/code are ALL VALID"
echo "Then we know 8-GPU issue is purely distributed communication"
echo "========================================="
