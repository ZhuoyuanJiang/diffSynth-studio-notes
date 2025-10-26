#!/bin/bash
#
# Wan2.2-S2V-14B LoRA Training on RenderMe360 Dataset (SMALL-SCALE TEST)
# Single-view input (cam_54) → Multi-view output (2×2 grid) with audio
#
# Purpose: Validate training setup with 50 samples for 1 epoch
# FIXED: Added NCCL environment variables to handle NUMA topology and timeout
#

# Activate conda environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Set cache directories to /ssd2 to avoid home directory quota (100GB limit)
export TRITON_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
export DEEPSPEED_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache

# NCCL Configuration for NUMA topology and debugging
export NCCL_DEBUG=INFO                      # Enable NCCL debug output
export NCCL_IB_DISABLE=1                    # Disable InfiniBand (not available)
export NCCL_P2P_DISABLE=0                   # Enable P2P (peer-to-peer) if available
export NCCL_SHM_DISABLE=0                   # Enable shared memory
export NCCL_SOCKET_IFNAME=^docker,lo        # Use all interfaces except docker/loopback
export NCCL_TIMEOUT=1800                    # Increase timeout to 30 minutes (in seconds)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Better error messages
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Detailed distributed debugging

# Point to CUDA 11.8 to match PyTorch
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Verify environment
echo "========================================="
echo "Environment Check"
echo "========================================="
echo "HF_HOME: ${HF_HOME:-Not set}"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "DEEPSPEED_CACHE_DIR: $DEEPSPEED_CACHE_DIR"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds"
echo "CUDA PATH: $(which nvcc)"
echo ""

echo "========================================="
echo "SMALL-SCALE VALIDATION TEST (FIXED)"
echo "50 samples, 1 epoch"
echo "NCCL timeout: 30 minutes"
echo "========================================="

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
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
  --output_path "/ssd2/zhuoyuan/diffsynth_training/Wan2.2-S2V-14B_RenderMe360_lora_TEST" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio,audio_sample_rate" \
  --use_gradient_checkpointing_offload \
  --save_steps 25

echo ""
echo "========================================="
echo "Test complete! Check for:"
echo "1. LoRA modules matched (should see 'Found X potential target modules', X > 0)"
echo "2. Loss decreases over 50 steps"
echo "3. VRAM < 45GB per GPU"
echo "4. No OOM errors"
echo "========================================="
