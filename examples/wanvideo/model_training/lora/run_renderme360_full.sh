#!/bin/bash
#
# RenderMe360 Training - Full Training Run
# Purpose: Train Wan2.2-S2V-14B LoRA on complete RenderMe360 dataset
#
# Hardware: 8× RTX 6000 Ada (48GB each)
# Dataset: 1,048 samples × 10 repeats = 10,480 samples/epoch
# Training: 50 epochs with DeepSpeed ZeRO-2
#

set -e

echo "========================================"
echo "RenderMe360 Training - FULL TRAINING"
echo "========================================"

# Activate conda environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Server-specific paths (vllab9)
DATASET_BASE="/ssd2/zhuoyuan/renderme360_4cam"
METADATA_PATH="${DATASET_BASE}/metadata_single2multi.csv"
OUTPUT_DIR="/ssd2/zhuoyuan/diffsynth_training/renderme360_lora"

# Verify dataset and metadata
if [ ! -d "$DATASET_BASE" ]; then
    echo "ERROR: Dataset not found at $DATASET_BASE"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "ERROR: Metadata not found at $METADATA_PATH"
    exit 1
fi

# Check GPU availability (need 8 GPUs)
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

if [ "$GPU_COUNT" -lt 8 ]; then
    echo ""
    echo "WARNING: Only $GPU_COUNT GPUs detected. This script expects 8 GPUs."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Count samples in metadata
SAMPLE_COUNT=$(tail -n +2 "$METADATA_PATH" | wc -l)
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_BASE"
echo "  Metadata: $METADATA_PATH"
echo "  Samples: $SAMPLE_COUNT"
echo "  Dataset repeat: 10 (effective: $((SAMPLE_COUNT * 10)) samples/epoch)"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: 8 (DeepSpeed ZeRO-2)"
echo "  Epochs: 50"
echo "  Learning rate: 1e-4"
echo "  LoRA rank: 32"
echo "  Checkpointing: Every 10,000 steps"
echo ""
echo "Expected training time: ~24-48 hours (depends on GPU utilization)"
echo ""
read -p "Press Enter to start training (Ctrl+C to cancel)..."

# Run training with 8 GPUs using custom ZeRO-3 config for memory efficiency
accelerate launch \
  --config_file examples/wanvideo/model_training/lora/accelerate_config_renderme360.yaml \
  examples/wanvideo/model_training/train_renderme360.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$METADATA_PATH" \
  --height 448 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio" \
  --use_gradient_checkpointing_offload \
  --save_steps 10000

echo ""
echo "========================================"
echo "Training completed!"
echo "LoRA checkpoints saved at: $OUTPUT_DIR"
echo ""
echo "To validate trained model:"
echo "  python examples/wanvideo/model_training/validate_lora/Wan2.2-S2V-14B.py \\"
echo "    --lora_checkpoint $OUTPUT_DIR/checkpoint-XXXXX"
echo "========================================"
